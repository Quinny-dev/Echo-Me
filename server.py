# Server application for sign language recognition
import socket
from pathlib import Path
from sys import path, argv

path.insert(1, './tools')
import common, encrypt
from holistic import normalize_features

from multiprocessing import Queue, Process, Manager
import threading
from ctypes import c_char_p
from queue import Empty
import atexit
from math import ceil
import numpy as np
import time

# Configuration flags
DEBUG = True
LOG = False
ENCRYPT = True
GPU = True

# Server network configuration
SERVER_ADDR = ("0.0.0.0", common.SERVER_RECV_PORT)
BLACKLIST_ADDRS = [('192.168.1.68', 9999)] # local router heartbeat thing

# Path and model configuration
CURRENT_WORKING_DIRECTORY = Path().absolute()

# Get the latest model from models directory
DEFAULT_MODEL = list((CURRENT_WORKING_DIRECTORY/'models').iterdir())[-1]

# Load sign language labels
LABELS = common.get_labels('data/')

# Processing parameters
PRINT_FREQ = 30  # Frequency for printing results
PRED_FREQ = 5  # Frequency for predictions
MAX_QUEUE_LEN = 25  # Maximum queue size
CONFIDENCE_THRESHOLD = 0.6  # Minimum confidence for predictions
POLL_INTERVAL = 30  # Interval for connection polling

class MissingModelException(Exception):
    pass

# Convert prediction array to class label
def array_to_class(out, addr, connected):
    # Get highest probability prediction
    prediction = np.argmax(out)

    # Check if prediction meets confidence threshold
    if out[prediction] > CONFIDENCE_THRESHOLD:
        print(f"{LABELS[prediction]} {out[prediction]*100} - {addr}")
        tag = LABELS[prediction]

        # Handle unconnected client
        if not connected:
            tag = "None" if tag is None else tag
            return encrypt.encrypt_chacha(tag) if ENCRYPT else tag.encode()

        # Return encrypted prediction for connected client
        if tag is not None:
            ret_val = encrypt.encrypt_chacha(tag) if ENCRYPT else tag.encode()
            return ret_val
    else:
        print("None ({} {}% Below threshold)".format(
            LABELS[prediction], out[prediction]*100))

# Handler for receiving and processing landmark data from clients
class LandmarkReceiver(common.UDPRequestHandler):
    def __init__(self, **kwargs):
        super().__init__()
        self.__dict__.update(kwargs)
        self.CLIENT_TIMEOUT = 30 # time allowed between messages
        # Dictionary mappings for client management
        self.client_to_process = {}  # Client to process mapping
        self.manager = Manager()
        self.client_to_last_msg = self.manager.dict()  # Track last message time
        self.client_to_f_q = {}  # Feature queues per client
        self.client_to_p_q = {}  # Prediction queues per client
        self.poll_connections()
        self.cleaning_process = None
        self.client_to_connected = {}  # Connection status per client

    # Decorator for creating periodic tasks
    def periodic_task(interval, times = -1):
        def outer_wrap(function):
            def wrap(*args, **kwargs):
                stop = threading.Event()
                def inner_wrap():
                    i = 0
                    while i != times and not stop.isSet():
                        stop.wait(interval)
                        function(*args, **kwargs)
                        i += 1

                t = threading.Timer(0, inner_wrap)
                t.daemon = True
                t.start()
                return stop
            return wrap
        return outer_wrap

    # Clean up resources for disconnected client
    def cleanup_client(self, addr):
        common.print_debug_banner(f"CLEANING UP CLIENT: {addr}")
        self.cleaning_process = addr
        # Remove client from all tracking dictionaries
        del self.client_to_f_q[addr]
        del self.client_to_p_q[addr]
        del self.client_to_last_msg[addr]
        del self.client_to_connected[addr]

        # Terminate the client's prediction process
        process_to_del = self.client_to_process[addr]
        process_to_del.terminate()

        common.print_debug_banner("FINISHED TERMINATING")
        # process_to_del.close()
        # common.print_debug_banner("FINISHED CLOSING")
        del self.client_to_process[addr]

        common.print_debug_banner(f"FINISHED CLEANUP")
        print(f"CURRENT PROCESS COUNT: {len(self.client_to_process.keys())}")
        self.cleaning_process = None
    
    # Periodically check for inactive clients
    @periodic_task(POLL_INTERVAL)
    def poll_connections(self):
        common.print_debug_banner(f"POLLING CONNECTIONS")
        print(f"CURRENT PROCESS COUNT: {len(self.client_to_process.keys())}")
        # Clean up clients that haven't sent messages recently
        for client, last_msg_ts in self.client_to_last_msg.items():
            if time.time() - last_msg_ts > self.CLIENT_TIMEOUT:
                common.print_debug_banner(f"FOUND OVERTIME CLIENT: {client}")
                self.cleanup_client(client)

    # Start a new prediction process for a client
    def start_process(self, addr):
        # Create queues for feature and prediction data
        f_q = Queue(MAX_QUEUE_LEN)
        p_q = Queue(MAX_QUEUE_LEN)
        self.client_to_f_q[addr] = f_q
        self.client_to_p_q[addr] = p_q
        self.client_to_connected[addr] = False
        self.client_to_last_msg[addr] = time.time()
        # Create prediction process
        predict = Process(
            target=predict_loop,
            args=(
                model_path,
                f_q,
                p_q,
            )
        )
        self.client_to_process[addr] = predict
        atexit.register(common.exit_handler, predict)
        predict.daemon = True
        predict.start()
        print(f"started new predict process for {addr}")

    # Handle incoming UDP datagrams from clients
    def datagram_received(self, data, addr):
        if addr is None:
            return

        # Block blacklisted addresses
        if addr in BLACKLIST_ADDRS:
            common.print_debug_banner(f"BLOCKED {addr}")
            return

        # Start new process for new client
        if addr not in self.client_to_f_q and addr != self.cleaning_process:
            self.start_process(addr)
            return

        # Update last message timestamp
        self.client_to_last_msg[addr] = time.time()

        # Process incoming data
        try:
            # Decrypt data if encryption is enabled
            if ENCRYPT:
                data = encrypt.decrypt_chacha(data)
            # Handle control messages
            if len(data) < 4:
                if data == "END":
                    # Client disconnection signal
                    common.print_debug_banner(f"RECEIVED 'END' FROM {addr}")
                    self.client_to_f_q[addr].put("END")
                    self.cleanup_client(addr)
                elif data == "ACK":
                    # Client acknowledgment signal
                    common.print_debug_banner(f"RECEIVED 'ACK' FROM {addr}")
                    self.client_to_connected[addr] = True
                return

            # Process landmark data
            landmark_arr = np.array([float(i.strip()) for i in data.split(",")])
            normalized_data = normalize_features(landmark_arr)
            
            # Add normalized data to feature queue
            self.client_to_f_q[addr].put_nowait(normalized_data)

            # Get prediction and send to client
            pred = self.client_to_p_q[addr].get_nowait()
            tag = array_to_class(pred, addr, self.client_to_connected[addr])
            self.transport.sendto(tag, addr)

        except encrypt.DecryptionError:
            print(f"tried to decrypt {data}")
        except Exception as e:
            # print(e)
            pass


# Prediction loop process for a single client
def predict_loop(model_path, f_q, p_q):
    # Configure GPU/CPU usage
    if not GPU:
        import os
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    import tensorflow as tf
    import keras
    from train import TIMESTEPS, init_gpu

    # Set up logging if enabled
    if LOG:
        import timeit
        import logging

        LOG_FILE_NAME = "logs/predict_log"
        logging.basicConfig(
            level=logging.DEBUG,
            filemode="a+",
            filename=LOG_FILE_NAME,
            format="%(message)s"
        )
        if GPU:
            logging.info(f"\n-----USING GPU------")
        else:
            logging.info(f"\n-----USING CPU------")
        
        times = []
        time_count = 0
        TIME_FREQ = 60
    
    # Sliding window function for temporal data
    def slide(w, new):
        # Discard oldest frame and append new frame to data window
        w[:-1] = w[1:]
        w[-1] = new
        return w

    # Initialize GPU and load model
    if GPU:
        init_gpu()
    model = keras.models.load_model(model_path)

    # Initialize processing variables
    delay = 0
    window = None  # Sliding window for temporal features
    results = None  # Buffer for smoothing predictions
    results_len = ceil(PRINT_FREQ / PRED_FREQ)

    if DEBUG:
        common.print_debug_banner("STARTED PREDICTION")

    # Main prediction loop
    while True:
        # Get feature data from queue
        row = f_q.get()

        # Check for termination signal
        if len(row) == 3 and row == "END":
            break
        
        # Initialize window on first data
        if window is None:
            window = np.zeros((TIMESTEPS, len(row)))

        # Update sliding window
        window = slide(window, row)
        
        # Make prediction at specified frequency
        if delay >= PRED_FREQ:
            # Get model prediction
            out = model(np.array([window]))

            # Initialize results buffer
            if results is None:
                results = np.zeros((results_len, len(LABELS)))
            
            # Smooth predictions over time
            results = slide(results, out)
            pred = np.mean(results, axis=0)
            p_q.put(pred)

            delay = 0
    
        delay += 1
    
    common.print_debug_banner("ENDING PREDICT PROCESS")

# Main function to start the prediction server
def live_predict(model_path, use_holistic):
    # Launch UDP server to receive landmark features
    common.start_server(
        LandmarkReceiver(),
        SERVER_ADDR
    )

if __name__ == "__main__":
    # Get model path from arguments or use default
    if len(argv) < 2:
        model_path = CURRENT_WORKING_DIRECTORY/'models'/DEFAULT_MODEL
        if not model_path.exists():
            raise MissingModelException("NO MODEL CAN BE USED!")
    else:
        model_path = argv[1]    
    
    if DEBUG:
        common.print_debug_banner(f"using model {model_path}")

    # Start the prediction server
    live_predict(model_path, False)

    