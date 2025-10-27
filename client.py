# Client application for sign language recognition
import socket
from sys import argv
import cv2
import mediapipe as mp
import itertools
import numpy as np
import time
import sys
from multiprocessing import Queue, Process
from queue import Empty
import atexit
from math import ceil
from collections import deque

sys.path.insert(1, './tools')
import holistic, common, encrypt

# Configuration constants
PRINT_FREQ = 30  # Frequency for printing predictions
SERVER_ADDR = "127.0.0.1"  # Server IP address
# SERVER_ADDR = "127.0.0.1"

# Server connection details
serverAddressPort = (SERVER_ADDR, 9999)

APP_NAME = "SignSense"

# Process to handle server communication
def server(landmark_queue, prediction_queue):
    common.print_debug_banner("STARTED SERVER")
    # Create non-blocking UDP socket
    UDPClientSocket = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)
    UDPClientSocket.setblocking(0)
    # Main server communication loop
    while True:
        try:
            # Get landmark data from queue
            landmark = landmark_queue.get()
            # Encrypt landmark data for security
            encrypted_landmark = encrypt.encrypt_chacha(landmark)
            # Send encrypted landmark data to server
            UDPClientSocket.sendto(encrypted_landmark, serverAddressPort)
            # Receive prediction response from server
            msgFromServer = UDPClientSocket.recvfrom(2048)[0]
            # Decrypt server response
            raw_data = encrypt.decrypt_chacha(msgFromServer)
            # Add prediction to queue
            prediction_queue.put(raw_data)
        except encrypt.DecryptionError:
            print(f"tried to decrypt {msgFromServer}")
        except socket.error as e:
            # print(f"SOCKET EXCEPTION: {e}")
            pass
        except Exception as e:
            print(f"SERVER EXCEPTION: {e}")
            pass


# Process to handle video capture and display
def video_loop(landmark_queue, prediction_queue, use_holistic=False):
    # Initialize camera capture
    cap = cv2.VideoCapture(0)
    cv2.namedWindow(APP_NAME, cv2.WINDOW_NORMAL) 
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    cap.set(cv2.CAP_PROP_FOURCC, fourcc)
    if not cap.isOpened():
        print("Error opening Camera")
    fps = cap.get(cv2.CAP_PROP_FPS)
    print("Webcam FPS = {}".format(fps))

    # Get camera properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    mp_drawing = mp.solutions.drawing_utils

    # Initialize state variables
    timestamp = None
    started = False
    predicted = None
    initialized = False  # Track server connection status
    delay = 0
    pred_history = deque([" "]*5, 5)  # Store last 5 predictions
    pdecay = time.time()  # Time since last prediction update

    print("starting image cap")

    # Main video processing loop
    for image, results in holistic.process_capture(cap, use_holistic):
        # Check if window was closed
        window_state = cv2.getWindowProperty(APP_NAME, 0)
        if started and window_state == -1:
            print("QUITTING")
            break

        started = True

        # Track frame timing
        newtime = time.time()
        if timestamp is not None:
            diff = newtime - timestamp
            # Uncomment to print time between each frame
            # print(diff)
        timestamp = newtime

        # Convert landmarks to row format
        row = holistic.to_landmark_row(results, use_holistic)

        # Convert to comma-separated string
        landmark_str = ','.join(map(str, row))

        # Send landmark data to server process
        try:
            landmark_queue.put_nowait(landmark_str)
        except Exception as e:
            print(e)

        # Check for predictions from server
        try:
            out = prediction_queue.get_nowait()

            # Handle first server connection
            if out and not initialized:
                initialized = True

                common.print_debug_banner("SENDING ACK TO SERVER FOR CONNECTION")
                # Send acknowledgment to server
                landmark_queue.put_nowait("ACK")

            # Update prediction history at specified frequency
            if delay >= PRINT_FREQ:
                if out and out != pred_history[-1] and out != "None":
                    pred_history.append(out)
                    pdecay = time.time()
                delay = 0
        except:
            pass

        # Increment frame counter
        delay += 1
        # Clear predictions after 7 seconds of inactivity
        if time.time() - pdecay > 7:
            pred_history = deque([" "]*5, 5)
        # Draw landmarks and predictions on frame
        holistic.draw_landmarks(image, results, use_holistic, ' '.join(pred_history))

        # Display connection status on video feed
        if initialized:
            # Show green circle for connected
            cv2.circle(image,(20,450), 10, (0,255,0), -1)
            cv2.putText(image,'online',(40,458), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),2,cv2.LINE_AA)
            cv2.imshow(APP_NAME, image)
        else:
            # Show red circle for connecting
            cv2.circle(image,(20,450), 10, (0,0,255), -1)
            cv2.putText(image,'connecting',(40,458), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),2,cv2.LINE_AA)
            cv2.imshow(APP_NAME, image)
    # Clean up video capture
    cap.release()
    cv2.destroyAllWindows()

    # Notify server of client disconnection
    landmark_queue.put("END")

if __name__ == "__main__":
    # Create queues for inter-process communication
    landmark_queue, prediction_queue = Queue(), Queue()

    # Start server communication process
    server_p = Process(target=server, args=(landmark_queue, prediction_queue, ))
    server_p.daemon = True
    atexit.register(common.exit_handler, server_p)
    server_p.start()

    # Start video capture and display process
    video_p = Process(target=video_loop, args=(landmark_queue, prediction_queue, ))
    video_p.daemon = True
    atexit.register(common.exit_handler, video_p)
    video_p.start()

    # Wait for video process to complete
    video_p.join()
