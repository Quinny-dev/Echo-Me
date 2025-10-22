import tkinter as tk
from tkinter import ttk, messagebox
import speech_recognition as sr
import queue, os, datetime

recognizer = sr.Recognizer()
stop_listening = None
audio_queue = queue.Queue()
mic_device_index = None
transcript_buffer = []
recording_timer = None
recording_interval = 60000  # 1 minute in ms

# --- Callback for background listening ---
def callback(recognizer, audio):
    try:
        text = recognizer.recognize_google(audio)
        msg = f"{text}"
        audio_queue.put(msg)
        transcript_buffer.append(msg)
        update_word_count()
        if auto_save_var.get():
            save_transcript(auto=True)
    except sr.UnknownValueError:
        audio_queue.put("⚠️ Could not understand the audio.")
    except sr.RequestError as e:
        audio_queue.put(f"API error: {e}")

# --- Start Recording ---
def start_recording():
    global stop_listening, mic, mic_device_index, recording_timer
    try:
        clear_output()  # auto-clear when starting
        if mic_device_index is None:
            mic = sr.Microphone()
        else:
            mic = sr.Microphone(device_index=mic_device_index)

        with mic as source:
            recognizer.adjust_for_ambient_noise(source)
        stop_listening = recognizer.listen_in_background(mic, callback)
        log("🎤 Recording started... Speak now!")
        status_var.set("Recording...")
        btn_start.config(state="disabled")
        btn_stop.config(state="normal")

        # Start 1-minute timer
        recording_timer = root.after(recording_interval, check_continue)
    except Exception as e:
        messagebox.showerror("Microphone Error", f"Could not access microphone:\n{e}")

# --- Stop Recording ---
def stop_recording(save_prompt=True):
    global stop_listening, recording_timer
    if stop_listening is not None:
        stop_listening(wait_for_stop=False)
        stop_listening = None
        if recording_timer:
            root.after_cancel(recording_timer)
            recording_timer = None
        log("⏹️ Recording stopped.")
        status_var.set("Stopped")
        btn_start.config(state="normal")
        btn_stop.config(state="disabled")

        if save_prompt:
            ask_save()

# --- Ask to continue after 1 min ---
def check_continue():
    global recording_timer
    choice = messagebox.askyesno("Continue Recording?", "1 minute has passed.\nDo you want to continue transcription?")
    if choice:
        # restart timer for another minute
        recording_timer = root.after(recording_interval, check_continue)
    else:
        stop_recording(save_prompt=True)

# --- Save confirmation dialog ---
def ask_save():
    if not transcript_buffer:
        return
    choice = messagebox.askyesno("Save Transcript?", "Do you want to save this transcript?")
    if choice:
        save_transcript()
    else:
        clear_output()

# --- Update GUI output ---
def update_output():
    try:
        while True:
            msg = audio_queue.get_nowait()
            txt_output.insert(tk.END, msg + "\n")
            txt_output.see(tk.END)
    except queue.Empty:
        pass
    root.after(200, update_output)

# --- Log helper ---
def log(msg):
    txt_output.insert(tk.END, msg + "\n")
    txt_output.see(tk.END)

# --- Clear panel ---
def clear_output():
    txt_output.delete(1.0, tk.END)
    transcript_buffer.clear()
    status_var.set("Ready")
    update_word_count()

# --- Save transcript ---
def save_transcript(auto=False):
    if not transcript_buffer:
        if not auto:
            messagebox.showinfo("Save Transcript", "Nothing to save yet.")
        return
    
    os.makedirs("transcripts", exist_ok=True)
    filename = f"transcripts/transcript_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(filename, "w", encoding="utf-8") as f:
        f.write("\n".join(transcript_buffer))
    if not auto:
        log(f"💾 Transcript saved as {filename}")
        messagebox.showinfo("Save Successful", f"Transcript saved to:\n{filename}")
    refresh_saved_list()

# --- Refresh saved transcripts tab ---
def refresh_saved_list():
    lst_saved.delete(0, tk.END)
    if os.path.exists("transcripts"):
        for f in os.listdir("transcripts"):
            if f.endswith(".txt"):
                lst_saved.insert(tk.END, f)

def open_selected_transcript(event=None):
    sel = lst_saved.get(tk.ACTIVE)
    if sel:
        path = os.path.join("transcripts", sel)
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        messagebox.showinfo(f"Transcript: {sel}", content)

# --- Word counter ---
def update_word_count():
    words = " ".join(transcript_buffer).split()
    word_count_var.set(f"Words: {len(words)}")

# --- Mic selection ---
def set_device(event):
    global mic_device_index
    selection = mic_choice.get()
    for idx, name in enumerate(sr.Microphone.list_microphone_names()):
        if selection == name:
            mic_device_index = idx
            log(f"🎧 Using microphone: {name}")
            status_var.set(f"Mic selected: {name}")
            break

# --- Menu actions ---
def about():
    messagebox.showinfo("About", "Speech-to-Text Recorder\nVersion 3.0\nBuilt with Python & Tkinter")

def exit_app():
    root.destroy()

# --- GUI Setup ---
root = tk.Tk()
root.title("Speech-to-Text Recorder")
root.geometry("750x500")

style = ttk.Style(root)
style.theme_use("clam")
style.configure("TButton", padding=6, relief="flat", font=("Segoe UI", 10, "bold"))
style.configure("Start.TButton", background="#4CAF50", foreground="white")  # Green
style.configure("Stop.TButton", background="#F44336", foreground="white")   # Red
style.configure("Save.TButton", background="#2196F3", foreground="white")   # Blue
style.configure("Clear.TButton", background="#FF9800", foreground="white")  # Orange
style.configure("TLabel", background="#2b2b2b", foreground="white")
style.configure("TFrame", background="#2b2b2b")

# Menu bar
menubar = tk.Menu(root)
filemenu = tk.Menu(menubar, tearoff=0)
filemenu.add_command(label="Save Transcript", command=save_transcript)
filemenu.add_command(label="Clear Panel", command=clear_output)
filemenu.add_separator()
filemenu.add_command(label="Exit", command=exit_app)
menubar.add_cascade(label="File", menu=filemenu)

helpmenu = tk.Menu(menubar, tearoff=0)
helpmenu.add_command(label="About", command=about)
menubar.add_cascade(label="Help", menu=helpmenu)
root.config(menu=menubar)

# Tabs
notebook = ttk.Notebook(root)
tab_record = ttk.Frame(notebook)
tab_saved = ttk.Frame(notebook)
notebook.add(tab_record, text="🎤 Live Recording")
notebook.add(tab_saved, text="📂 Saved Transcripts")
notebook.pack(expand=True, fill="both")

# --- Recording Tab Layout ---
frame_controls = tk.Frame(tab_record, bg="#2b2b2b")
frame_controls.pack(padx=10, pady=10, fill="x")

btn_start = ttk.Button(frame_controls, text="▶ Start Recording", style="Start.TButton", command=start_recording)
btn_start.grid(row=0, column=0, padx=5, pady=5)

btn_stop = ttk.Button(frame_controls, text="⏹ Stop Recording", style="Stop.TButton", command=lambda: stop_recording(True), state="disabled")
btn_stop.grid(row=0, column=1, padx=5, pady=5)

btn_clear = ttk.Button(frame_controls, text="🧹 Clear Panel", style="Clear.TButton", command=clear_output)
btn_clear.grid(row=0, column=2, padx=5, pady=5)

btn_save = ttk.Button(frame_controls, text="💾 Save Transcript", style="Save.TButton", command=save_transcript)
btn_save.grid(row=0, column=3, padx=5, pady=5)

# Auto-save checkbox
auto_save_var = tk.BooleanVar()
chk_autosave = ttk.Checkbutton(frame_controls, text="Auto-save", variable=auto_save_var)
chk_autosave.grid(row=0, column=4, padx=10)

# Microphone dropdown
mic_names = sr.Microphone.list_microphone_names()
mic_choice = tk.StringVar()
dropdown = ttk.OptionMenu(frame_controls, mic_choice, "Select Microphone", *mic_names, command=set_device)
dropdown.grid(row=1, column=0, columnspan=3, pady=10, sticky="w")

# Output text box
txt_output = tk.Text(tab_record, wrap="word", height=20, width=90, bg="#1e1e1e", fg="white", insertbackground="white")
txt_output.pack(padx=10, pady=10, fill="both", expand=True)

# --- Saved Tab Layout ---
lst_saved = tk.Listbox(tab_saved, bg="#1e1e1e", fg="white", height=20)
lst_saved.pack(padx=10, pady=10, fill="both", expand=True)
lst_saved.bind("<Double-1>", open_selected_transcript)
refresh_saved_list()

# Status bar
status_var = tk.StringVar(value="Ready")
word_count_var = tk.StringVar(value="Words: 0")
status_bar = tk.Label(root, textvariable=status_var, relief="sunken", anchor="w", bg="#3c3f41", fg="white")
status_bar.pack(side="left", fill="x", expand=True)
word_bar = tk.Label(root, textvariable=word_count_var, relief="sunken", anchor="e", bg="#3c3f41", fg="white")
word_bar.pack(side="right")

# Start updating recognized text
update_output()

root.mainloop()
