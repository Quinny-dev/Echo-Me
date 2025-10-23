import tkinter as tk
from tkinter import scrolledtext, messagebox, filedialog, ttk
from gtts import gTTS
import pygame
import time
import textwrap
import os
import threading
import shutil
import uuid
import pyttsx3
from googletrans import Translator

MAX_CHARS = 4900
OUTPUT_DIR = "tts_chunks"
os.makedirs(OUTPUT_DIR, exist_ok=True)

audio_files = []
translator = Translator()

def clear_output_dir():
    """Safely clear old audio files"""
    if pygame.mixer.get_init():
        pygame.mixer.music.stop()
        pygame.mixer.quit()  # release lock on files

    for f in os.listdir(OUTPUT_DIR):
        if f.endswith(".mp3") or f.endswith(".wav"):
            try:
                os.remove(os.path.join(OUTPUT_DIR, f))
            except PermissionError:
                # Skip if file is still in use
                pass

LANGUAGE_OPTIONS = {
    'No Translation': {'code': None, 'tts_lang': None, 'tts_tld': None},
    'English (US)': {'code': 'en', 'tts_lang': 'en', 'tts_tld': 'com'},
    'English (UK)': {'code': 'en', 'tts_lang': 'en', 'tts_tld': 'co.uk'},
    'English (Australia)': {'code': 'en', 'tts_lang': 'en', 'tts_tld': 'com.au'},
    'English (India)': {'code': 'en', 'tts_lang': 'en', 'tts_tld': 'co.in'},
    'Afrikaans': {'code': 'af', 'tts_lang': 'af', 'tts_tld': 'com'},  # This is supported by gTTS
    'Spanish (Spain)': {'code': 'es', 'tts_lang': 'es', 'tts_tld': 'es'},
    'Spanish (Mexico)': {'code': 'es', 'tts_lang': 'es', 'tts_tld': 'com.mx'},
    'French (France)': {'code': 'fr', 'tts_lang': 'fr', 'tts_tld': 'fr'},
    'French (Canada)': {'code': 'fr', 'tts_lang': 'fr', 'tts_tld': 'ca'},
    'German': {'code': 'de', 'tts_lang': 'de', 'tts_tld': 'de'},
    'Italian': {'code': 'it', 'tts_lang': 'it', 'tts_tld': 'it'},
    'Portuguese (Brazil)': {'code': 'pt', 'tts_lang': 'pt', 'tts_tld': 'com.br'},
    'Portuguese (Portugal)': {'code': 'pt', 'tts_lang': 'pt', 'tts_tld': 'pt'},
    'Japanese': {'code': 'ja', 'tts_lang': 'ja', 'tts_tld': 'com'},
    'Korean': {'code': 'ko', 'tts_lang': 'ko', 'tts_tld': 'com'},
    'Chinese (Mandarin)': {'code': 'zh', 'tts_lang': 'zh', 'tts_tld': 'com'},
    'Russian': {'code': 'ru', 'tts_lang': 'ru', 'tts_tld': 'ru'},
    'Arabic': {'code': 'ar', 'tts_lang': 'ar', 'tts_tld': 'com'},
    'Hindi': {'code': 'hi', 'tts_lang': 'hi', 'tts_tld': 'com'},
    'Dutch': {'code': 'nl', 'tts_lang': 'nl', 'tts_tld': 'com'},
    'Swedish': {'code': 'sv', 'tts_lang': 'sv', 'tts_tld': 'com'},
    'Norwegian': {'code': 'no', 'tts_lang': 'no', 'tts_tld': 'com'},
    'Danish': {'code': 'da', 'tts_lang': 'da', 'tts_tld': 'com'},
    'Turkish': {'code': 'tr', 'tts_lang': 'tr', 'tts_tld': 'com'},
    'Polish': {'code': 'pl', 'tts_lang': 'pl', 'tts_tld': 'com'},
    'Czech': {'code': 'cs', 'tts_lang': 'cs', 'tts_tld': 'com'},
    'Hungarian': {'code': 'hu', 'tts_lang': 'hu', 'tts_tld': 'com'},
    'Finnish': {'code': 'fi', 'tts_lang': 'fi', 'tts_tld': 'com'},
    'Greek': {'code': 'el', 'tts_lang': 'el', 'tts_tld': 'com'},
    'Hebrew': {'code': 'he', 'tts_lang': 'he', 'tts_tld': 'com'},
    'Thai': {'code': 'th', 'tts_lang': 'th', 'tts_tld': 'com'},
    'Vietnamese': {'code': 'vi', 'tts_lang': 'vi', 'tts_tld': 'com'},
    # South African languages - Note: Limited TTS support
    'Zulu (Translation only)': {'code': 'zu', 'tts_lang': 'en', 'tts_tld': 'com'},  # Fallback to English TTS
    'Xhosa (Translation only)': {'code': 'xh', 'tts_lang': 'en', 'tts_tld': 'com'},  # Fallback to English TTS
}

GTTS_VOICES = {
    'English (US)': {'lang': 'en', 'tld': 'com'},
    'English (UK)': {'lang': 'en', 'tld': 'co.uk'},
    'English (Australia)': {'lang': 'en', 'tld': 'com.au'},
    'English (India)': {'lang': 'en', 'tld': 'co.in'},
    'Afrikaans': {'lang': 'af', 'tld': 'com'},  
    'Spanish (Spain)': {'lang': 'es', 'tld': 'es'},
    'Spanish (Mexico)': {'lang': 'es', 'tld': 'com.mx'},
    'French (France)': {'lang': 'fr', 'tld': 'fr'},
    'French (Canada)': {'lang': 'fr', 'tld': 'ca'},
    'German': {'lang': 'de', 'tld': 'de'},
    'Italian': {'lang': 'it', 'tld': 'it'},
    'Portuguese (Brazil)': {'lang': 'pt', 'tld': 'com.br'},
    'Portuguese (Portugal)': {'lang': 'pt', 'tld': 'pt'},
    'Japanese': {'lang': 'ja', 'tld': 'com'},
    'Korean': {'lang': 'ko', 'tld': 'com'},
    'Chinese (Mandarin)': {'lang': 'zh', 'tld': 'com'},
    'Russian': {'lang': 'ru', 'tld': 'ru'},
    'Arabic': {'lang': 'ar', 'tld': 'com'},
    'Hindi': {'lang': 'hi', 'tld': 'com'},
    'Dutch': {'lang': 'nl', 'tld': 'com'},
    'Swedish': {'lang': 'sv', 'tld': 'com'},
    'Norwegian': {'lang': 'no', 'tld': 'com'},
    'Danish': {'lang': 'da', 'tld': 'com'},
}

def get_pyttsx3_voices():
    """Get available system voices for pyttsx3"""
    try:
        engine = pyttsx3.init()
        voices = engine.getProperty('voices')
        voice_dict = {}
        for i, voice in enumerate(voices):
            name = voice.name if hasattr(voice, 'name') else f"Voice {i+1}"
            voice_dict[name] = voice.id
        engine.stop()
        return voice_dict
    except:
        return {}

def translate_text(text, target_language):
    """Translate text to target language using Google Translate"""
    try:
        if target_language == 'No Translation':
            return text
        
        lang_config = LANGUAGE_OPTIONS[target_language]
        if lang_config['code'] is None:
            return text
        
        # Split into chunks to avoid API limits
        chunks = textwrap.wrap(text, 4500, break_long_words=False, break_on_hyphens=False)
        translated_chunks = []
        
        for chunk in chunks:
            result = translator.translate(chunk, dest=lang_config['code'])
            translated_chunks.append(result.text)
        
        return ' '.join(translated_chunks)
    except Exception as e:
        raise Exception(f"Translation failed: {str(e)}")

def text_to_speech():
    user_text = text_area.get("1.0", tk.END).strip()
    if not user_text:
        messagebox.showwarning("Warning", "Please enter some text first.")
        return

    convert_btn.config(state=tk.DISABLED)
    play_btn.config(state=tk.DISABLED)
    download_btn.config(state=tk.DISABLED)
    status_label.config(text="Processing...")

    # Clear previous files safely
    clear_output_dir()

    def convert_thread():
        try:
            audio_files.clear()
            
            # Translate text if needed
            target_language = translation_var.get()
            if target_language != 'No Translation':
                status_label.config(text="Translating text...")
                translated_text = translate_text(user_text, target_language)
                status_label.config(text="Converting translated text to speech...")
            else:
                translated_text = user_text
                status_label.config(text="Converting text to speech...")
            
            # Show translated text in preview area
            if target_language != 'No Translation':
                root.after(0, lambda text=translated_text: update_translation_preview(text))
            else:
                root.after(0, lambda: update_translation_preview(""))
            
            if engine_var.get() == "Google TTS":
                if target_language != 'No Translation':
                    # Use the translation target language for TTS
                    convert_with_gtts_translated(translated_text, target_language)
                else:
                    convert_with_gtts(translated_text)
            else:
                convert_with_pyttsx3(translated_text)
                
            root.after(0, conversion_complete)
        except Exception as e:
            error_msg = str(e)  # Capture the error message properly
            root.after(0, lambda msg=error_msg: conversion_error(msg))

    threading.Thread(target=convert_thread, daemon=True).start()

def update_translation_preview(translated_text):
    """Update the translation preview area"""
    if translated_text:
        translation_preview.config(state=tk.NORMAL)
        translation_preview.delete("1.0", tk.END)
        translation_preview.insert("1.0", translated_text)
        translation_preview.config(state=tk.DISABLED)
        translation_frame.pack(pady=5, padx=20, fill=tk.X, after=text_area)
    else:
        translation_frame.pack_forget()

def convert_with_gtts_translated(text, target_language):
    """Convert translated text using Google TTS with appropriate language settings"""
    lang_config = LANGUAGE_OPTIONS[target_language]
    
    # Special handling for languages with limited TTS support
    if target_language in ['Zulu (Translation only)', 'Xhosa (Translation only)']:
        # Show warning to user about TTS limitation
        root.after(0, lambda: messagebox.showinfo("TTS Limitation", 
            f"{target_language.split(' ')[0]} translation is available, but TTS will use English voice due to limited language support."))
    
    chunks = textwrap.wrap(text, MAX_CHARS, break_long_words=False, break_on_hyphens=False)
    
    for i, chunk in enumerate(chunks):
        try:
            filename = os.path.join(OUTPUT_DIR, f"chunk_{uuid.uuid4()}.mp3")
            tts = gTTS(
                text=chunk, 
                lang=lang_config['tts_lang'], 
                tld=lang_config['tts_tld'],
                slow=speed_var.get() == "Slow"
            )
            tts.save(filename)
            audio_files.append(filename)
        except Exception as e:
            # If TTS fails for a specific language, try fallback to English
            if lang_config['tts_lang'] not in ['en']:
                try:
                    tts = gTTS(
                        text=chunk, 
                        lang='en', 
                        tld='com',
                        slow=speed_var.get() == "Slow"
                    )
                    tts.save(filename)
                    audio_files.append(filename)
                    # Notify user about fallback
                    if i == 0:  # Only show once
                        root.after(0, lambda: messagebox.showwarning("TTS Fallback", 
                            f"TTS for {target_language} not available. Using English voice instead."))
                except Exception as e2:
                    raise Exception(f"TTS failed for both target language and English fallback: {str(e2)}")
            else:
                raise e


def convert_with_gtts(text):
    """Convert text using Google TTS"""
    selected_voice = voice_var.get()
    voice_config = GTTS_VOICES[selected_voice]
    
    chunks = textwrap.wrap(text, MAX_CHARS, break_long_words=False, break_on_hyphens=False)
    
    for i, chunk in enumerate(chunks):
        filename = os.path.join(OUTPUT_DIR, f"chunk_{uuid.uuid4()}.mp3")
        tts = gTTS(
            text=chunk, 
            lang=voice_config['lang'], 
            tld=voice_config['tld'],
            slow=speed_var.get() == "Slow"
        )
        tts.save(filename)
        audio_files.append(filename)

def convert_with_pyttsx3(text):
    """Convert text using pyttsx3 (offline)"""
    try:
        engine = pyttsx3.init()
        
        # Set voice
        voices = engine.getProperty('voices')
        selected_voice_name = voice_var.get()
        pyttsx3_voices = get_pyttsx3_voices()
        
        if selected_voice_name in pyttsx3_voices:
            engine.setProperty('voice', pyttsx3_voices[selected_voice_name])
        
        # Set rate (speed)
        rate = engine.getProperty('rate')
        if speed_var.get() == "Slow":
            engine.setProperty('rate', rate - 50)
        elif speed_var.get() == "Fast":
            engine.setProperty('rate', rate + 50)
        
        # Set volume
        engine.setProperty('volume', 1.0)
        
        # Split text into chunks for large texts
        chunks = textwrap.wrap(text, MAX_CHARS, break_long_words=False, break_on_hyphens=False)
        
        for i, chunk in enumerate(chunks):
            filename = os.path.join(OUTPUT_DIR, f"chunk_{uuid.uuid4()}.wav")
            engine.save_to_file(chunk, filename)
            audio_files.append(filename)
        
        engine.runAndWait()
        engine.stop()
        
    except Exception as e:
        raise Exception(f"pyttsx3 error: {str(e)}")

def conversion_complete():
    """Called when conversion is complete"""
    target_lang = translation_var.get()
    if target_lang != 'No Translation':
        status_label.config(text=f"Translation and conversion complete! Generated {len(audio_files)} audio chunks.")
    else:
        status_label.config(text=f"Conversion complete! Generated {len(audio_files)} audio chunks.")
    
    convert_btn.config(state=tk.NORMAL)
    play_btn.config(state=tk.NORMAL)
    download_btn.config(state=tk.NORMAL)

def conversion_error(error_msg):
    """Called when conversion fails"""
    status_label.config(text="Conversion failed.")
    messagebox.showerror("Error", f"Conversion failed: {error_msg}")
    convert_btn.config(state=tk.NORMAL)

def play_audio():
    if not audio_files:
        messagebox.showwarning("Warning", "No audio files to play. Please convert text first.")
        return

    def play_sequence():
        if pygame.mixer.get_init():
            pygame.mixer.music.stop()
            pygame.mixer.quit()

        pygame.mixer.init()
        play_btn.config(state=tk.DISABLED)
        convert_btn.config(state=tk.DISABLED)
        download_btn.config(state=tk.DISABLED)

        try:
            for audio_file in audio_files:
                root.after(0, lambda f=audio_file: status_label.config(text=f"Playing: {os.path.basename(f)}"))
                pygame.mixer.music.load(audio_file)
                pygame.mixer.music.play()
                while pygame.mixer.music.get_busy():
                    time.sleep(0.5)

            root.after(0, lambda: status_label.config(text="Playback finished."))
        except Exception as e:
            root.after(0, lambda: messagebox.showerror("Playback Error", f"Error playing audio: {str(e)}"))
        finally:
            root.after(0, lambda: [
                play_btn.config(state=tk.NORMAL),
                convert_btn.config(state=tk.NORMAL),
                download_btn.config(state=tk.NORMAL)
            ])

    threading.Thread(target=play_sequence, daemon=True).start()

def download_audio():
    if not audio_files:
        messagebox.showwarning("Warning", "No audio files to download. Please convert text first.")
        return

    folder_selected = filedialog.askdirectory()
    if not folder_selected:
        return

    try:
        for file_path in audio_files:
            shutil.copy(file_path, folder_selected)
        messagebox.showinfo("Download Complete", f"Audio files saved to:\n{folder_selected}")
    except Exception as e:
        messagebox.showerror("Error", f"Failed to save files: {e}")

def on_engine_change(*args):
    """Update voice options when engine changes"""
    if engine_var.get() == "Google TTS":
        voice_combo['values'] = list(GTTS_VOICES.keys())
        voice_var.set('English (US)')
        speed_combo.config(state='readonly')
    else:
        pyttsx3_voices = get_pyttsx3_voices()
        if pyttsx3_voices:
            voice_combo['values'] = list(pyttsx3_voices.keys())
            voice_var.set(list(pyttsx3_voices.keys())[0])
        else:
            voice_combo['values'] = ['No voices available']
            voice_var.set('No voices available')
        speed_combo.config(state='readonly')

def on_translation_change(*args):
    """Handle translation language change"""
    if translation_var.get() != 'No Translation':
        # Show note about Google TTS being recommended for translations
        if engine_var.get() != "Google TTS":
            messagebox.showinfo("Recommendation", 
                              "For best results with translations, consider using Google TTS engine which supports more languages.")

# GUI Setup
root = tk.Tk()
root.title("Enhanced Text-to-Speech Converter with Translation")
root.geometry("700x750")

# Main title
title_label = tk.Label(root, text="Text-to-Speech with Translation", font=("Arial", 16, "bold"))
title_label.pack(pady=10)

# Voice options frame
options_frame = tk.Frame(root)
options_frame.pack(pady=10, padx=20, fill=tk.X)

# Engine selection
tk.Label(options_frame, text="TTS Engine:", font=("Arial", 10, "bold")).grid(row=0, column=0, sticky=tk.W, pady=2)
engine_var = tk.StringVar(value="Google TTS")
engine_combo = ttk.Combobox(options_frame, textvariable=engine_var, values=["Google TTS", "System TTS (pyttsx3)"], 
                           state="readonly", width=20)
engine_combo.grid(row=0, column=1, sticky=tk.W, padx=(10, 0), pady=2)
engine_var.trace('w', on_engine_change)

# Translation language selection
tk.Label(options_frame, text="Translate to:", font=("Arial", 10, "bold")).grid(row=1, column=0, sticky=tk.W, pady=2)
translation_var = tk.StringVar(value="No Translation")
translation_combo = ttk.Combobox(options_frame, textvariable=translation_var, 
                                values=list(LANGUAGE_OPTIONS.keys()), state="readonly", width=25)
translation_combo.grid(row=1, column=1, sticky=tk.W, padx=(10, 0), pady=2)
translation_var.trace('w', on_translation_change)

# Voice selection (for non-translation mode or system TTS)
tk.Label(options_frame, text="Voice:", font=("Arial", 10, "bold")).grid(row=2, column=0, sticky=tk.W, pady=2)
voice_var = tk.StringVar(value="English (US)")
voice_combo = ttk.Combobox(options_frame, textvariable=voice_var, state="readonly", width=25)
voice_combo.grid(row=2, column=1, sticky=tk.W, padx=(10, 0), pady=2)

# Speed selection
tk.Label(options_frame, text="Speed:", font=("Arial", 10, "bold")).grid(row=3, column=0, sticky=tk.W, pady=2)
speed_var = tk.StringVar(value="Normal")
speed_combo = ttk.Combobox(options_frame, textvariable=speed_var, values=["Slow", "Normal", "Fast"], 
                          state="readonly", width=15)
speed_combo.grid(row=3, column=1, sticky=tk.W, padx=(10, 0), pady=2)

# Initialize voice options
on_engine_change()

# Text input
tk.Label(root, text="Enter your text below:", font=("Arial", 12, "bold")).pack(pady=(20, 5))
text_area = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=70, height=12, font=("Arial", 11))
text_area.pack(padx=20, pady=5)

# Translation preview frame (initially hidden)
translation_frame = tk.Frame(root)
tk.Label(translation_frame, text="Translation Preview:", font=("Arial", 10, "bold")).pack(anchor=tk.W)
translation_preview = scrolledtext.ScrolledText(translation_frame, wrap=tk.WORD, width=70, height=4, 
                                               font=("Arial", 10), state=tk.DISABLED, bg="#f0f0f0")
translation_preview.pack(fill=tk.X)

# Buttons frame
buttons_frame = tk.Frame(root)
buttons_frame.pack(pady=15)

convert_btn = tk.Button(buttons_frame, text="Convert to Speech", command=text_to_speech, 
                       font=("Arial", 12), bg="#4CAF50", fg="white", width=15)
convert_btn.grid(row=0, column=0, padx=5)

play_btn = tk.Button(buttons_frame, text="Play Audio", command=play_audio, 
                    font=("Arial", 12), bg="#2196F3", fg="white", width=15, state=tk.DISABLED)
play_btn.grid(row=0, column=1, padx=5)

download_btn = tk.Button(buttons_frame, text="Download Audio", command=download_audio, 
                        font=("Arial", 12), bg="#FF9800", fg="white", width=15, state=tk.DISABLED)
download_btn.grid(row=0, column=2, padx=5)

# Status label
status_label = tk.Label(root, text="Ready to convert text to speech.", font=("Arial", 10), fg="green")
status_label.pack(pady=10)

# Instructions
instructions = tk.Label(root, 
    text="Instructions: Choose translation language (optional), TTS engine and voice, enter text above, then click 'Convert to Speech'.\n"
         "Translation uses Google Translate and requires internet. Google TTS is recommended for translated content.\n"
         "System TTS works offline but voice availability depends on your system.",
    font=("Arial", 9), fg="gray", justify=tk.CENTER)
instructions.pack(pady=5, padx=20)

# Cleanup on app exit
def on_closing():
    """Clean up resources when closing the application"""
    try:
        if pygame.mixer.get_init():
            pygame.mixer.music.stop()
            pygame.mixer.quit()
    except:
        pass
    
    # Clean up temporary files
    try:
        for f in os.listdir(OUTPUT_DIR):
            if f.endswith(".mp3") or f.endswith(".wav"):
                try:
                    os.remove(os.path.join(OUTPUT_DIR, f))
                except:
                    pass
    except:
        pass
    
    root.destroy()

root.protocol("WM_DELETE_WINDOW", on_closing)
root.mainloop()