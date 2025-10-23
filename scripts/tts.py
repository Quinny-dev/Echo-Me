from gtts import gTTS
import pygame
import time
import textwrap
import os
import threading
import shutil
import uuid
import pyttsx3
try:
    from deep_translator import GoogleTranslator
    TRANSLATOR_AVAILABLE = True
except ImportError:
    TRANSLATOR_AVAILABLE = False
    print("Warning: deep-translator not installed. Translation features disabled.")
    print("Install with: pip install deep-translator")

MAX_CHARS = 4900
OUTPUT_DIR = "tts_chunks"
os.makedirs(OUTPUT_DIR, exist_ok=True)

audio_files = []

def clear_output_dir():
    """Safely clear old audio files"""
    if pygame.mixer.get_init():
        pygame.mixer.music.stop()
        pygame.mixer.quit()

    for f in os.listdir(OUTPUT_DIR):
        if f.endswith(".mp3") or f.endswith(".wav"):
            try:
                os.remove(os.path.join(OUTPUT_DIR, f))
            except PermissionError:
                pass

LANGUAGE_OPTIONS = {
    'No Translation': {'code': None, 'tts_lang': None, 'tts_tld': None},
    'English (US)': {'code': 'en', 'tts_lang': 'en', 'tts_tld': 'com'},
    'English (UK)': {'code': 'en', 'tts_lang': 'en', 'tts_tld': 'co.uk'},
    'English (Australia)': {'code': 'en', 'tts_lang': 'en', 'tts_tld': 'com.au'},
    'English (India)': {'code': 'en', 'tts_lang': 'en', 'tts_tld': 'co.in'},
    'Afrikaans': {'code': 'af', 'tts_lang': 'af', 'tts_tld': 'com'},
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
    'Zulu (Translation only)': {'code': 'zu', 'tts_lang': 'en', 'tts_tld': 'com'},
    'Xhosa (Translation only)': {'code': 'xh', 'tts_lang': 'en', 'tts_tld': 'com'},
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
    """Translate text to target language using deep-translator (Google Translate)"""
    try:
        if target_language == 'No Translation':
            return text
        
        lang_config = LANGUAGE_OPTIONS[target_language]
        if lang_config['code'] is None:
            return text
        
        # Create translator instance
        translator = GoogleTranslator(source='auto', target=lang_config['code'])
        
        # Split into chunks to avoid API limits (deep-translator limit is 5000 chars)
        chunks = textwrap.wrap(text, 4500, break_long_words=False, break_on_hyphens=False)
        translated_chunks = []
        
        for chunk in chunks:
            try:
                result = translator.translate(chunk)
                translated_chunks.append(result)
            except Exception as chunk_error:
                # If a chunk fails, try to continue with others
                print(f"Warning: Failed to translate chunk: {chunk_error}")
                translated_chunks.append(chunk)  # Use original text for failed chunk
        
        return ' '.join(translated_chunks)
    except Exception as e:
        raise Exception(f"Translation failed: {str(e)}")

def convert_with_gtts_translated(text, target_language, speed="Normal"):
    """Convert translated text using Google TTS with appropriate language settings"""
    lang_config = LANGUAGE_OPTIONS[target_language]
    chunks = textwrap.wrap(text, MAX_CHARS, break_long_words=False, break_on_hyphens=False)
    
    for i, chunk in enumerate(chunks):
        try:
            filename = os.path.join(OUTPUT_DIR, f"chunk_{uuid.uuid4()}.mp3")
            tts = gTTS(
                text=chunk, 
                lang=lang_config['tts_lang'], 
                tld=lang_config['tts_tld'],
                slow=speed == "Slow"
            )
            tts.save(filename)
            audio_files.append(filename)
        except Exception as e:
            if lang_config['tts_lang'] not in ['en']:
                try:
                    tts = gTTS(text=chunk, lang='en', tld='com', slow=speed == "Slow")
                    tts.save(filename)
                    audio_files.append(filename)
                    if i == 0:
                        print(f"Warning: TTS for {target_language} not available. Using English voice.")
                except Exception as e2:
                    raise Exception(f"TTS failed: {str(e2)}")
            else:
                raise e

def convert_with_gtts(text, voice="English (US)", speed="Normal"):
    """Convert text using Google TTS"""
    voice_config = GTTS_VOICES[voice]
    chunks = textwrap.wrap(text, MAX_CHARS, break_long_words=False, break_on_hyphens=False)
    
    for chunk in chunks:
        filename = os.path.join(OUTPUT_DIR, f"chunk_{uuid.uuid4()}.mp3")
        tts = gTTS(
            text=chunk, 
            lang=voice_config['lang'], 
            tld=voice_config['tld'],
            slow=speed == "Slow"
        )
        tts.save(filename)
        audio_files.append(filename)

def convert_with_pyttsx3(text, speed="Normal"):
    """Convert text using pyttsx3 (offline)"""
    try:
        engine = pyttsx3.init()
        
        rate = engine.getProperty('rate')
        if speed == "Slow":
            engine.setProperty('rate', rate - 50)
        elif speed == "Fast":
            engine.setProperty('rate', rate + 50)
        
        engine.setProperty('volume', 1.0)
        chunks = textwrap.wrap(text, MAX_CHARS, break_long_words=False, break_on_hyphens=False)
        
        for chunk in chunks:
            filename = os.path.join(OUTPUT_DIR, f"chunk_{uuid.uuid4()}.wav")
            engine.save_to_file(chunk, filename)
            audio_files.append(filename)
        
        engine.runAndWait()
        engine.stop()
    except Exception as e:
        raise Exception(f"pyttsx3 error: {str(e)}")

def play_audio_sync():
    """Play all generated audio files sequentially (blocking)"""
    if not audio_files:
        return
    
    if pygame.mixer.get_init():
        pygame.mixer.music.stop()
        pygame.mixer.quit()
    
    pygame.mixer.init()
    
    try:
        for audio_file in audio_files:
            pygame.mixer.music.load(audio_file)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                time.sleep(0.3)
    finally:
        pygame.mixer.quit()

def download_audio_files(destination_folder):
    """Copy all generated audio files to destination folder"""
    if not audio_files:
        raise Exception("No audio files to download")
    
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)
    
    for file_path in audio_files:
        shutil.copy(file_path, destination_folder)
    
    return len(audio_files)

def convert_and_play(text, user_prefs):
    """
    Main function to handle translation + TTS + playback.
    Called from gui.py.
    
    Args:
        text (str): Text to convert and play.
        user_prefs (dict): Dictionary containing:
            {
                "translate_to": "English (US)" or "No Translation",
                "voice": "English (US)",
                "speed": "Normal"
            }
    
    Returns:
        str: Translated text (or original if no translation)
    
    Raises:
        Exception: If conversion or playback fails
    """
    global audio_files
    audio_files.clear()
    clear_output_dir()

    if not text.strip():
        raise Exception("No text provided for TTS")

    target_language = user_prefs.get("translate_to", "No Translation")
    speed = user_prefs.get("speed", "Normal")
    voice = user_prefs.get("voice", "English (US)")

    # Translate if needed
    if target_language != 'No Translation':
        print(f"Translating text to {target_language}...")
        translated_text = translate_text(text, target_language)
    else:
        translated_text = text

    # Convert to speech
    print(f"Converting text using Google TTS...")
    if target_language != 'No Translation':
        convert_with_gtts_translated(translated_text, target_language, speed)
    else:
        convert_with_gtts(translated_text, voice, speed)

    # Play audio
    print(f"Playing {len(audio_files)} audio chunks...")
    play_audio_sync()
    
    return translated_text

def cleanup():
    """Clean up resources and temporary files"""
    try:
        if pygame.mixer.get_init():
            pygame.mixer.music.stop()
            pygame.mixer.quit()
    except:
        pass
    
    try:
        for f in os.listdir(OUTPUT_DIR):
            if f.endswith(".mp3") or f.endswith(".wav"):
                try:
                    os.remove(os.path.join(OUTPUT_DIR, f))
                except:
                    pass
    except:
        pass