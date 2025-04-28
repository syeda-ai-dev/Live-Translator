import queue
import threading
import tkinter as tk
from tkinter import ttk, scrolledtext
import sounddevice as sd
import numpy as np
import os
from faster_whisper import WhisperModel
from argostranslate import package, translate

class TranscriptionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Real-time Speech Transcription and Translation")
        
        # Initialize audio parameters
        self.sample_rate = 16000
        self.audio_queue = queue.Queue()
        self.running = False
        
        # Initialize Whisper model (tiny for fast processing)
        self.model = WhisperModel("tiny", device="cpu", compute_type="int8")
        
        # Supported languages
        self.languages = {
            "Auto": None,
            "Arabic": "ar",
            "English": "en"
        }
        
        # Initialize translation modules
        self.translation_pairs = {}
        self.init_translation()
        
        self.setup_gui()
        
    def init_translation(self):
        """Initialize translation pairs using Argos Translate"""
        print("Initializing translation pairs...")
        
        # First, check if packages are installed
        installed_languages = translate.get_installed_languages()
        print(f"Found {len(installed_languages)} installed languages:")
        for lang in installed_languages:
            print(f"  - {lang.code}: {lang.name}")
            
        # If no languages are installed, try manual installation
        if len(installed_languages) == 0:
            self.manual_package_install()
            # Refresh installed languages
            installed_languages = translate.get_installed_languages()
            print(f"After manual installation, found {len(installed_languages)} languages:")
            for lang in installed_languages:
                print(f"  - {lang.code}: {lang.name}")
        
        # Create translation pairs for our supported languages
        for from_lang in ["ar", "en"]:
            for to_lang in ["ar", "en"]:
                if from_lang != to_lang:
                    from_code = from_lang
                    to_code = to_lang
                    
                    print(f"Setting up translation: {from_code} -> {to_code}")
                    
                    # Find the language objects
                    from_lang_obj = next((lang for lang in installed_languages if lang.code.startswith(from_code)), None)
                    to_lang_obj = next((lang for lang in installed_languages if lang.code.startswith(to_code)), None)
                    
                    if from_lang_obj and to_lang_obj:
                        print(f"  Found models for {from_lang_obj.code} -> {to_lang_obj.code}")
                        try:
                            translation = from_lang_obj.get_translation(to_lang_obj)
                            self.translation_pairs[(from_code, to_code)] = translation
                            print(f"  Successfully created translator for {from_code} -> {to_code}")
                            
                            # Test the translation
                            test_result = translation.translate("Hello world")
                            print(f"  Test translation: 'Hello world' -> '{test_result}'")
                        except Exception as e:
                            print(f"  Error creating translator for {from_code} -> {to_code}: {e}")
                    else:
                        print(f"  Missing language model: from_lang={from_lang_obj}, to_lang={to_lang_obj}")
        
        print(f"Initialized {len(self.translation_pairs)} translation pairs: {list(self.translation_pairs.keys())}")
    
    def manual_package_install(self):
        """Manually install packages from files"""
        print("Attempting to manually install language packages...")
        
        # Check common locations for package files
        package_dirs = [
            os.path.join(os.path.expanduser("~"), ".local", "share", "argos-translate", "packages"),
            os.path.join(os.path.expanduser("~"), ".argos-translate", "packages")
        ]
        
        for package_dir in package_dirs:
            if os.path.exists(package_dir):
                print(f"Found package directory: {package_dir}")
                for filename in os.listdir(package_dir):
                    if filename.endswith('.argosmodel'):
                        package_path = os.path.join(package_dir, filename)
                        print(f"Installing package from: {package_path}")
                        try:
                            package.install_from_path(package_path)
                            print(f"Successfully installed {filename}")
                        except Exception as e:
                            print(f"Failed to install {filename}: {e}")
            else:
                print(f"Package directory not found: {package_dir}")
    
    def translate_text(self, text, src_lang, tgt_lang):
        """Translate text using Argos Translate"""
        if src_lang == tgt_lang:
            return text
            
        if src_lang == "Auto":
            # Use detected language from Whisper
            segments, info = self.model.transcribe(text)
            src_lang = info.language
            print(f"Detected language: {src_lang} with probability {info.language_probability}")
        
        print(f"Attempting to translate: {src_lang} -> {tgt_lang}")
        print(f"Available translation pairs: {list(self.translation_pairs.keys())}")
        
        # Ensure we have the correct language pair
        if (src_lang, tgt_lang) not in self.translation_pairs:
            print(f"Translation pair not found: {src_lang} -> {tgt_lang}")
            return f"[Missing translation model: {src_lang}->{tgt_lang}]"
        
        try:
            translator = self.translation_pairs.get((src_lang, tgt_lang))
            if translator:
                translated = translator.translate(text)
                if translated:
                    return translated
                else:
                    print("Translation returned empty result")
                    return f"[Empty translation: {src_lang}->{tgt_lang}]"
            else:
                print(f"Translator object not found for {src_lang}->{tgt_lang}")
                return f"[Translator not found: {src_lang}->{tgt_lang}]"
        except Exception as e:
            print(f"Translation error: {e}")
            return f"[Translation error: {src_lang}->{tgt_lang}] - {str(e)}"

    def setup_gui(self):
        # Create main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Create language selection frame
        lang_frame = ttk.Frame(main_frame)
        lang_frame.grid(row=0, column=0, columnspan=2, pady=5)
        
        # Source language selection
        src_frame = ttk.LabelFrame(lang_frame, text="Source Language")
        src_frame.pack(side=tk.LEFT, padx=5)
        self.src_lang_var = tk.StringVar(value="Auto")
        src_combo = ttk.Combobox(src_frame, textvariable=self.src_lang_var, 
                               values=list(self.languages.keys()), 
                               state="readonly", width=10)
        src_combo.pack(padx=5, pady=5)
        
        # Target language selection
        tgt_frame = ttk.LabelFrame(lang_frame, text="Target Language")
        tgt_frame.pack(side=tk.LEFT, padx=5)
        self.tgt_lang_var = tk.StringVar(value="English")
        tgt_combo = ttk.Combobox(tgt_frame, textvariable=self.tgt_lang_var,
                               values=[lang for lang in self.languages.keys() if lang != "Auto"],
                               state="readonly", width=10)
        tgt_combo.pack(padx=5, pady=5)
        
        # Create text areas frame
        text_frame = ttk.Frame(main_frame)
        text_frame.grid(row=1, column=0, columnspan=2, pady=5)
        
        # Transcription text area
        trans_frame = ttk.LabelFrame(text_frame, text="Transcription")
        trans_frame.pack(side=tk.LEFT, padx=5)
        self.text_area = scrolledtext.ScrolledText(trans_frame, width=40, height=20)
        self.text_area.pack(padx=5, pady=5)
        
        # Translation text area
        tran_frame = ttk.LabelFrame(text_frame, text="Translation")
        tran_frame.pack(side=tk.LEFT, padx=5)
        self.translation_area = scrolledtext.ScrolledText(tran_frame, width=40, height=20)
        self.translation_area.pack(padx=5, pady=5)
        
        # Create buttons
        self.start_button = ttk.Button(main_frame, text="Start", command=self.start_transcription)
        self.start_button.grid(row=2, column=0, pady=5)
        
        self.stop_button = ttk.Button(main_frame, text="Stop", command=self.stop_transcription, state=tk.DISABLED)
        self.stop_button.grid(row=2, column=1, pady=5)
        
    def audio_callback(self, indata, frames, time, status):
        """Callback for audio input"""
        if status:
            print(status)
        self.audio_queue.put(indata.copy())
        
    def process_audio(self):
        """Process audio chunks and transcribe"""
        audio_data = []
        
        while self.running:
            try:
                data = self.audio_queue.get(timeout=0.5)
                audio_data.append(data.flatten())
                
                # Process every 2 seconds of audio
                if len(audio_data) * (len(data) / self.sample_rate) >= 2.0:
                    audio_chunk = np.concatenate(audio_data)
                    
                    # Get selected languages
                    selected_src_lang = self.languages[self.src_lang_var.get()]
                    selected_tgt_lang = self.languages[self.tgt_lang_var.get()]
                    
                    # Transcribe the audio chunk with optimized parameters
                    segments, _ = self.model.transcribe(
                        audio_chunk,
                        language=selected_src_lang,
                        beam_size=3,
                        best_of=3,
                        temperature=[0.0, 0.2, 0.4],
                        compression_ratio_threshold=2.2,
                        vad_filter=True,
                        vad_parameters={"threshold": 0.45}
                    )
                    
                    # Process transcription and translation
                    transcription = ""
                    for segment in segments:
                        transcription += segment.text + " "
                    
                    if transcription.strip():
                        # Update transcription area
                        self.text_area.insert(tk.END, transcription + "\n")
                        self.text_area.see(tk.END)
                        
                        # Perform translation and update translation area
                        translation = self.translate_text(
                            transcription.strip(),
                            selected_src_lang or "en",  # fallback to English if Auto
                            selected_tgt_lang
                        )
                        self.translation_area.insert(tk.END, translation + "\n")
                        self.translation_area.see(tk.END)
                    
                    # Clear the audio buffer
                    audio_data = []
                    
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error during processing: {e}")
                continue
                
    def start_transcription(self):
        """Start the transcription process"""
        self.running = True
        self.stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            callback=self.audio_callback
        )
        self.stream.start()
        
        # Start processing thread
        self.process_thread = threading.Thread(target=self.process_audio)
        self.process_thread.start()
        
        # Update button states
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        
        # Add initial message
        src_lang = self.src_lang_var.get()
        tgt_lang = self.tgt_lang_var.get()
        self.text_area.insert(tk.END, f"Listening... Speak something! (Source: {src_lang})\n")
        self.translation_area.insert(tk.END, f"Translation will appear here (Target: {tgt_lang})\n")
        
    def stop_transcription(self):
        """Stop the transcription process"""
        self.running = False
        if hasattr(self, 'stream'):
            self.stream.stop()
            self.stream.close()
        if hasattr(self, 'process_thread'):
            self.process_thread.join()
            
        # Update button states
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        
        # Add stopped message
        self.text_area.insert(tk.END, "\nTranscription stopped.\n")
        self.translation_area.insert(tk.END, "\nTranslation stopped.\n")

def main():
    root = tk.Tk()
    app = TranscriptionApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()