import queue
import threading
import tkinter as tk
from tkinter import ttk, scrolledtext
import sounddevice as sd
import numpy as np
import os
from faster_whisper import WhisperModel
import ctranslate2
from transformers import MarianTokenizer, MarianMTModel
from huggingface_hub import snapshot_download
import torch

class TranscriptionApp:
    def init_whisper_model(self):
        """Initialize Whisper model with local storage"""
        # Create models directory if it doesn't exist
        models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
        whisper_dir = os.path.join(models_dir, "whisper_tiny")
        os.makedirs(whisper_dir, exist_ok=True)

        try:
            # If model already exists locally, load it
            if os.path.exists(os.path.join(whisper_dir, "model.bin")):
                print(f"Loading existing Whisper model from {whisper_dir}")
                return WhisperModel(whisper_dir, device="cpu", compute_type="int8")
            
            # If not, download and save it locally
            print("Downloading Whisper model to local directory...")
            from faster_whisper.utils import download_model
            model_path = download_model(
                "tiny",
                output_dir=whisper_dir,
                local_files_only=False
            )
            return WhisperModel(model_path, device="cpu", compute_type="int8")
            
        except Exception as e:
            print(f"Error initializing Whisper model: {e}")
            # Fallback to direct loading if local storage fails
            return WhisperModel("tiny", device="cpu", compute_type="int8")

    def __init__(self, root):
        self.root = root
        self.root.title("Real-time Multilingual Speech Translation")
        
        # Initialize audio parameters
        self.sample_rate = 16000
        self.audio_queue = queue.Queue()
        self.running = False
        
        # Initialize Whisper model (tiny for fast processing)
        self.model = self.init_whisper_model()
        
        # Supported languages and their corresponding codes
        self.languages = {
            "Auto": None,
            "Arabic": "ar",
            "English": "en",
            "German": "de"
        }
        
        # Initialize translation models
        self.translation_pairs = {}
        self.tokenizers = {}
        self.init_translation()
        
        self.setup_gui()
        
        # Bind window close event
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
    def ensure_ct2_conversion(self, lang_pair_dir):
        """Ensure model is converted to CT2 format"""
        ct2_dir = os.path.join(lang_pair_dir, "ct2")
        if not os.path.exists(ct2_dir) or not os.path.exists(os.path.join(ct2_dir, "model.bin")):
            print(f"Converting model in {lang_pair_dir} to CT2 format...")
            try:
                # Clean up any existing failed conversion
                if os.path.exists(ct2_dir):
                    import shutil
                    shutil.rmtree(ct2_dir)
                
                # Convert directly from the pytorch model to CT2 format
                ctranslate2.converters.TransformersConverter(lang_pair_dir).convert(
                    output_dir=ct2_dir,
                    quantization="int8",
                    force=True
                )
                return True
            except Exception as e:
                print(f"Error converting to CT2 format: {e}")
                return False
        return True

    def init_translation(self):
        """Initialize translation models using CTranslate2"""
        print("Initializing translation models...")
        
        # Define model pairs we want to support
        model_pairs = [
            ("en", "ar", "Helsinki-NLP/opus-mt-en-ar"),
            ("ar", "en", "Helsinki-NLP/opus-mt-ar-en"),
            ("en", "de", "Helsinki-NLP/opus-mt-en-de"),
            ("de", "en", "Helsinki-NLP/opus-mt-de-en"),
            ("ar", "de", "Helsinki-NLP/opus-mt-ar-de"),
            ("de", "ar", "Helsinki-NLP/opus-mt-de-ar")
        ]
        
        # Get the models directory path
        models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
        print(f"Looking for models in: {models_dir}")
        
        for src_lang, tgt_lang, model_name in model_pairs:
            try:
                # Construct paths
                lang_pair_dir = os.path.join(models_dir, f"{src_lang}_{tgt_lang}")
                
                if os.path.exists(lang_pair_dir):
                    # Ensure model is converted to CT2 format
                    if self.ensure_ct2_conversion(lang_pair_dir):
                        ct2_dir = os.path.join(lang_pair_dir, "ct2")
                        print(f"Loading model for {src_lang}->{tgt_lang} from {ct2_dir}")
                        
                        # Load the CTranslate2 model
                        translator = ctranslate2.Translator(
                            ct2_dir,
                            device="cpu",
                            compute_type="int8",
                            inter_threads=2,
                            intra_threads=2
                        )
                        
                        # Load the tokenizer
                        try:
                            # First try loading from the original model directory
                            tokenizer = MarianTokenizer.from_pretrained(lang_pair_dir)
                        except:
                            # Fallback to loading from Hugging Face
                            tokenizer = MarianTokenizer.from_pretrained(f"Helsinki-NLP/opus-mt-{src_lang}-{tgt_lang}")
                        
                        # Store both the translator and tokenizer
                        self.translation_pairs[(src_lang, tgt_lang)] = translator
                        self.tokenizers[(src_lang, tgt_lang)] = tokenizer
                        
                        print(f"Successfully loaded translation model for {src_lang}->{tgt_lang}")
                    else:
                        print(f"Failed to convert model for {src_lang}->{tgt_lang} to CT2 format")
                else:
                    print(f"Model directory not found: {lang_pair_dir}")
                
            except Exception as e:
                print(f"Error setting up translation for {src_lang}->{tgt_lang}: {e}")
                print(f"Model files available in {lang_pair_dir}:")
                if os.path.exists(lang_pair_dir):
                    print("\n".join(os.listdir(lang_pair_dir)))
    
    def update_gui_safely(self, widget, text):
        """Thread-safe method to update GUI widgets"""
        try:
            self.root.after(0, lambda: widget.insert(tk.END, text))
            self.root.after(0, lambda: widget.see(tk.END))
        except Exception as e:
            print(f"GUI update error: {e}")
    
    def clean_translated_text(self, text):
        """Clean up translated text by removing redundant spaces and repeated words/phrases"""
        # Remove multiple spaces
        text = ' '.join(text.split())
        
        # Handle specific common repeated tokens
        common_repeats = ['Hallo', 'Hello', 'Hi', 'Thank', 'Danke', 'Thanks']
        words = text.split()
        cleaned_words = []
        
        i = 0
        while i < len(words):
            word = words[i]
            # Skip consecutive repeats of common words
            if word in common_repeats:
                # Add only one instance of the repeated word
                cleaned_words.append(word)
                # Skip all consecutive occurrences
                while i + 1 < len(words) and words[i + 1] == word:
                    i += 1
            else:
                # For other words, check for immediate repetition
                if not cleaned_words or word != cleaned_words[-1]:
                    cleaned_words.append(word)
            i += 1
            
        # Clean up punctuation
        text = ' '.join(cleaned_words)
        text = text.replace(' ,', ',').replace(' .', '.').replace(' !', '!').replace(' ?', '?')
        return text

    def translate_text(self, text, src_lang, tgt_lang):
        """Translate text using CTranslate2"""
        if not text.strip():
            return ""
            
        if src_lang == tgt_lang:
            return text
            
        if src_lang == "Auto":
            segments, info = self.model.transcribe(text)
            src_lang = info.language
            if src_lang not in ["en", "ar", "de"]:
                src_lang = "en"
        
        try:
            translator = self.translation_pairs.get((src_lang, tgt_lang))
            tokenizer = self.tokenizers.get((src_lang, tgt_lang))
            
            if translator and tokenizer:
                # Pre-process input text to handle sentence boundaries
                sentences = [s.strip() for s in text.split('.') if s.strip()]
                if not sentences:
                    sentences = [text]
                
                translated_sentences = []
                for sentence in sentences:
                    # Use the newer tokenizer API
                    model_inputs = tokenizer(
                        sentence,
                        return_tensors=None,
                        padding=False,
                        truncation=True,
                        max_length=512
                    )
                    
                    # Convert input ids to tokens
                    tokens = tokenizer.convert_ids_to_tokens(model_inputs["input_ids"])
                    
                    # Translate using more constrained parameters
                    results = translator.translate_batch(
                        [tokens],
                        beam_size=5,
                        length_penalty=1.0,
                        max_decoding_length=128,
                        return_scores=True,
                        sampling_topk=1,
                        no_repeat_ngram_size=3
                    )
                    
                    # Get highest scoring translation
                    translated_tokens = results[0].hypotheses[0]
                    
                    # Convert tokens back to text using the tokenizer's decode method
                    translated = tokenizer.decode(
                        tokenizer.convert_tokens_to_ids(translated_tokens),
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=True
                    )
                    
                    if translated.strip():
                        translated_sentences.append(translated.strip())
                
                # Join sentences with proper punctuation
                final_translation = '. '.join(translated_sentences)
                if final_translation and not final_translation.endswith('.'):
                    final_translation += '.'
                    
                return final_translation.strip()
            else:
                # Try pivot translation through English if direct path not available
                if src_lang != "en" and tgt_lang != "en":
                    en_translation = self.translate_text(text, src_lang, "en")
                    return self.translate_text(en_translation, "en", tgt_lang)
                return f"[No translation model: {src_lang}->{tgt_lang}]"
                
        except Exception as e:
            print(f"[Translation Error] {src_lang}->{tgt_lang}: {str(e)}")
            print(f"Text being translated: {text}")
            print(f"Available translation pairs: {list(self.translation_pairs.keys())}")
            return f"[Error: {str(e)}]"

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
                        # Update transcription area safely
                        self.update_gui_safely(self.text_area, transcription + "\n")
                        
                        # Perform translation and update translation area safely
                        translation = self.translate_text(
                            transcription.strip(),
                            selected_src_lang or "en",
                            selected_tgt_lang
                        )
                        self.update_gui_safely(self.translation_area, translation + "\n")
                    
                    # Clear the audio buffer
                    audio_data = []
                    
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error during processing: {e}")
                continue
    
    def on_closing(self):
        """Handle window closing event"""
        self.stop_transcription()
        self.root.destroy()
    
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
        if self.running:
            self.running = False
            if hasattr(self, 'stream'):
                self.stream.stop()
                self.stream.close()
            if hasattr(self, 'process_thread'):
                self.process_thread.join(timeout=1.0)  # Wait up to 1 second for thread to finish
            
            # Update button states safely
            try:
                self.start_button.config(state=tk.NORMAL)
                self.stop_button.config(state=tk.DISABLED)
                
                # Add stopped message safely
                self.update_gui_safely(self.text_area, "\nTranscription stopped.\n")
                self.update_gui_safely(self.translation_area, "\nTranslation stopped.\n")
            except:
                pass  # Ignore errors during shutdown

def main():
    root = tk.Tk()
    app = TranscriptionApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()