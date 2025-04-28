import queue
import threading
import tkinter as tk
from tkinter import ttk, scrolledtext
import sounddevice as sd
import numpy as np
from faster_whisper import WhisperModel

class TranscriptionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Real-time Speech Transcription")
        
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
            "English": "en",
            "German": "de"
        }
        self.current_language = None
        
        self.setup_gui()
        
    def setup_gui(self):
        # Create main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Create language selection
        lang_frame = ttk.Frame(main_frame)
        lang_frame.grid(row=0, column=0, columnspan=2, pady=5)
        
        ttk.Label(lang_frame, text="Language:").pack(side=tk.LEFT, padx=5)
        self.lang_var = tk.StringVar(value="Auto")
        lang_combo = ttk.Combobox(lang_frame, textvariable=self.lang_var, 
                                values=list(self.languages.keys()), 
                                state="readonly", width=10)
        lang_combo.pack(side=tk.LEFT, padx=5)
        
        # Create text area for transcription
        self.text_area = scrolledtext.ScrolledText(main_frame, width=60, height=20)
        self.text_area.grid(row=1, column=0, columnspan=2, pady=5)
        
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
                    
                    # Get selected language
                    selected_lang = self.languages[self.lang_var.get()]
                    
                    # Transcribe the audio chunk with optimized parameters
                    segments, _ = self.model.transcribe(
                        audio_chunk,
                        language=selected_lang,
                        beam_size=3,  # Slightly increased beam size for better accuracy
                        best_of=3,    # Consider more candidates
                        temperature=[0.0, 0.2, 0.4],  # Temperature fallback for better results
                        compression_ratio_threshold=2.2,  # Slightly lower for better quality
                        vad_filter=True,
                        vad_parameters={"threshold": 0.45}  # More sensitive VAD
                    )
                    
                    # Update the text area with transcription
                    transcription = ""
                    for segment in segments:
                        transcription += segment.text + " "
                    
                    if transcription.strip():
                        self.text_area.insert(tk.END, transcription + "\n")
                        self.text_area.see(tk.END)
                    
                    # Clear the audio buffer
                    audio_data = []
                    
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error during transcription: {e}")
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
        lang = self.lang_var.get()
        self.text_area.insert(tk.END, f"Listening... Speak something! (Language: {lang})\n")
        
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

def main():
    root = tk.Tk()
    app = TranscriptionApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()