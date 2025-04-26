from faster_whisper import WhisperModel
import numpy as np
from pathlib import Path
import logging
import time
import traceback
import os
import queue
import torch

logger = logging.getLogger(__name__)

class WhisperTranscriber:
    def __init__(self, model_size="base", device="auto", compute_type="float16", min_samples=4000):
        """
        Initialize the transcriber with a Whisper model.
        :param model_size: Size of the Whisper model
        :param device: Device to run the model on
        :param compute_type: Computation type for the model
        :param min_samples: Minimum number of samples in buffer to trigger transcription
        """
        logger.info(f"Initializing WhisperTranscriber with model: {model_size}")
        
        # Choose device based on availability
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Using device: {device}")
        
        # Adjust compute type based on device
        if device == "cpu" and compute_type == "float16":
            compute_type = "float32"  # CPUs work better with float32
            logger.info("Switched to float32 compute type for CPU")
        
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)
        logger.info("Whisper model loaded successfully")
        
        # Audio buffer to accumulate audio samples
        self.buffer = np.array([], dtype=np.float32)
        self.min_samples = min_samples  # Reduced from the default for faster processing
        self.sample_rate = 16000  # Whisper expects 16kHz audio
        
        # Lower energy threshold for more sensitive speech detection
        self.energy_threshold = 0.001  # Reduced from 0.005 for better sensitivity
        self.silence_threshold = 0.5  # Seconds of silence to consider end of speech
        self.max_buffer_seconds = 30  # Maximum buffer size in seconds
        self.transcription_queue = queue.Queue()
        
        # Tracking the last transcription time for continuous transcription
        self.last_transcription_time = time.time()
        self.forced_transcribe_interval = 2.0  # Seconds between forced transcriptions even if buffer not full
        
        # Default language for transcription (can be changed in router)
        self.language = "en"

    def add_audio_chunk(self, audio_chunk):
        """Add audio chunk to buffer and check if transcription should be triggered."""
        if len(audio_chunk) == 0:
            return
            
        # Add to buffer
        self.buffer = np.append(self.buffer, audio_chunk)
        
        # Limit buffer size to prevent memory issues
        max_buffer_size = self.sample_rate * self.max_buffer_seconds
        if len(self.buffer) > max_buffer_size:
            logger.warning(f"Buffer too large, truncating to {self.max_buffer_seconds} seconds")
            self.buffer = self.buffer[-max_buffer_size:]
        
        # Calculate audio energy for activity detection
        energy = np.sqrt(np.mean(np.square(audio_chunk)))
        has_speech = energy > self.energy_threshold
        
        # Trigger transcription if:
        # 1. We have enough audio data (min_samples)
        # 2. We detect speech activity
        # 3. It's been too long since last transcription
        current_time = time.time()
        time_since_last = current_time - self.last_transcription_time
        
        buffer_has_min_samples = len(self.buffer) >= self.min_samples
        should_force_transcribe = time_since_last > self.forced_transcribe_interval
        
        if (buffer_has_min_samples and has_speech) or (buffer_has_min_samples and should_force_transcribe):
            self._transcribe_buffer()
            self.last_transcription_time = current_time

    def _transcribe_buffer(self):
        """Transcribe the current audio buffer."""
        if len(self.buffer) < self.min_samples:
            logger.debug("Buffer too small for transcription")
            return
            
        logger.debug(f"Transcribing buffer with {len(self.buffer)} samples")
        
        try:
            # Make a copy to avoid modifying during transcription
            audio_data = self.buffer.copy()
            
            # Transcribe with optimized parameters for real-time
            segments, _ = self.model.transcribe(
                audio_data, 
                language=self.language,  # Use specific language instead of "auto"
                beam_size=3,  # Reduced from 5 for speed
                word_timestamps=False,  # Disable for faster processing
                vad_filter=True,  # Enable VAD filtering to ignore silence
                vad_parameters=dict(min_silence_duration_ms=500),
                condition_on_previous_text=True,  # Improve continuity with previous transcription
                initial_prompt=None,
                temperature=0.0,  # Use greedy decoding for speed
                compression_ratio_threshold=2.4,  # Default is fine for short segments
                log_prob_threshold=-1.0  # Default is fine for short segments
            )
            
            # Process the result
            text = " ".join([segment.text for segment in segments])
            text = text.strip()
            
            if text:  # Only queue if there's actual text
                logger.debug(f"Transcribed: {text}")
                self.transcription_queue.put(text)
                
                # Keep a small portion of the buffer to maintain context
                # This helps with sentence continuity
                if len(self.buffer) > 0:
                    # Keep last 0.5 seconds of audio for context
                    context_samples = int(0.5 * self.sample_rate)
                    if len(self.buffer) > context_samples:
                        self.buffer = self.buffer[-context_samples:]
                    else:
                        # Clear buffer only if we have enough for a good transcription
                        if len(self.buffer) > self.min_samples * 2:
                            self.buffer = np.array([], dtype=np.float32)
            else:
                logger.debug("No transcription result - silence or noise detected")
                
        except Exception as e:
            logger.error(f"Error during transcription: {e}")
            
    def get_transcription(self, timeout=0.2):
        """
        Get the next transcription from the queue.
        :param timeout: Timeout in seconds
        :return: Transcription text or None if no transcription available
        """
        try:
            return self.transcription_queue.get(block=True, timeout=timeout)
        except queue.Empty:
            return None
            
    def set_language(self, language_code):
        """
        Set the language for transcription.
        :param language_code: ISO language code
        """
        self.language = language_code
        logger.info(f"Transcription language set to: {language_code}")