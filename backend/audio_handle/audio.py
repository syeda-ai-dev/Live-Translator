import numpy as np
import logging
from scipy import signal

logger = logging.getLogger(__name__)

class AudioProcessor:
    def __init__(self):
        self.sample_rate = 16000  # Whisper expects 16kHz
        self.max_audio_length = 15  # Reduced from 30 to 15 seconds for faster processing
        
    async def process_chunk(self, audio_data: bytes) -> np.ndarray:
        """
        Process incoming audio chunk from WebSocket
        Returns: numpy array of audio samples at 16kHz
        """
        try:
            # Use debug instead of info for less logging in production
            logger.debug(f"Processing audio chunk of size: {len(audio_data)} bytes")
            
            # Convert the raw bytes to a float32 numpy array
            # The data is already Float32Array from JavaScript
            try:
                audio_chunk = np.frombuffer(audio_data, dtype=np.float32)
                logger.debug(f"Converted to numpy array with {len(audio_chunk)} samples")
                
                # Check for valid data
                if len(audio_chunk) == 0:
                    logger.warning("Empty audio chunk received")
                    return np.array([], dtype=np.float32)
                    
                # Check for NaN values that can cause issues
                if np.isnan(audio_chunk).any():
                    logger.warning("Audio chunk contains NaN values, fixing...")
                    audio_chunk = np.nan_to_num(audio_chunk)
            except Exception as e:
                logger.error(f"Error converting audio data to numpy array: {e}")
                return np.array([], dtype=np.float32)
            
            # Set source rate based on browser audio context
            src_rate = 48000  # Higher rate to ensure quality (most browsers use 44.1kHz or 48kHz)
            
            # Lower minimum sample size for faster processing
            if len(audio_chunk) < 256:  # Reduced from 512 for lower latency
                logger.debug(f"Audio chunk too small: {len(audio_chunk)} samples")
                return np.array([], dtype=np.float32)
            
            # Convert to mono if stereo
            if len(audio_chunk.shape) > 1:
                logger.debug("Converting stereo to mono")
                audio_chunk = np.mean(audio_chunk, axis=1)
            
            # Ensure we're not processing too much audio at once
            max_samples = self.max_audio_length * self.sample_rate
            if len(audio_chunk) > max_samples:
                logger.warning(f"Audio chunk too long, truncating to {self.max_audio_length}s")
                audio_chunk = audio_chunk[:max_samples]
            
            # Resample to 16kHz for Whisper using optimized method for small chunks
            try:
                num_samples = int(len(audio_chunk) * self.sample_rate / src_rate)
                # Use a more efficient resampling method for small chunks
                if len(audio_chunk) < 4096:
                    # Simple linear interpolation for small chunks (faster)
                    indices = np.linspace(0, len(audio_chunk) - 1, num_samples)
                    audio_chunk = np.interp(indices, np.arange(len(audio_chunk)), audio_chunk)
                else:
                    # Use signal.resample for larger chunks (better quality)
                    audio_chunk = signal.resample(audio_chunk, num_samples)
                logger.debug(f"Resampled to {len(audio_chunk)} samples at {self.sample_rate}Hz")
            except Exception as e:
                logger.error(f"Error resampling audio: {e}")
                return np.array([], dtype=np.float32)
            
            # Calculate audio level for debugging
            rms = np.sqrt(np.mean(np.square(audio_chunk)))
            logger.debug(f"Audio level (RMS): {rms:.6f}")
            
            # Normalize audio to [-1, 1] range if needed
            max_val = np.max(np.abs(audio_chunk))
            if max_val > 0.001:  # Only normalize if there's actual sound
                if max_val > 1.0:
                    audio_chunk = audio_chunk / max_val
                    logger.debug(f"Normalized audio, max value was: {max_val:.6f}")
            else:
                logger.debug("Audio chunk very quiet, possibly silence")
            
            return audio_chunk.astype(np.float32)
            
        except Exception as e:
            logger.error(f"Error processing audio chunk: {e}")
            return np.array([], dtype=np.float32)  # Return empty array on error