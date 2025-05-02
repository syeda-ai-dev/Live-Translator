import asyncio
import json
import websockets
import numpy as np
import os
import queue
import threading
from faster_whisper import WhisperModel
import ctranslate2
from transformers import MarianTokenizer, MarianMTModel

class TranslationServer:
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

    def __init__(self):
        # Initialize audio parameters
        self.sample_rate = 16000
        self.audio_queue = queue.Queue()
        
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
        
        # Active clients and their language preferences
        self.clients = {}
        
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

    async def process_audio(self, websocket, audio_data, sample_rate):
        """Process audio data received from client"""
        try:
            # Convert audio data to numpy array
            audio_np = np.array(audio_data, dtype=np.float32)
            
            # Get client's language preferences
            client_id = id(websocket)
            client_info = self.clients.get(client_id, {})
            src_lang_name = client_info.get('source_language', 'Auto')
            tgt_lang_name = client_info.get('target_language', 'English')
            
            # Get language codes
            src_lang = self.languages.get(src_lang_name)
            tgt_lang = self.languages.get(tgt_lang_name)
            
            # Transcribe the audio
            segments, _ = self.model.transcribe(
                audio_np,
                language=src_lang,
                beam_size=3,
                best_of=3,
                temperature=[0.0, 0.2, 0.4],
                compression_ratio_threshold=2.2,
                vad_filter=True,
                vad_parameters={"threshold": 0.45}
            )
            
            # Process transcription
            transcription = ""
            for segment in segments:
                transcription += segment.text + " "
            
            if transcription.strip():
                # Send transcription to client
                await websocket.send(json.dumps({
                    'type': 'transcription',
                    'text': transcription.strip()
                }))
                
                # Translate and send translation
                translation = self.translate_text(
                    transcription.strip(),
                    src_lang or "en",
                    tgt_lang
                )
                
                await websocket.send(json.dumps({
                    'type': 'translation',
                    'text': translation
                }))
                
        except Exception as e:
            print(f"Error processing audio: {e}")
            await websocket.send(json.dumps({
                'type': 'error',
                'message': str(e)
            }))

    async def handle_client(self, websocket, path):
        """Handle WebSocket client connection"""
        client_id = id(websocket)
        self.clients[client_id] = {
            'source_language': 'Auto',
            'target_language': 'English'
        }
        
        print(f"Client connected: {client_id}")
        
        try:
            async for message in websocket:
                data = json.loads(message)
                
                if data['type'] == 'audio':
                    # Process audio data
                    await self.process_audio(
                        websocket,
                        data['data'],
                        data.get('sampleRate', self.sample_rate)
                    )
                    
                elif data['type'] == 'language':
                    # Update client language preferences
                    self.clients[client_id]['source_language'] = data.get('sourceLanguage', 'Auto')
                    self.clients[client_id]['target_language'] = data.get('targetLanguage', 'English')
                    print(f"Client {client_id} language settings updated: {self.clients[client_id]}")
                    
        except websockets.exceptions.ConnectionClosed:
            print(f"Client disconnected: {client_id}")
        except Exception as e:
            print(f"Error handling client {client_id}: {e}")
        finally:
            # Clean up client data when they disconnect
            if client_id in self.clients:
                del self.clients[client_id]

    async def start_server(self, host='0.0.0.0', port=8765):
        """Start the WebSocket server"""
        # Create a wrapper function that ignores the path parameter
        async def handler_wrapper(websocket):
            await self.handle_client(websocket, None)
            
        server = await websockets.serve(handler_wrapper, host, port)
        print(f"WebSocket server started on ws://{host}:{port}")
        return server

async def main():
    # Initialize the translation server
    translation_server = TranslationServer()
    
    # Start the WebSocket server
    server = await translation_server.start_server()
    
    # Keep the server running
    await asyncio.Future()  # Run forever

if __name__ == "__main__":
    print("Starting Live Translator WebSocket Server...")
    asyncio.run(main())