from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException
from typing import Dict, List
import json
import logging
import asyncio
from audio_handle.audio import AudioProcessor
from speech_translation.transcribe import WhisperTranscriber
from speech_translation.translate import TranslationService
import time

router = APIRouter(tags=["speech"])
logger = logging.getLogger(__name__)

# Set up more detailed logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Initialize services
audio_processor = AudioProcessor()
transcriber = WhisperTranscriber(model_size="base", device="auto", compute_type="float16", min_samples=4000)
translator = TranslationService(batch_size=8)

# Define supported languages (will use translator.supported_languages if available)
supported_languages = {
    'en': 'English',
    'de': 'German',
    'ar': 'Arabic'
}

@router.websocket("/ws/speech")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time speech translation.
    Receives binary audio data and returns transcription/translation.
    """
    await websocket.accept()
    target_lang = "en"  # default
    source_lang = "en"  # default
    logger.info("WebSocket connection established")
    
    # Flag to indicate when processing should stop
    processing_active = True
    
    # Heartbeat to keep connection alive
    async def send_heartbeat():
        try:
            while processing_active:
                await asyncio.sleep(15)  # Send heartbeat every 15 seconds
                if websocket.client_state.value == 0:  # 0 = WebSocketState.CONNECTED
                    try:
                        await websocket.send_json({"type": "heartbeat"})
                        logger.debug("Sent heartbeat")
                    except Exception as e:
                        logger.error(f"Failed to send heartbeat: {e}")
                        break
        except Exception as e:
            logger.error(f"Heartbeat task error: {e}")
    
    # Background task for processing audio and sending responses
    async def process_audio():
        try:
            consecutive_empty_chunks = 0
            last_send_time = 0  # Track when we last sent results
            while processing_active:
                try:
                    # Get transcription with timeout (non-blocking)
                    transcription = transcriber.get_transcription(timeout=0.2)
                    
                    if transcription:
                        logger.info(f"Transcription: '{transcription}'")
                        
                        # Add to translation queue if needed
                        if target_lang != source_lang:
                            # Check if translator has add_text method with appropriate signature
                            if hasattr(translator, 'supported_languages') and hasattr(translator, 'translate_text'):
                                # Using fallback translator
                                translation = await translator.translate_text(
                                    transcription, 
                                    target_lang=target_lang,
                                    source_lang=source_lang
                                )
                                if translation:
                                    logger.info(f"Translation: '{translation}'")
                                else:
                                    translation = transcription
                                    logger.warning("Translation not available, using original text")
                            else:
                                # Using standard translator
                                translator.add_text(transcription)
                                
                                # Get translation with timeout
                                translation = translator.get_translation(timeout=0.3)
                                
                                if translation:
                                    logger.info(f"Translation: '{translation}'")
                                else:
                                    # Use the original text if translation not available
                                    translation = transcription
                                    logger.warning("Translation not available, using original text")
                        else:
                            # No translation needed
                            translation = transcription
                        
                        # Send results back immediately
                        current_time = time.time()
                        if websocket.client_state.value == 0:  # 0 = WebSocketState.CONNECTED
                            await websocket.send_json({
                                "type": "result",
                                "source_text": transcription,
                                "translated_text": translation
                            })
                            logger.info("Sent result to client")
                            last_send_time = current_time
                    
                    # Short sleep to prevent tight loop
                    await asyncio.sleep(0.05)
                        
                except Exception as e:
                    logger.error(f"Error in background processing: {str(e)}")
                    if websocket.client_state.value == 0:
                        await websocket.send_json({
                            "type": "error",
                            "message": "Error processing audio"
                        })
                    await asyncio.sleep(0.1)  # Wait a bit after error
        except Exception as e:
            logger.error(f"Background task error: {str(e)}")
    
    # Start the processing tasks
    processing_task = asyncio.create_task(process_audio())
    heartbeat_task = asyncio.create_task(send_heartbeat())

    try:
        # Notify client that we're ready
        await websocket.send_json({
            "type": "ready",
            "message": "Server ready to receive audio"
        })
        
        message_count = 0
        audio_chunks_received = 0
        
        while True:
            # Receive data from WebSocket
            message = await websocket.receive()
            message_count += 1
            
            if message_count % 100 == 0:
                logger.info(f"Processed {message_count} messages, {audio_chunks_received} audio chunks")
            
            # Check message type
            if "text" in message:
                # Handle configuration messages
                try:
                    msg = json.loads(message["text"])
                    logger.info(f"Received config: {msg}")
                    if msg.get("type") == "config":
                        source_lang = msg.get("sourceLang", "en")
                        target_lang = msg.get("targetLang", "en")
                        
                        # Use translator's supported languages if available, otherwise use local definition
                        language_list = getattr(translator, 'supported_languages', supported_languages)
                        
                        # Validate languages
                        if source_lang not in language_list:
                            source_lang = "en"
                            logger.warning(f"Unsupported source language: {source_lang}, defaulting to English")
                            
                        if target_lang not in language_list:
                            target_lang = "en"
                            logger.warning(f"Unsupported target language: {target_lang}, defaulting to English")
                        
                        # Set the language in the transcriber
                        transcriber.set_language(source_lang)
                        logger.info(f"Set transcriber language to: {source_lang}")
                        
                        await websocket.send_json({
                            "type": "config",
                            "status": "ok",
                            "source_language": source_lang,
                            "target_language": target_lang
                        })
                        logger.info(f"Set source language to: {source_lang}, target language to: {target_lang}")
                except json.JSONDecodeError:
                    logger.error("Invalid JSON configuration received")
                    continue
                    
            elif "bytes" in message:
                # Process audio chunk
                try:
                    binary_data = message["bytes"]
                    audio_chunks_received += 1
                    
                    if len(binary_data) > 0:
                        logger.debug(f"Received audio chunk #{audio_chunks_received}: {len(binary_data)} bytes")
                        
                        # Process audio data
                        audio_chunk = await audio_processor.process_chunk(binary_data)
                        
                        # Add directly to transcriber
                        if len(audio_chunk) > 0:
                            transcriber.add_audio_chunk(audio_chunk)
                except Exception as e:
                    logger.error(f"Error processing audio: {str(e)}")
                    await websocket.send_json({
                        "type": "error",
                        "message": "Error processing audio chunk"
                    })
                    
    except WebSocketDisconnect:
        logger.info("Client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
    finally:
        # Clean up
        processing_active = False
        logger.info("Closing WebSocket and cleaning up resources")
        
        try:
            # Wait for the tasks to complete
            tasks = [processing_task, heartbeat_task]
            for task in tasks:
                task.cancel()
            
            await asyncio.gather(*tasks, return_exceptions=True)
            
            # Stop the translation service (if supported)
            if hasattr(translator, 'stop'):
                translator.stop()
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
            
        try:
            await websocket.close()
        except:
            pass