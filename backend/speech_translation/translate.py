import logging
import time
import queue
import asyncio
from typing import Dict, Optional, List
from deep_translator import GoogleTranslator

logger = logging.getLogger(__name__)

class TranslationService:
    def __init__(self, model_name: str = None, batch_size: int = 8):
        """
        Initialize the translation service with Google Translator.
        
        Args:
            model_name: Ignored in this version
            batch_size: Batch size for translation queue
        """
        logger.info(f"Initializing TranslationService")
        
        # Define supported languages - using specific language codes that work with Google Translator
        self.supported_languages = {
            'en': 'English',
            'de': 'German',
            'ar': 'Arabic'
        }
        
        # Language pairs we need to support
        self.language_pairs = [
            ('en', 'de'), ('en', 'ar'),
            ('de', 'en'), ('de', 'ar'),
            ('ar', 'en'), ('ar', 'de')
        ]
        
        # Prepare translators cache
        self.translators = {}
        self.batch_size = batch_size
        self.input_queue = queue.Queue()
        self.output_queue = queue.Queue()
        self.translation_cache = {}  # Simple cache to avoid re-translating identical text
        self.cache_size_limit = 1000  # Limit cache size to prevent memory issues
        self.last_source_text = ""  # Track last text to avoid redundant translations
        
        # Pre-initialize translators for all language pairs
        for source, target in self.language_pairs:
            try:
                key = f"{source}-{target}"
                self.translators[key] = GoogleTranslator(source=source, target=target)
                logger.info(f"Created translator for {source} to {target}")
            except Exception as e:
                logger.error(f"Error creating translator for {source} to {target}: {e}")
        
        # Start translation worker
        self.running = True
        self.worker_thread = asyncio.create_task(self._translation_worker())
        
        logger.info("TranslationService initialized successfully")
    
    def _get_translator(self, source_lang: str, target_lang: str):
        """Get or create a translator for the language pair"""
        key = f"{source_lang}-{target_lang}"
        
        if key not in self.translators:
            try:
                self.translators[key] = GoogleTranslator(source=source_lang, target=target_lang)
                logger.info(f"Created translator for {source_lang} to {target_lang}")
            except Exception as e:
                logger.error(f"Error creating translator: {e}")
                return None
                
        return self.translators[key]
    
    async def _translation_worker(self):
        """Background worker to process translation requests from the queue."""
        while self.running:
            try:
                # Collect batch of texts to translate
                translation_tasks = []
                
                # Try to get at least one item, blocking with timeout
                try:
                    first_item = await asyncio.wait_for(
                        asyncio.get_event_loop().run_in_executor(
                            None, lambda: self.input_queue.get(block=True, timeout=0.1)
                        ),
                        timeout=0.2
                    )
                    if isinstance(first_item, dict):
                        translation_tasks.append(first_item)
                        self.input_queue.task_done()
                except (asyncio.TimeoutError, queue.Empty):
                    # No items in queue, sleep briefly and continue
                    await asyncio.sleep(0.05)
                    continue
                
                # Get any additional items in the queue (non-blocking)
                while len(translation_tasks) < self.batch_size:
                    try:
                        item = self.input_queue.get_nowait()
                        if isinstance(item, dict):
                            translation_tasks.append(item)
                            self.input_queue.task_done()
                    except queue.Empty:
                        break
                
                if not translation_tasks:
                    continue
                
                # Process batch
                start_time = time.time()
                
                # Process each translation task
                for task in translation_tasks:
                    text = task.get('text', '')
                    source_lang = task.get('source_lang', 'en')
                    target_lang = task.get('target_lang', 'en')
                    
                    # Skip if languages are the same
                    if source_lang == target_lang:
                        self.output_queue.put(text)
                        continue
                    
                    # Check cache
                    cache_key = f"{source_lang}-{target_lang}-{text}"
                    if cache_key in self.translation_cache:
                        self.output_queue.put(self.translation_cache[cache_key])
                        continue
                    
                    # Translate
                    translated = await self._translate_text(text, source_lang, target_lang)
                    
                    # Store in cache if valid
                    if translated and translated != text:
                        if len(self.translation_cache) < self.cache_size_limit:
                            self.translation_cache[cache_key] = translated
                    else:
                        # Use original text if translation failed
                        translated = text
                    
                    # Put in output queue
                    self.output_queue.put(translated)
                
                process_time = time.time() - start_time
                logger.debug(f"Translated batch of {len(translation_tasks)} texts in {process_time:.3f}s")
                
            except Exception as e:
                logger.error(f"Error in translation worker: {e}")
                await asyncio.sleep(0.1)  # Prevent tight loop on errors
    
    async def _translate_text(self, text: str, source_lang: str, target_lang: str, max_retries: int = 3) -> str:
        """
        Translate text with retries
        """
        if not text:
            return text
            
        # Create translator
        translator = self._get_translator(source_lang, target_lang)
        if not translator:
            return text
            
        # Try with retries
        for attempt in range(max_retries):
            try:
                # Run in thread pool to avoid blocking
                result = await asyncio.get_event_loop().run_in_executor(
                    None, 
                    lambda: translator.translate(text)
                )
                
                if result:
                    return result
                return text
            except Exception as e:
                logger.error(f"Translation error (attempt {attempt+1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(1)  # Wait before retry
        
        # Return original if all retries fail
        return text
    
    def add_text(self, text: str, source_lang: str = "en", target_lang: str = "en") -> None:
        """
        Add text to the translation queue.
        Args:
            text: Text to translate
            source_lang: Source language code
            target_lang: Target language code
        """
        # Skip if text is empty or identical to the last one
        if not text or text == self.last_source_text:
            return
            
        # Update last text
        self.last_source_text = text
        
        # Add to queue with language information
        self.input_queue.put({
            'text': text,
            'source_lang': source_lang,
            'target_lang': target_lang
        })
    
    async def translate_text(self, text: str, source_lang: str = "en", target_lang: str = "en") -> Optional[str]:
        """
        Translate text directly (used by the API endpoint).
        Args:
            text: Text to translate
            source_lang: Source language code
            target_lang: Target language code
        Returns:
            Translated text or None on error
        """
        if not text:
            return text
            
        if source_lang == target_lang:
            return text
            
        # Check cache first
        cache_key = f"{source_lang}-{target_lang}-{text}"
        if cache_key in self.translation_cache:
            return self.translation_cache[cache_key]
            
        # Translate
        translated = await self._translate_text(text, source_lang, target_lang)
        
        # Store in cache if valid
        if translated and translated != text:
            if len(self.translation_cache) < self.cache_size_limit:
                self.translation_cache[cache_key] = translated
                
        return translated
    
    def get_translation(self, timeout: float = 0.1) -> Optional[str]:
        """
        Get the next available translation.
        Args:
            timeout: Time to wait for translation
        Returns:
            Translated text or None if nothing available
        """
        try:
            return self.output_queue.get(block=True, timeout=timeout)
        except queue.Empty:
            return None
    
    def stop(self):
        """Stop the translation worker."""
        self.running = False
        if self.worker_thread:
            try:
                self.worker_thread.cancel()
            except:
                pass
        logger.info("TranslationService stopped") 