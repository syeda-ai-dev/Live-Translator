from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import logging
import time
import uvicorn
from router.speech_router import router as speech_router

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Real-time Speech Translation API",
    description="API for real-time speech transcription and translation"
)

# Configure CORS
# Note: In production, you should restrict origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(speech_router, prefix="/api")

# Translation service instance (will be set at startup)
translation_service = None

@app.get("/health")
async def health_check(request: Request):
    """Health check endpoint"""
    client_host = request.client.host if request.client else "unknown"
    logger.info(f"Health check from {client_host}")
    return {
        "status": "ok",
        "timestamp": time.time(),
        "service": "speech-translation-api"
    }

@app.on_event("startup")
async def startup_event():
    """Startup event handler"""
    logger.info("Starting up Speech Translation API")
    # Log application configuration
    logger.info("Application initialized and ready to serve requests")

@app.on_event("shutdown")
async def shutdown_event():
    """Shutdown event handler"""
    logger.info("Shutting down Speech Translation API")
    
    # Clean up resources
    from router.speech_router import translator
    if hasattr(translator, 'stop'):
        try:
            logger.info("Stopping translation service")
            translator.stop()
            logger.info("Translation service stopped")
        except Exception as e:
            logger.error(f"Error stopping translation service: {e}")

    logger.info("Application shutdown complete")

if __name__ == "__main__":
    # This block is executed when the script is run directly
    uvicorn.run("main:app", host="127.0.0.1", port=8888, reload=False)