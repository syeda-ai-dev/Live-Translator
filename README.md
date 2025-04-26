# Real-time Speech Translation Application

This application provides real-time speech transcription and translation using:
- Whisper model for speech recognition
- Neural machine translation for text translation
- WebSocket communication for real-time processing

## Setup

1. Make sure you have Python 3.10+ installed
2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Running the Application

### Starting the Backend Server
```
cd backend
uvicorn main:app --reload
```

### Starting the Frontend
Simply open the `frontend/index.html` file in your web browser.

## System Requirements

- Python 3.10+
- Modern web browser with WebSocket support
- Microphone for audio input
- Internet connection (if using fallback translation)

## Troubleshooting

- If you see connection errors in the frontend, make sure the backend server is running
- Check that port 8000 is not blocked by your firewall
- If audio transcription is not working, check that your browser has permission to access your microphone

## License

This project is licensed under the MIT License - see the LICENSE file for details.