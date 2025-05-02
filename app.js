// Web Audio API and WebSocket handling for Live Translator

class LiveTranslator {
    constructor() {
        // DOM elements
        this.startButton = document.getElementById('start-button');
        this.stopButton = document.getElementById('stop-button');
        this.transcriptionDisplay = document.getElementById('transcription');
        this.translationDisplay = document.getElementById('translation');
        this.sourceLanguageSelect = document.getElementById('source-language');
        this.targetLanguageSelect = document.getElementById('target-language');
        this.statusIndicator = document.getElementById('status-indicator');
        this.statusText = document.getElementById('status-text');
        
        // Audio processing parameters
        this.audioContext = null;
        this.microphone = null;
        this.processor = null;
        this.stream = null;
        this.isRecording = false;
        this.sampleRate = 16000; // Match the backend sample rate
        this.audioChunks = [];
        this.audioSendInterval = 2000; // Send audio every 2 seconds
        this.lastSendTime = 0;
        
        // WebSocket connection
        this.socket = null;
        this.socketReady = false;
        
        // Initialize the application
        this.init();
    }
    
    init() {
        // Set up event listeners
        this.startButton.addEventListener('click', () => this.startRecording());
        this.stopButton.addEventListener('click', () => this.stopRecording());
        
        // Initialize WebSocket connection
        this.initWebSocket();
    }
    
    initWebSocket() {
        // Create WebSocket connection to the backend server
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const host = window.location.hostname || 'localhost';
        const port = 8765; // Match the port in the server file
        
        this.socket = new WebSocket(`${protocol}//${host}:${port}`);
        
        this.socket.onopen = () => {
            console.log('WebSocket connection established');
            this.socketReady = true;
            this.updateStatus('Connected to server', false);
        };
        
        this.socket.onmessage = (event) => {
            const data = JSON.parse(event.data);
            
            if (data.type === 'transcription') {
                this.addMessage(this.transcriptionDisplay, data.text);
            } else if (data.type === 'translation') {
                this.addMessage(this.translationDisplay, data.text);
            } else if (data.type === 'error') {
                console.error('Server error:', data.message);
                this.addMessage(this.transcriptionDisplay, `Error: ${data.message}`, 'error-message');
            }
        };
        
        this.socket.onclose = () => {
            console.log('WebSocket connection closed');
            this.socketReady = false;
            this.updateStatus('Disconnected from server', false);
            
            // Attempt to reconnect after a delay
            setTimeout(() => this.initWebSocket(), 3000);
        };
        
        this.socket.onerror = (error) => {
            console.error('WebSocket error:', error);
            this.updateStatus('Connection error', false);
        };
    }
    
    async startRecording() {
        if (this.isRecording) return;
        
        try {
            // Request microphone access
            this.stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            
            // Initialize audio context
            this.audioContext = new (window.AudioContext || window.webkitAudioContext)();
            
            // Create microphone source
            this.microphone = this.audioContext.createMediaStreamSource(this.stream);
            
            // Create script processor for audio processing
            this.processor = this.audioContext.createScriptProcessor(4096, 1, 1);
            
            // Connect the audio processing pipeline
            this.microphone.connect(this.processor);
            this.processor.connect(this.audioContext.destination);
            
            // Set up audio processing callback
            this.processor.onaudioprocess = (e) => this.processAudio(e);
            
            // Update UI state
            this.isRecording = true;
            this.startButton.disabled = true;
            this.stopButton.disabled = false;
            this.updateStatus('Listening...', true);
            
            // Clear previous messages
            this.clearDisplays();
            
            // Add initial messages
            const srcLang = this.sourceLanguageSelect.value;
            const tgtLang = this.targetLanguageSelect.value;
            this.addMessage(this.transcriptionDisplay, `Listening... Speak something! (Source: ${srcLang})`);
            this.addMessage(this.translationDisplay, `Translation will appear here (Target: ${tgtLang})`);
            
            // Send language selection to the server
            this.sendLanguageSelection();
            
            // Reset audio chunks and timing
            this.audioChunks = [];
            this.lastSendTime = Date.now();
            
        } catch (error) {
            console.error('Error accessing microphone:', error);
            this.addMessage(this.transcriptionDisplay, `Error: ${error.message}`, 'error-message');
        }
    }
    
    stopRecording() {
        if (!this.isRecording) return;
        
        // Stop the audio processing
        if (this.processor) {
            this.processor.disconnect();
            this.processor = null;
        }
        
        if (this.microphone) {
            this.microphone.disconnect();
            this.microphone = null;
        }
        
        // Stop all audio tracks
        if (this.stream) {
            this.stream.getTracks().forEach(track => track.stop());
            this.stream = null;
        }
        
        // Close audio context
        if (this.audioContext) {
            if (this.audioContext.state !== 'closed') {
                this.audioContext.close();
            }
            this.audioContext = null;
        }
        
        // Update UI state
        this.isRecording = false;
        this.startButton.disabled = false;
        this.stopButton.disabled = true;
        this.updateStatus('Ready', false);
        
        // Add stopped messages
        this.addMessage(this.transcriptionDisplay, 'Transcription stopped.');
        this.addMessage(this.translationDisplay, 'Translation stopped.');
        
        // Send any remaining audio data
        if (this.audioChunks.length > 0) {
            this.sendAudioData();
        }
    }
    
    processAudio(e) {
        // Get audio data from the input channel
        const inputData = e.inputBuffer.getChannelData(0);
        
        // Store the audio data
        this.audioChunks.push(new Float32Array(inputData));
        
        // Send audio data every 2 seconds
        const currentTime = Date.now();
        if (currentTime - this.lastSendTime >= this.audioSendInterval) {
            this.sendAudioData();
            this.lastSendTime = currentTime;
        }
    }
    
    sendAudioData() {
        if (!this.socketReady || this.audioChunks.length === 0) return;
        
        // Concatenate all audio chunks
        let concatenated = new Float32Array(this.audioChunks.reduce((acc, chunk) => acc + chunk.length, 0));
        let offset = 0;
        
        for (const chunk of this.audioChunks) {
            concatenated.set(chunk, offset);
            offset += chunk.length;
        }
        
        // Convert to 16-bit PCM (same format expected by the backend)
        const pcmData = new Int16Array(concatenated.length);
        for (let i = 0; i < concatenated.length; i++) {
            // Convert float audio data to 16-bit signed integers
            const s = Math.max(-1, Math.min(1, concatenated[i]));
            pcmData[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
        }
        
        // Send the audio data to the server
        const message = {
            type: 'audio',
            data: Array.from(pcmData), // Convert to regular array for JSON serialization
            sampleRate: this.sampleRate
        };
        
        this.socket.send(JSON.stringify(message));
        
        // Clear the audio chunks
        this.audioChunks = [];
    }
    
    sendLanguageSelection() {
        if (!this.socketReady) return;
        
        const message = {
            type: 'language',
            sourceLanguage: this.sourceLanguageSelect.value,
            targetLanguage: this.targetLanguageSelect.value
        };
        
        this.socket.send(JSON.stringify(message));
    }
    
    addMessage(element, text, className = '') {
        const paragraph = document.createElement('p');
        paragraph.textContent = text;
        
        if (className) {
            paragraph.classList.add(className);
        }
        
        element.appendChild(paragraph);
        element.scrollTop = element.scrollHeight; // Auto-scroll to bottom
    }
    
    clearDisplays() {
        this.transcriptionDisplay.innerHTML = '';
        this.translationDisplay.innerHTML = '';
    }
    
    updateStatus(message, isActive) {
        this.statusText.textContent = message;
        
        if (isActive) {
            this.statusIndicator.className = 'status-active';
        } else {
            this.statusIndicator.className = 'status-inactive';
        }
    }
}

// Initialize the application when the DOM is fully loaded
document.addEventListener('DOMContentLoaded', () => {
    const app = new LiveTranslator();
});