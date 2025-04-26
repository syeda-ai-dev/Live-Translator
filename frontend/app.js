class AudioRecorder {
    constructor() {
        this.mediaRecorder = null;
        this.audioContext = null;
        this.audioStream = null;
        this.audioProcessor = null;
        this.isRecording = false;
        this.ws = null;
        this.chunkInterval = 800; // ms
        this.debugMode = true; // Enable debugging
        this.connectionAttempts = 0;
        this.maxConnectionAttempts = 5;
        
        // WebSocket configuration
        this.wsBaseUrl = 'ws://127.0.0.1:8000'; // Default port when running with uvicorn
        this.wsEndpoint = '/api/ws/speech';
        this.wsUrl = this.wsBaseUrl + this.wsEndpoint;
        
        // UI elements
        this.startButton = document.getElementById('startButton');
        this.stopButton = document.getElementById('stopButton');
        this.transcriptionOutput = document.getElementById('transcriptionOutput');
        this.translationOutput = document.getElementById('translationOutput');
        this.sourceLanguage = document.getElementById('sourceLanguage');
        this.targetLanguage = document.getElementById('targetLanguage');
        this.swapLanguagesButton = document.getElementById('swapLanguages');
        this.connectionStatus = document.getElementById('connectionStatus');
        this.audioIndicator = document.querySelector('.audio-indicator');
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 3;
        
        console.log("Audio recorder initialized");
        console.log(`WebSocket URL: ${this.wsUrl}`);
        
        // Bind event listeners
        this.startButton.addEventListener('click', () => this.startRecording());
        this.stopButton.addEventListener('click', () => this.stopRecording());
        
        // Handle language change
        this.sourceLanguage.addEventListener('change', () => this.updateLanguageConfig());
        this.targetLanguage.addEventListener('change', () => this.updateLanguageConfig());
        
        // Handle language swap
        this.swapLanguagesButton.addEventListener('click', () => this.swapLanguages());
        
        // Test WebSocket connectivity immediately
        this.testWebSocketConnection();
    }
    
    updateConnectionStatus(status) {
        const statusDot = this.connectionStatus.querySelector('.status-dot');
        const statusText = this.connectionStatus.querySelector('.status-text');
        
        statusDot.classList.remove('connected', 'connecting', 'disconnected');
        
        switch(status) {
            case 'connected':
                statusDot.classList.add('connected');
                statusText.textContent = 'Connected';
                break;
            case 'connecting':
                statusDot.classList.add('connecting');
                statusText.textContent = 'Connecting...';
                break;
            case 'disconnected':
                statusDot.classList.add('disconnected');
                statusText.textContent = 'Disconnected';
                break;
            default:
                statusDot.classList.add('disconnected');
                statusText.textContent = 'Unknown';
        }
    }
    
    swapLanguages() {
        const sourceValue = this.sourceLanguage.value;
        const targetValue = this.targetLanguage.value;
        
        this.sourceLanguage.value = targetValue;
        this.targetLanguage.value = sourceValue;
        
        // Update config if connected
        this.updateLanguageConfig();
    }
    
    updateLanguageConfig() {
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify({
                type: 'config',
                sourceLang: this.sourceLanguage.value,
                targetLang: this.targetLanguage.value
            }));
            console.log(`Updated language config: source=${this.sourceLanguage.value}, target=${this.targetLanguage.value}`);
        }
    }

    // Test WebSocket connection to help with debugging
    async testWebSocketConnection() {
        try {
            this.updateConnectionStatus('connecting');
            console.log(`Testing WebSocket connection to ${this.wsUrl}...`);
            const testWs = new WebSocket(this.wsUrl);
            
            // Set timeout for connection
            const timeout = setTimeout(() => {
                console.error("❌ WebSocket connection test timed out");
                testWs.close();
                this.updateConnectionStatus('disconnected');
                this.showError("Connection to speech service timed out. Please check if the server is running at " + this.wsBaseUrl);
            }, 3000);
            
            testWs.onopen = () => {
                clearTimeout(timeout);
                console.log("✅ WebSocket connection test successful");
                this.updateConnectionStatus('connected');
                testWs.close();
            };
            
            testWs.onerror = (error) => {
                clearTimeout(timeout);
                console.error("❌ WebSocket connection test failed:", error);
                this.updateConnectionStatus('disconnected');
                this.showError(`Unable to connect to speech service at ${this.wsUrl}. Please check if the server is running.`);
            };
        } catch (error) {
            console.error("Error testing WebSocket connection:", error);
            this.updateConnectionStatus('disconnected');
        }
    }

    async setupWebSocket() {
        try {
            this.connectionAttempts++;
            this.updateConnectionStatus('connecting');
            console.log(`Connecting to WebSocket (attempt ${this.connectionAttempts}/${this.maxConnectionAttempts})...`);
            
            // Close existing connection if any
            if (this.ws) {
                this.ws.close();
                this.ws = null;
            }
            
            this.ws = new WebSocket(this.wsUrl);
            
            this.ws.onopen = () => {
                console.log('✅ WebSocket connected');
                this.updateConnectionStatus('connected');
                this.reconnectAttempts = 0;
                this.connectionAttempts = 0;
                // Send initial configuration
                this.ws.send(JSON.stringify({
                    type: 'config',
                    sourceLang: this.sourceLanguage.value,
                    targetLang: this.targetLanguage.value
                }));
                console.log(`Sent config: source=${this.sourceLanguage.value}, target=${this.targetLanguage.value}`);
            };
            
            this.ws.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data);
                    console.log("Received message:", data);
                    
                    if (data.type === 'result') {
                        this.updateUI(data.source_text, data.translated_text);
                    } else if (data.type === 'error') {
                        this.showError(data.message);
                    } else if (data.type === 'ready') {
                        console.log("Server ready to receive audio");
                    } else if (data.type === 'heartbeat') {
                        console.log("Received heartbeat from server");
                    }
                } catch (error) {
                    console.error("Error parsing message:", error);
                }
            };
            
            this.ws.onerror = async (error) => {
                console.error('WebSocket error:', error);
                this.updateConnectionStatus('disconnected');
                if (this.reconnectAttempts < this.maxReconnectAttempts) {
                    this.reconnectAttempts++;
                    console.log(`Attempting to reconnect (${this.reconnectAttempts}/${this.maxReconnectAttempts})...`);
                    await new Promise(resolve => setTimeout(resolve, 1000));
                    await this.setupWebSocket();
                } else {
                    this.showError('Connection failed. Please try again later.');
                    this.stopRecording();
                }
            };
            
            this.ws.onclose = () => {
                console.log('WebSocket closed');
                this.updateConnectionStatus('disconnected');
                if (this.isRecording) {
                    this.stopRecording();
                }
            };
        } catch (error) {
            console.error('Failed to connect to server:', error);
            this.updateConnectionStatus('disconnected');
            this.showError('Failed to connect to server. Please try again.');
            this.stopRecording();
        }
    }

    showError(message) {
        console.error("ERROR:", message);
        this.transcriptionOutput.innerHTML = `<span style="color: red;">Error: ${message}</span>`;
    }

    async startRecording() {
        console.log("Starting recording...");
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ 
                audio: {
                    echoCancellation: true,
                    noiseSuppression: true,
                    autoGainControl: true
                } 
            });
            
            // Set up audio context and processor
            this.audioContext = new (window.AudioContext || window.webkitAudioContext)();
            console.log(`Audio context created. Sample rate: ${this.audioContext.sampleRate}Hz`);
            this.audioStream = stream;
            
            // Create audio source from microphone
            const source = this.audioContext.createMediaStreamSource(stream);
            console.log("Audio source created from microphone");
            
            // Smaller buffer size for more frequent updates (1024 is better for real-time)
            const bufferSize = 1024;
            console.log(`Using buffer size: ${bufferSize}`);
            
            // Connect WebSocket first before processing audio
            await this.setupWebSocket();
                
            // For older browsers using ScriptProcessor
            if (this.audioContext.createScriptProcessor) {
                this.audioProcessor = this.audioContext.createScriptProcessor(bufferSize, 1, 1);
                console.log("Created ScriptProcessor");
                
                // Send audio chunks more frequently (every 100ms)
                let lastSendTime = 0;
                let accumulatedChunks = new Float32Array();
                
                this.audioProcessor.onaudioprocess = (event) => {
                    if (this.ws && this.ws.readyState === WebSocket.OPEN && this.isRecording) {
                        try {
                            // Get audio data from buffer
                            const audioData = event.inputBuffer.getChannelData(0);
                            
                            // Check if audio contains actual sound (not just silence)
                            const isSilent = this.isAudioSilent(audioData);
                            
                            // Update audio visualization
                            this.updateAudioIndicator(!isSilent);
                            
                            // Send audio data immediately to reduce latency
                            if (!isSilent || Date.now() - lastSendTime > 300) {
                                lastSendTime = Date.now();
                                const floatArray = new Float32Array(audioData);
                                this.ws.send(floatArray);
                            }
                            
                        } catch (error) {
                            console.error('Error sending audio data:', error);
                            this.showError('Error sending audio data');
                        }
                    }
                };
                
                // Connect the nodes
                source.connect(this.audioProcessor);
                this.audioProcessor.connect(this.audioContext.destination);
            } 
            // For newer browsers with AudioWorklet
            else if (this.audioContext.audioWorklet) {
                console.log('Using AudioWorklet for audio processing');
                // This is a future improvement - stick with ScriptProcessor for now
                this.showError('Your browser is using newer audio API. Please use Chrome for best compatibility.');
                return;
            }
            else {
                this.showError('Web Audio API not fully supported in this browser');
                return;
            }
            
            this.isRecording = true;
            
            // Update UI
            this.startButton.disabled = true;
            this.stopButton.disabled = false;
            this.transcriptionOutput.textContent = 'Listening...';
            this.translationOutput.textContent = '';
            
        } catch (error) {
            console.error('Error starting recording:', error);
            this.showError('Could not access microphone. Please ensure you have granted microphone permissions.');
        }
    }

    stopRecording() {
        if (this.audioStream) {
            // Stop all audio tracks
            this.audioStream.getTracks().forEach(track => track.stop());
            this.audioStream = null;
            
            // Clean up audio context
            if (this.audioProcessor) {
                this.audioProcessor.disconnect();
                this.audioProcessor = null;
            }
            
            if (this.audioContext && this.audioContext.state !== 'closed') {
                this.audioContext.close();
                this.audioContext = null;
            }
            
            this.isRecording = false;
            this.updateAudioIndicator(false);
        }
        
        if (this.ws) {
            this.ws.close();
            this.ws = null;
        }
        
        // Update UI
        this.startButton.disabled = false;
        this.stopButton.disabled = true;
    }

    updateAudioIndicator(isActive) {
        if (isActive) {
            this.audioIndicator.classList.add('active');
        } else {
            this.audioIndicator.classList.remove('active');
        }
    }

    updateUI(transcription, translation) {
        this.transcriptionOutput.textContent = transcription;
        this.translationOutput.textContent = translation;
    }

    isAudioSilent(audioData) {
        const rms = this.getRMS(audioData);
        return rms < 0.005; // Lower threshold to detect more audio
    }
    
    getRMS(audioData) {
        let sum = 0;
        for (let i = 0; i < audioData.length; i++) {
            sum += audioData[i] * audioData[i];
        }
        return Math.sqrt(sum / audioData.length);
    }
}

// Initialize the application
document.addEventListener('DOMContentLoaded', () => {
    const recorder = new AudioRecorder();
});