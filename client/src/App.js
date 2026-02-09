import React, { useState, useEffect, useRef } from 'react';
import { useMicVAD } from '@ricky0123/vad-react';
import axios from 'axios';
import './App.css';
import { transcribe, sendTranscribedText, getSTTMode } from './utils/sttService';

// Dynamically build the URL based on the current browser address
const API_URL = `${window.location.protocol}//${window.location.hostname}:3000`;

// Strip bracketed annotations like [warmly], [laughs], etc. from display text
const stripBrackets = (text) => text.replace(/\[.*?\]\s*/g, '').trim();

// Simple SVG Icons as components
const SendIcon = () => (
  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <line x1="22" y1="2" x2="11" y2="13"></line>
    <polygon points="22 2 15 22 11 13 2 9 22 2"></polygon>
  </svg>
);

const MicIcon = () => (
  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"></path>
    <path d="M19 10v2a7 7 0 0 1-14 0v-2"></path>
    <line x1="12" y1="19" x2="12" y2="23"></line>
    <line x1="8" y1="23" x2="16" y2="23"></line>
  </svg>
);

const StopIcon = () => (
  <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor">
    <rect x="4" y="4" width="16" height="16" rx="2"></rect>
  </svg>
);

const SpeakerOnIcon = () => (
  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"></polygon>
    <path d="M19.07 4.93a10 10 0 0 1 0 14.14"></path>
    <path d="M15.54 8.46a5 5 0 0 1 0 7.07"></path>
  </svg>
);

const SpeakerOffIcon = () => (
  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"></polygon>
    <line x1="23" y1="9" x2="17" y2="15"></line>
    <line x1="17" y1="9" x2="23" y2="15"></line>
  </svg>
);

const App = () => {
  const [messages, setMessages] = useState([]);
  const [vadState, setVadState] = useState('idle'); // 'idle' | 'listening' | 'speaking'
  const messagesEndRef = useRef(null);
  const [userInput, setUserInput] = useState('');
  const [isWaitingForResponse, setIsWaitingForResponse] = useState(false);
  const [backendReady, setBackendReady] = useState(false);
  const [audioQueue, setAudioQueue] = useState([]);
  const [isPlayingAudio, setIsPlayingAudio] = useState(false);
  const audioRef = useRef(null);
  const cancelledBeforeRef = useRef(0);  // Timestamp: ignore audio from responses before this
  const [ttsEnabled, setTtsEnabled] = useState(true);

  useEffect(() => {
    const checkBackendReady = async () => {
      try {
        const response = await axios.get(`${API_URL}/health`);
        if (response.data.ready) {
          setBackendReady(true);
          return;
        }
      } catch (error) {
        // Backend not reachable yet
      }
      setTimeout(checkBackendReady, 1000);
    };
    checkBackendReady();
  }, []);

  // SSE connection for real-time audio notifications
  useEffect(() => {
    if (backendReady) {
      const eventSource = new EventSource(`${API_URL}/events`);

      eventSource.addEventListener('audio', (event) => {
        const data = JSON.parse(event.data);
        const filename = data.file;

        // Extract response ID (timestamp) from filename: response_TIMESTAMP_INDEX.wav
        const match = filename.match(/^response_(\d+)_\d+\.wav$/);
        if (match) {
          const responseTimestamp = parseInt(match[1]);
          // Ignore chunks from cancelled/old responses
          if (responseTimestamp <= cancelledBeforeRef.current) {
            console.log(`Ignoring cancelled audio: ${filename}`);
            return;
          }
        }

        console.log('SSE audio received:', filename);
        setAudioQueue(prev => [...prev, filename]);
      });

      eventSource.addEventListener('connected', () => {
        console.log('SSE connected to backend');
      });

      eventSource.onerror = () => {
        console.warn('SSE connection error, will auto-reconnect...');
      };

      return () => {
        eventSource.close();
      };
    }
  }, [backendReady]);

  // Audio playback effect - play queued audio files sequentially
  useEffect(() => {
    if (audioQueue.length > 0 && !isPlayingAudio) {
      const playNextAudio = () => {
        const nextAudioFile = audioQueue[0];
        const audio = new Audio(`${API_URL}/audio-output/${nextAudioFile}`);
        audioRef.current = audio;
        setIsPlayingAudio(true);

        audio.onended = () => {
          setAudioQueue(prev => prev.slice(1));
          setIsPlayingAudio(false);
          audioRef.current = null;
        };

        audio.onerror = (e) => {
          console.error('Audio playback error:', e);
          setAudioQueue(prev => prev.slice(1));
          setIsPlayingAudio(false);
          audioRef.current = null;
        };

        audio.play().catch(err => {
          console.error('Failed to play audio:', err);
          setAudioQueue(prev => prev.slice(1));
          setIsPlayingAudio(false);
        });
      };

      playNextAudio();
    }
  }, [audioQueue, isPlayingAudio]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const toggleTts = async () => {
    const newValue = !ttsEnabled;
    setTtsEnabled(newValue);

    // If turning TTS off, stop any currently playing audio
    if (!newValue) {
      if (audioRef.current) {
        audioRef.current.pause();
        audioRef.current = null;
      }
      setAudioQueue([]);
      setIsPlayingAudio(false);
      // Cancel all audio from responses generated before now
      cancelledBeforeRef.current = Date.now();
    }

    try {
      await axios.post(`${API_URL}/tts-setting`, { enabled: newValue });
    } catch (error) {
      console.error('Error updating TTS setting:', error);
    }
  };

  const sendTextMessage = async () => {
    if (userInput.trim() !== '' && !isWaitingForResponse) {
      const message = userInput;
      setUserInput('');
      setIsWaitingForResponse(true);

      try {
        setMessages(prev => [...prev, { type: 'user', content: message }]);
        await axios.post(`${API_URL}/text-message`, { message });
        await pollForTextOutput();
      } catch (error) {
        console.error('Error sending text message:', error);
      } finally {
        setIsWaitingForResponse(false);
      }
    }
  };

  // VAD hook for voice activity detection
  const vad = useMicVAD({
    startOnLoad: false,
    onSpeechStart: () => {
      console.log('Speech started');
      setVadState('speaking');
    },
    onSpeechEnd: async (audio) => {
      console.log('Speech ended');
      setVadState('idle');
      vad.pause();

      if (isWaitingForResponse) return;
      setIsWaitingForResponse(true);

      try {
        const sttMode = getSTTMode();

        if (sttMode === 'browser') {
          // Browser STT: transcribe locally, send text
          const text = await transcribe(audio);
          if (text) {
            setMessages(prev => [...prev, { type: 'user', content: text }]);
            await sendTranscribedText(text);
            await pollForTextOutput(true); // Skip user message parsing
          }
        } else {
          // API STT: upload WAV, server transcribes
          await transcribe(audio);
          await pollForTextOutput(false);
        }
      } catch (error) {
        console.error('STT error:', error);
      } finally {
        setIsWaitingForResponse(false);
      }
    },
    onVADMisfire: () => {
      console.log('VAD misfire - speech too short');
    },
  });

  const toggleVAD = () => {
    if (vadState === 'idle') {
      // Stop any currently playing audio
      if (audioRef.current) {
        audioRef.current.pause();
        audioRef.current = null;
      }
      setAudioQueue([]);
      setIsPlayingAudio(false);
      // Cancel all audio from responses generated before now
      cancelledBeforeRef.current = Date.now();

      vad.start();
      setVadState('listening');
    } else {
      vad.pause();
      setVadState('idle');
    }
  };

  const pollForTextOutput = async (skipUserMessage = false) => {
    // Poll for text messages only - audio arrives via SSE independently
    const maxAttempts = 60;
    const interval = 500;

    for (let attempts = 0; attempts < maxAttempts; attempts++) {
      try {
        const response = await axios.get(`${API_URL}/text-output`);
        const newTextOutput = response.data;

        if (newTextOutput && newTextOutput.trim()) {
          const newMessages = [];
          const lines = newTextOutput.split('\n');

          for (const line of lines) {
            const trimmed = line.trim();
            if (trimmed.startsWith('Transcribed Text:')) {
              // Skip user message if we already added it (browser STT mode)
              if (!skipUserMessage) {
                newMessages.push({ type: 'user', content: trimmed.substring(17).trim() });
              }
            } else if (trimmed.startsWith('Model:')) {
              newMessages.push({ type: 'model', content: trimmed.substring(6).trim() });
            } else if (trimmed && newMessages.length > 0) {
              newMessages[newMessages.length - 1].content += '\n' + trimmed;
            }
          }

          if (newMessages.length > 0) {
            setMessages(prev => [...prev, ...newMessages]);
          }

          // Return once we have the model response - audio arrives via SSE
          if (newMessages.some(m => m.type === 'model')) {
            return;
          }
        }

        await new Promise(resolve => setTimeout(resolve, interval));
      } catch (error) {
        console.error('Error retrieving text output:', error);
        await new Promise(resolve => setTimeout(resolve, interval));
      }
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendTextMessage();
    }
  };

  if (!backendReady) {
    return (
      <div className="loading-screen">
        <h2>J.A.R.V.I.S.</h2>
        <p>Initializing AI systems...</p>
        <div className="loader"></div>
      </div>
    );
  }

  return (
    <div className="chat-container">
      {/* Header */}
      <div className="chat-header">
        <div className="header-avatar">J</div>
        <div className="header-info">
          <h1>J.A.R.V.I.S.</h1>
          <div className="header-status">
            <span className="status-dot"></span>
            Online
          </div>
        </div>
        <button
          className={`btn-tts-toggle ${ttsEnabled ? 'enabled' : ''}`}
          onClick={toggleTts}
          title={ttsEnabled ? 'Voice responses on' : 'Voice responses off'}
        >
          {ttsEnabled ? <SpeakerOnIcon /> : <SpeakerOffIcon />}
        </button>
      </div>

      {/* Messages */}
      <div className="messages-container">
        {messages.length === 0 ? (
          <div className="empty-state">
            <div className="empty-state-icon">💬</div>
            <h3>Start a conversation</h3>
            <p>Type a message or use voice input</p>
          </div>
        ) : (
          messages.map((message, index) => (
            <div key={index} className={`message ${message.type}`}>
              <div className="message-label">
                {message.type === 'user' ? 'You' : 'JARVIS'}
              </div>
              <div className="message-bubble">
                {message.type === 'model' ? stripBrackets(message.content) : message.content}
              </div>
            </div>
          ))
        )}

        {/* Typing Indicator */}
        {isWaitingForResponse && (
          <div className="message model">
            <div className="message-label">JARVIS</div>
            <div className="typing-indicator">
              <span></span>
              <span></span>
              <span></span>
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Input Area */}
      <div className="input-container">
        <button
          className={`btn btn-mic ${vadState !== 'idle' ? 'recording' : ''} ${vadState === 'speaking' ? 'speaking' : ''}`}
          onClick={toggleVAD}
          disabled={isWaitingForResponse}
          title={
            vadState === 'idle' ? 'Click to listen' :
            vadState === 'listening' ? 'Listening for speech...' :
            'Recording...'
          }
        >
          {vadState === 'idle' ? <MicIcon /> : <StopIcon />}
        </button>

        <input
          type="text"
          className="text-input"
          value={userInput}
          onChange={(e) => setUserInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Type a message..."
        />

        <button
          className="btn btn-send"
          onClick={sendTextMessage}
          disabled={isWaitingForResponse || !userInput.trim()}
          title="Send message"
        >
          <SendIcon />
        </button>
      </div>
    </div>
  );
};

export default App;
