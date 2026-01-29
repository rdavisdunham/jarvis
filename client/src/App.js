import React, { useState, useEffect, useRef } from 'react';
import { ReactMediaRecorder } from 'react-media-recorder';
import axios from 'axios';
import './App.css';

const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:3000';

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

const App = () => {
  const [mediaRecorderKey, setMediaRecorderKey] = useState(Date.now());
  const [messages, setMessages] = useState([]);
  const [isRecording, setIsRecording] = useState(false);
  const messagesEndRef = useRef(null);
  const [userInput, setUserInput] = useState('');
  const [isWaitingForResponse, setIsWaitingForResponse] = useState(false);
  const [backendReady, setBackendReady] = useState(false);
  const [audioQueue, setAudioQueue] = useState([]);
  const [isPlayingAudio, setIsPlayingAudio] = useState(false);
  const audioRef = useRef(null);

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

  const handleStop = async (blobUrl, blob) => {
    setMediaRecorderKey(Date.now());
    setIsRecording(false);
    await handleUpload(blob);
  };

  const handleUpload = async (blob) => {
    if (blob && !isWaitingForResponse) {
      setIsWaitingForResponse(true);
      const formData = new FormData();
      formData.append('audio', blob, 'recording.webm');

      try {
        await axios.post(`${API_URL}/upload`, formData, {
          headers: { 'Content-Type': 'multipart/form-data' },
        });
        await pollForTextOutput();
      } catch (error) {
        console.error('Error uploading audio:', error);
      } finally {
        setIsWaitingForResponse(false);
      }
    }
  };

  const pollForTextOutput = async () => {
    const maxAttempts = 60;
    const interval = 500;
    let hasModelResponse = false;
    let audioPollingAttempts = 0;
    const maxAudioPollingAttempts = 20; // Keep polling for audio after model response

    for (let attempts = 0; attempts < maxAttempts; attempts++) {
      try {
        const response = await axios.get(`${API_URL}/text-output`);
        const newTextOutput = response.data;

        if (newTextOutput && newTextOutput.trim()) {
          const newMessages = [];
          const lines = newTextOutput.split('\n');
          let gotAudioThisPoll = false;

          for (const line of lines) {
            const trimmed = line.trim();
            if (trimmed.startsWith('AUDIO:')) {
              // Queue audio file for playback
              const audioFile = trimmed.substring(6).trim();
              setAudioQueue(prev => [...prev, audioFile]);
              gotAudioThisPoll = true;
            } else if (trimmed.startsWith('Transcribed Text:')) {
              newMessages.push({ type: 'user', content: trimmed.substring(17).trim() });
            } else if (trimmed.startsWith('Model:')) {
              newMessages.push({ type: 'model', content: trimmed.substring(6).trim() });
              hasModelResponse = true;
            } else if (trimmed && newMessages.length > 0) {
              newMessages[newMessages.length - 1].content += '\n' + trimmed;
            }
          }

          if (newMessages.length > 0) {
            setMessages(prev => [...prev, ...newMessages]);
          }

          // If we got audio after model response, we're done
          if (hasModelResponse && gotAudioThisPoll) {
            return;
          }
        }

        // After getting model response, keep polling briefly for audio
        if (hasModelResponse) {
          audioPollingAttempts++;
          if (audioPollingAttempts >= maxAudioPollingAttempts) {
            return; // Timeout waiting for audio, but we have the text
          }
        }

        await new Promise(resolve => setTimeout(resolve, interval));
      } catch (error) {
        console.error('Error retrieving text output:', error);
        await new Promise(resolve => setTimeout(resolve, interval));
      }
    }
  };

  const toggleRecording = (start, stop) => {
    if (isRecording) {
      stop();
    } else {
      start();
    }
    setIsRecording(!isRecording);
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
                {message.content}
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
        <ReactMediaRecorder
          key={mediaRecorderKey}
          audio
          onStop={handleStop}
          render={({ startRecording, stopRecording }) => (
            <button
              className={`btn btn-mic ${isRecording ? 'recording' : ''}`}
              onClick={() => toggleRecording(startRecording, stopRecording)}
              disabled={isWaitingForResponse}
              title={isRecording ? 'Stop recording' : 'Start recording'}
            >
              {isRecording ? <StopIcon /> : <MicIcon />}
            </button>
          )}
        />

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
