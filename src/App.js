import React, { useState, useEffect, useRef } from 'react';
import { ReactMediaRecorder } from 'react-media-recorder';
import axios from 'axios';

const App = () => {
  const [mediaRecorderKey, setMediaRecorderKey] = useState(Date.now());
  const [textOutput, setTextOutput] = useState([]); // Now an array of messages
  const [isRecording, setIsRecording] = useState(false); // New state to track recording status
  const messagesEndRef = useRef(null); // Create a ref for the messages container
  const [userInput, setUserInput] = useState('');

  // Scroll to the bottom function
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  // Use useEffect to scroll to the bottom whenever textOutput changes
  useEffect(() => {
    scrollToBottom();
  }, [textOutput]);

  const sendTextMessage = async () => {
    if (userInput.trim() !== '') {
      try {
        setTextOutput(prevTextOutput => [...prevTextOutput, `User: ${userInput}`]); // Add user's input to textOutput
        await axios.post('http://192.168.1.189:3000/text-message', { message: userInput });
        setUserInput(''); // Clear the input field after sending
        pollForTextOutput(); // Start polling for the response
      } catch (error) {
        console.error('Error sending text message:', error);
      }
    }
  };

  const handleStop = async (blobUrl, blob) => {
    setMediaRecorderKey(Date.now()); // Reset the ReactMediaRecorder component
    setIsRecording(false); // Update recording status
    await handleUpload(blob); // Call handleUpload directly after stopping the recording
  };

  const handleUpload = async (blob) => {
    if (blob) {
      const formData = new FormData();
      formData.append('audio', blob, 'recording.webm');

      try {
        await axios.post('http://192.168.1.189:3000/upload', formData, {
          headers: { 'Content-Type': 'multipart/form-data' },
        });
        console.log('Audio uploaded successfully');
        pollForTextOutput(); // Start polling for text output after successful upload
      } catch (error) {
        console.error('Error uploading audio:', error);
      }
    }
  };

  const pollForTextOutput = async () => {
    const maxAttempts = 50; // Maximum number of attempts to check for text output
    const interval = 500; // Interval between attempts in milliseconds

    let attempts = 0;

    const checkForTextOutput = async () => {
      try {
        const response = await axios.get('http://192.168.1.189:3000/text-output');
        const newTextOutput = response.data;
        if (newTextOutput) { // Assuming the server returns an empty response or specific status if not ready
          console.log('Text output from server:', newTextOutput);
          setTextOutput(prevTextOutput => [...prevTextOutput, newTextOutput]); // Append new message
          return; // Stop polling since we got the output
        }
        throw new Error('Text output not ready'); // Trigger retry
      } catch (error) {
        console.error('Error retrieving text output:', error);
        if (++attempts < maxAttempts) {
          setTimeout(checkForTextOutput, interval); // Schedule next attempt
        } else {
          console.error('Maximum attempts reached, stopping poll for text output.');
        }
      }
    };

    checkForTextOutput(); // Start the polling process
  };

  const toggleRecording = (start, stop) => {
    if (isRecording) {
      stop();
    } else {
      start();
    }
    setIsRecording(!isRecording); // Toggle the recording status
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter') {
      sendTextMessage();
    }
  };

  return (
    <div>
      <ReactMediaRecorder
        key={mediaRecorderKey} // Use a unique key to force remount
        audio
        onStop={handleStop}
        render={({ status, startRecording, stopRecording }) => (
          <div>
            <p>{status}</p>
            <button onClick={() => toggleRecording(startRecording, stopRecording)}>
              {isRecording ? 'Stop Recording' : 'Start Recording'}
            </button>
          </div>
        )}
      />
      <div
        style={{
          height: '80vh',
          width: '90vw',
          overflowY: 'scroll',
          backgroundColor: '#f0f0f0',
          padding: '10px',
          boxSizing: 'border-box',
          resize: 'none',
          marginTop: '10px',
          border: '1px solid #ccc',
        }}
      >
        {textOutput.map((message, index) => (
          <div key={index}>
            {message.split('\n').reduce((acc, line, lineIndex) => {
              // Determine the color based on the line prefix or carry over the previous color
              const color = line.startsWith('User:') ? 'black' :
                            line.startsWith('Model:') ? 'blue' : acc.prevColor;

              acc.lines.push(
                <div
                  key={lineIndex}
                  style={{
                    color: color,
                    marginBottom: '10px',
                  }}
                >
                  {line}
                </div>
              );

              // Update the previous color for the next iteration
              acc.prevColor = color;
              return acc;
            }, { lines: [], prevColor: 'black' }).lines // Initialize with black as the default color
          }
          </div>
        ))}
        <div ref={messagesEndRef} /> {/* Invisible element at the end of the messages */}
      </div>
      <div style={{ display: 'flex', marginTop: '10px' }}>
        <input
          type="text"
          value={userInput}
          onChange={(e) => setUserInput(e.target.value)}
          onKeyDown={handleKeyDown} // Add this line
          placeholder="Type your message..."
          style={{ flex: 1, marginRight: '10px' }}
        />
        <button onClick={sendTextMessage}>Send</button>
      </div>
    </div>
  );
};

export default App;