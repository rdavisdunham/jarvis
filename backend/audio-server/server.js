const express = require('express');
const multer = require('multer');
const path = require('path');
const ffmpeg = require('fluent-ffmpeg');
const fs = require('fs');

const { spawn } = require('child_process');

// Spawn the Python script
const pythonScript = process.env.JARVIS_SCRIPT || path.join(__dirname, '..', 'JARVIS.py');
const pythonProcess = spawn('python', [pythonScript]);

// Message queue for responses from Python
let messageQueue = [];
let outputBuffer = '';
let pythonReady = false;
let pendingMessages = []; // Messages received before Python is ready

// SSE client management for real-time audio notifications
let sseClients = [];

function broadcastEvent(eventType, data) {
  sseClients.forEach(client => {
    client.res.write(`event: ${eventType}\ndata: ${JSON.stringify(data)}\n\n`);
  });
}

// Listen for data from the Python script
pythonProcess.stdout.on('data', (data) => {
  const output = data.toString();
  console.log('Python script output:', output);

  // Buffer the output and split by newlines to get complete messages
  outputBuffer += output;
  const lines = outputBuffer.split('\n');

  // Keep the last incomplete line in the buffer
  outputBuffer = lines.pop();

  // Process complete lines
  for (const line of lines) {
    const trimmed = line.trim();
    if (trimmed === 'READY') {
      console.log('Python script is ready');
      pythonReady = true;
      // Send any pending messages
      for (const msg of pendingMessages) {
        pythonProcess.stdin.write(`TEXT:${msg}\n`);
      }
      pendingMessages = [];
    } else if (trimmed.startsWith('Audio:')) {
      // TTS audio file notification - push via SSE for real-time delivery
      const audioFile = trimmed.substring(6).trim();
      broadcastEvent('audio', { file: audioFile });
      console.log('TTS audio pushed via SSE:', audioFile);
    } else if (trimmed) {
      messageQueue.push(trimmed);
    }
  }
});

pythonProcess.stderr.on('data', (data) => {
  console.error('Python script info:', data.toString());
});

const app = express();
const port = 3000; // You can change the port number if needed

const cors = require('cors');
app.use(cors());
app.use(express.json());

// Serve audio output files for TTS
app.use('/audio-output', express.static(path.join(__dirname, 'outputs')));

// Set up multer for handling file uploads
const storage = multer.diskStorage({
  destination: function (req, file, cb) {
    cb(null, 'uploads/'); // Specify the directory where uploaded files will be stored
  },
  filename: function (req, file, cb) {
    cb(null, Date.now() + path.extname(file.originalname)); // Generate a unique filename
  }
});

const upload = multer({ storage: storage });

// Handle POST request for audio upload
app.post('/upload', upload.single('audio'), (req, res) => {
  if (!req.file) {
    return res.status(400).send('No audio file uploaded');
  }

  console.log('Audio file received:', req.file.filename);

  // Convert the audio file to WAV format
  const inputFile = path.join(__dirname, 'uploads', req.file.filename);
  const outputFile = path.join(__dirname, 'uploads', path.parse(req.file.filename).name + '.wav');

  // Log input file size
  const inputStats = fs.statSync(inputFile);
  console.log(`Input file size: ${inputStats.size} bytes`);

  ffmpeg(inputFile)
    .audioCodec('pcm_s16le')  // Standard WAV codec
    .audioFrequency(16000)    // Whisper expects 16kHz
    .audioChannels(1)         // Mono
    .output(outputFile)
    .on('end', () => {
      // Log output file size
      const outputStats = fs.statSync(outputFile);
      console.log(`Output WAV size: ${outputStats.size} bytes`);
      console.log('Audio file converted to WAV');

      // Delete the original audio file
      fs.unlink(inputFile, (err) => {
        if (err) {
          console.error('Error deleting original audio file:', err);
        } else {
          console.log('Original file deleted');
        }
      });

      res.send('Audio file uploaded and converted successfully');
    })
    .on('error', (err) => {
      console.error('Error converting audio file:', err);
      res.status(500).send('Error converting audio file');
    })
    .run();
});

// Handle POST request for text messages
app.post('/text-message', (req, res) => {
  const { message } = req.body;
  if (!message) {
    return res.status(400).send('No message provided');
  }

  if (pythonReady) {
    // Send the text message to the Python script via stdin
    pythonProcess.stdin.write(`TEXT:${message}\n`);
    res.send('Message sent');
  } else {
    // Queue the message until Python is ready
    console.log('Python not ready, queuing message:', message);
    pendingMessages.push(message);
    res.send('Message queued');
  }
});

// Health check endpoint
app.get('/health', (req, res) => {
  res.json({ ready: pythonReady });
});

// SSE endpoint for real-time audio notifications
app.get('/events', (req, res) => {
  res.setHeader('Content-Type', 'text/event-stream');
  res.setHeader('Cache-Control', 'no-cache');
  res.setHeader('Connection', 'keep-alive');
  res.setHeader('Access-Control-Allow-Origin', '*');

  // Send initial connection confirmation
  res.write('event: connected\ndata: {}\n\n');

  const clientId = Date.now();
  sseClients.push({ id: clientId, res });
  console.log(`SSE client connected: ${clientId}, total clients: ${sseClients.length}`);

  req.on('close', () => {
    sseClients = sseClients.filter(c => c.id !== clientId);
    console.log(`SSE client disconnected: ${clientId}, total clients: ${sseClients.length}`);
  });
});

// Handle POST request to update TTS setting for audio responses
app.post('/tts-setting', (req, res) => {
  const { enabled } = req.body;
  if (typeof enabled !== 'boolean') {
    return res.status(400).send('Invalid TTS setting');
  }

  if (pythonReady) {
    // Send the TTS setting to the Python script
    pythonProcess.stdin.write(`TTS_SETTING:${enabled ? 'on' : 'off'}\n`);
    res.json({ success: true, enabled });
  } else {
    res.status(503).send('Python not ready');
  }
});

// Handle GET request to retrieve the text output
app.get('/text-output', (req, res) => {
  if (messageQueue.length > 0) {
    // Return all queued messages and clear the queue
    const messages = messageQueue.join('\n');
    messageQueue = [];
    res.send(messages);
  } else {
    res.send('');
  }
});

// Start the server
app.listen(port, () => {
  console.log(`Server is running on port ${port}`);
});