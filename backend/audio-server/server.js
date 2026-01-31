const express = require('express');
const multer = require('multer');
const path = require('path');
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
        const prefix = msg.enableTts ? 'TEXT_TTS:' : 'TEXT:';
        pythonProcess.stdin.write(`${prefix}${msg.message}\n`);
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
// Now expects WAV files directly from browser (no FFmpeg conversion needed)
app.post('/upload', upload.single('audio'), (req, res) => {
  if (!req.file) {
    return res.status(400).send('No audio file uploaded');
  }

  console.log('Audio file received:', req.file.filename);

  const inputFile = path.join(__dirname, 'uploads', req.file.filename);
  const outputFile = path.join(__dirname, 'uploads', path.parse(req.file.filename).name + '.wav');

  // Log file size
  const stats = fs.statSync(inputFile);
  console.log(`WAV file size: ${stats.size} bytes`);

  // Rename to ensure .wav extension (file is already WAV from browser)
  if (inputFile !== outputFile) {
    fs.renameSync(inputFile, outputFile);
    console.log('WAV file ready:', outputFile);
  } else {
    console.log('WAV file ready:', inputFile);
  }

  res.send('Audio uploaded successfully');
});

// Handle POST request for text messages
app.post('/text-message', (req, res) => {
  const { message, enableTts } = req.body;
  if (!message) {
    return res.status(400).send('No message provided');
  }

  // Use TEXT_TTS: prefix when TTS is explicitly requested, TEXT: otherwise
  const prefix = enableTts ? 'TEXT_TTS:' : 'TEXT:';

  if (pythonReady) {
    // Send the text message to the Python script via stdin
    pythonProcess.stdin.write(`${prefix}${message}\n`);
    res.send('Message sent');
  } else {
    // Queue the message until Python is ready
    console.log('Python not ready, queuing message:', message);
    pendingMessages.push({ message, enableTts });
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