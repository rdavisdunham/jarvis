const express = require('express');
const multer = require('multer');
const path = require('path');
const fs = require('fs');
const http = require('http');
const { WebSocketServer } = require('ws');

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

// WebSocket client management for streaming audio
const wsClients = new Set();

function broadcastEvent(eventType, data) {
  sseClients.forEach(client => {
    client.res.write(`event: ${eventType}\ndata: ${JSON.stringify(data)}\n\n`);
  });
}

function wsBroadcastJSON(obj) {
  const msg = JSON.stringify(obj);
  for (const ws of wsClients) {
    if (ws.readyState === 1) { // WebSocket.OPEN
      ws.send(msg);
    }
  }
}

function wsBroadcastBinary(buffer) {
  for (const ws of wsClients) {
    if (ws.readyState === 1) {
      ws.send(buffer);
    }
  }
}

// Listen for data from the Python script
pythonProcess.stdout.on('data', (data) => {
  const output = data.toString();

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
    } else if (trimmed === 'KOKORO_READY') {
      console.log('Kokoro model loaded and ready');
      wsBroadcastJSON({ type: 'kokoro_ready' });
    } else if (trimmed === 'INTERRUPT_ACK') {
      console.log('Interrupt acknowledged by Python');
      wsBroadcastJSON({ type: 'interrupt_ack' });
    } else if (trimmed.startsWith('STREAM_START:')) {
      // STREAM_START:<responseId>:<sampleRate>
      const parts = trimmed.substring(13).split(':');
      const responseId = parts[0];
      const sampleRate = parseInt(parts[1]) || 24000;
      console.log(`Stream start: ${responseId} @ ${sampleRate}Hz`);
      wsBroadcastJSON({
        type: 'audio_start',
        responseId,
        sampleRate,
        channels: 1,
        bitDepth: 16
      });
    } else if (trimmed.startsWith('STREAM_CHUNK:')) {
      // STREAM_CHUNK:<responseId>:<chunkIndex>:<base64_pcm_data>
      const firstColon = 13; // after "STREAM_CHUNK:"
      const secondColon = trimmed.indexOf(':', firstColon);
      const thirdColon = trimmed.indexOf(':', secondColon + 1);
      const chunkIndex = parseInt(trimmed.substring(secondColon + 1, thirdColon));
      const b64Data = trimmed.substring(thirdColon + 1);

      // Decode base64 to binary PCM
      const pcmBuffer = Buffer.from(b64Data, 'base64');

      // Build binary message: [4-byte chunkIndex LE][PCM data]
      const header = Buffer.alloc(4);
      header.writeUInt32LE(chunkIndex, 0);
      const binaryMsg = Buffer.concat([header, pcmBuffer]);
      wsBroadcastBinary(binaryMsg);
    } else if (trimmed.startsWith('STREAM_END:')) {
      // STREAM_END:<responseId>
      const responseId = trimmed.substring(11);
      console.log(`Stream end: ${responseId}`);
      wsBroadcastJSON({ type: 'audio_end', responseId });
    } else if (trimmed.startsWith('Audio:')) {
      // TTS audio file notification (Groq path) - push via SSE for real-time delivery
      const audioFile = trimmed.substring(6).trim();
      broadcastEvent('audio', { file: audioFile });
      console.log('TTS audio pushed via SSE:', audioFile);
    } else if (trimmed) {
      // Don't log STREAM_ lines as generic output
      console.log('Python script output:', trimmed);
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

// Create HTTP server and attach WebSocket
const server = http.createServer(app);

const wss = new WebSocketServer({ server, path: '/ws' });

wss.on('connection', (ws) => {
  wsClients.add(ws);
  console.log(`WebSocket client connected, total: ${wsClients.size}`);

  ws.on('message', (data) => {
    try {
      const msg = JSON.parse(data.toString());
      if (msg.type === 'interrupt') {
        console.log('Interrupt requested via WebSocket');
        pythonProcess.stdin.write('INTERRUPT\n');
      }
    } catch (e) {
      console.error('Invalid WebSocket message:', e.message);
    }
  });

  ws.on('close', () => {
    wsClients.delete(ws);
    console.log(`WebSocket client disconnected, total: ${wsClients.size}`);
  });
});

// Start the server
server.listen(port, () => {
  console.log(`Server is running on port ${port}`);
});
