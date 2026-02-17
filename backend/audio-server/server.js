const express = require('express');
const multer = require('multer');
const path = require('path');
const fs = require('fs');
const http = require('http');
const { WebSocket, WebSocketServer } = require('ws');

const { spawn } = require('child_process');

// Configuration
const STT_PROVIDER = process.env.STT_PROVIDER || 'groq';
const STT_PORT = process.env.STT_PORT || 9001;

// Spawn the Python script
const pythonScript = process.env.JARVIS_SCRIPT || path.join(__dirname, '..', 'JARVIS.py');
const pythonProcess = spawn('python', [pythonScript]);

// Spawn the streaming STT server if using local STT
let sttProcess = null;
if (STT_PROVIDER === 'local') {
  const sttScript = path.join(__dirname, '..', 'stt_streaming.py');
  sttProcess = spawn('python', [sttScript], {
    env: { ...process.env },
    stdio: ['pipe', 'pipe', 'pipe'],
  });
  sttProcess.stdout.on('data', (data) => {
    const output = data.toString().trim();
    if (output === 'STT_READY') {
      console.log('Streaming STT server is ready');
    } else {
      console.log('STT stdout:', output);
    }
  });
  sttProcess.stderr.on('data', (data) => {
    console.error('STT:', data.toString().trimEnd());
  });
  sttProcess.on('exit', (code) => {
    console.error(`STT process exited with code ${code}`);
  });
  console.log(`Streaming STT server started (provider: ${STT_PROVIDER})`);
} else {
  console.log(`STT provider: ${STT_PROVIDER} (no local STT server)`);
}

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
    } else if (trimmed === 'EOU_WAITING') {
      console.log('EOU: waiting for more speech');
      wsBroadcastJSON({ type: 'eou_waiting' });
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

// Handle POST request to update memory mode
app.post('/memory-mode', (req, res) => {
  const { mode } = req.body;
  const validModes = ['full', 'read-only', 'off'];
  if (!validModes.includes(mode)) {
    return res.status(400).json({ error: 'Invalid mode. Must be: full, read-only, or off' });
  }

  if (pythonReady) {
    pythonProcess.stdin.write(`MEMORY_MODE:${mode}\n`);
    res.json({ success: true, mode });
  } else {
    res.status(503).send('Python not ready');
  }
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

  // Per-client STT WebSocket connection to Python STT server
  let sttWs = null;
  let sttPendingChunks = []; // Buffer PCM chunks while STT WS is connecting

  function connectSTT() {
    if (STT_PROVIDER !== 'local') return;
    try {
      sttWs = new WebSocket(`ws://localhost:${STT_PORT}`);
      sttWs.on('open', () => {
        console.log('Connected to STT server for client');
        // Flush any buffered PCM chunks
        for (const chunk of sttPendingChunks) {
          sttWs.send(chunk);
        }
        sttPendingChunks = [];
      });
      sttWs.on('message', (data) => {
        // Relay partial/final transcripts back to browser
        try {
          const msg = JSON.parse(data.toString());
          if (msg.type === 'partial') {
            console.log('[STT Relay] Partial:', msg.text?.substring(0, 60), msg.eou ? '[EOU:done]' : '[EOU:continue]');
            ws.send(JSON.stringify({ type: 'stt_partial', text: msg.text, eou: msg.eou }));
          } else if (msg.type === 'final') {
            ws.send(JSON.stringify({ type: 'stt_final', text: msg.text }));
            // Also send to JARVIS.py as TEXT_TTS: for LLM processing
            if (msg.text && pythonReady) {
              pythonProcess.stdin.write(`TEXT_TTS:${msg.text}\n`);
            }
          } else if (msg.type === 'eou_waiting') {
            console.log('[STT Relay] EOU waiting for more speech');
            ws.send(JSON.stringify({ type: 'eou_waiting' }));
          }
        } catch (e) {
          console.error('Error parsing STT message:', e);
        }
      });
      sttWs.on('error', (err) => {
        console.error('STT WebSocket error:', err.message);
      });
      sttWs.on('close', () => {
        sttWs = null;
      });
    } catch (e) {
      console.error('Failed to connect to STT server:', e);
    }
  }

  ws.on('message', (data) => {
    // Check if binary data (PCM audio for STT)
    if (Buffer.isBuffer(data) || data instanceof ArrayBuffer) {
      if (sttWs && sttWs.readyState === WebSocket.OPEN) {
        sttWs.send(data);
      } else if (sttWs) {
        // Still connecting — buffer chunks
        sttPendingChunks.push(data);
      }
      return;
    }

    try {
      const msg = JSON.parse(data.toString());
      if (msg.type === 'interrupt') {
        console.log('Interrupt requested via WebSocket');
        pythonProcess.stdin.write('INTERRUPT\n');
      } else if (msg.type === 'get_config') {
        ws.send(JSON.stringify({
          type: 'config',
          sttProvider: STT_PROVIDER,
        }));
      } else if (msg.type === 'stt_start') {
        // Client starting to stream audio for STT
        if (!sttWs || sttWs.readyState !== WebSocket.OPEN) {
          connectSTT();
        }
      } else if (msg.type === 'stt_end') {
        // Client finished speaking, tell STT to finalize
        if (sttWs && sttWs.readyState === WebSocket.OPEN) {
          sttWs.send(JSON.stringify({ type: 'finalize' }));
        }
      }
    } catch (e) {
      // Could be binary data that's not valid JSON - that's fine
      if (!(data instanceof Buffer)) {
        console.error('Invalid WebSocket message:', e.message);
      }
    }
  });

  ws.on('close', () => {
    wsClients.delete(ws);
    if (sttWs) {
      sttWs.close();
      sttWs = null;
    }
    console.log(`WebSocket client disconnected, total: ${wsClients.size}`);
  });
});

// Start the server
server.listen(port, () => {
  console.log(`Server is running on port ${port}`);
});
