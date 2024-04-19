const express = require('express');
const multer = require('multer');
const path = require('path');
const ffmpeg = require('fluent-ffmpeg');
const fs = require('fs');


const app = express();
const port = 3000; // You can change the port number if needed

const cors = require('cors');
app.use(cors());

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
  const outputFile = path.join(__dirname, 'uploads', path.parse(req.file.originalname).name + '.wav');

  ffmpeg(inputFile)
    .output(outputFile)
    .on('end', () => {
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

// Start the server
app.listen(port, () => {
  console.log(`Server is running on port ${port}`);
});