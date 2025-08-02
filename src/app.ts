import express from 'express';
import cors from 'cors';
import multer from 'multer';
import path from 'path';
import fs from 'fs';
import { InferenceSession, Tensor } from 'onnxruntime-node';
import sharp from 'sharp';

const app = express();
const PORT = 5000;

// Configuration
import os from 'os';

const UPLOAD_FOLDER = path.join(process.env.HOME || '/tmp', '.dejpeg', 'models');
const ALLOWED_EXTENSIONS = new Set(['.onnx']);
const MAX_MODEL_SIZE = 2 * 1024 * 1024 * 1024; // 2GB
const MAX_IMAGE_SIZE = 500 * 1024 * 1024; // 500MB

// Ensure upload folder exists
if (!fs.existsSync(UPLOAD_FOLDER)) {
  fs.mkdirSync(UPLOAD_FOLDER, { recursive: true });
}

// Middleware
app.use(cors());
const staticPath = path.join(__dirname, 'static');
app.use('/static', express.static(staticPath));


// Multer configuration for file uploads
const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    cb(null, UPLOAD_FOLDER);
  },
  filename: (req, file, cb) => {
    cb(null, file.originalname);
  }
});

const fileFilter = (req: any, file: Express.Multer.File, cb: any) => {
  const extname = path.extname(file.originalname).toLowerCase();
  if (ALLOWED_EXTENSIONS.has(extname)) {
    cb(null, true);
  } else {
    cb(new Error('Invalid file type. Only .onnx files are allowed'), false);
  }
};

const upload = multer({
  storage,
  fileFilter,
  limits: {
    fileSize: MAX_MODEL_SIZE
  }
});

const imageUpload = multer({
  storage: multer.memoryStorage(),
  limits: {
    fileSize: MAX_IMAGE_SIZE
  }
});

// Image Processor Class
class ImageProcessor {
  private session: InferenceSession | null = null;
  private modelName: string | null = null;
  private inputName: string | null = null;
  private outputName: string | null = null;
  private processing = false;

  async loadModel(modelPath: string): Promise<boolean> {
    try {
      if (this.session) {
        this.session = null;
      }

      if (!fs.existsSync(modelPath)) {
        throw new Error(`Model file not found: ${modelPath}`);
      }

      console.log(`Loading model: ${modelPath}`);
      this.session = await InferenceSession.create(modelPath);

      this.inputName = this.session.inputNames[0];
      this.outputName = this.session.outputNames[0];
      this.modelName = path.basename(modelPath, '.onnx');

      console.log(`Model loaded successfully: ${this.modelName}`);
      return true;
    } catch (error) {
      console.error(`Failed to load model: ${error}`);
      this.session = null;
      this.modelName = null;
      throw error;
    }
  }

  async processImage(imageBuffer: Buffer, strength: number = 0.5): Promise<Buffer> {
    if (!this.session) {
      throw new Error('No model loaded. Please load a model first.');
    }

    if (this.processing) {
      throw new Error('Another image is currently being processed');
    }

    this.processing = true;

    try {
      // Convert image to RGB and normalize
      const { data, info } = await sharp(imageBuffer)
        .removeAlpha()
        .raw()
        .toBuffer({ resolveWithObject: true });

      const [width, height, channels] = [info.width, info.height, info.channels];

      // reorder data from HWC to CHW
      const imgArray = new Float32Array(channels * height * width);
      for (let c = 0; c < channels; c++) {
        for (let h = 0; h < height; h++) {
          for (let w = 0; w < width; w++) {
            imgArray[c * height * width + h * width + w] = data[h * width * channels + w * channels + c] / 255.0;
          }
        }
      }

      // Prepare input tensor
      const inputTensor = new Tensor('float32', imgArray, [
        1, 
        info.channels, 
        info.height, 
        info.width
      ]);

      // Run inference
      const feeds: Record<string, Tensor> = {};
      feeds[this.inputName!] = inputTensor;

      // Add quality factor if model expects it
      if (this.session.inputNames.includes('qf')) {
        feeds['qf'] = new Tensor('float32', new Float32Array([strength]), [1, 1]);
      }

      const output = await this.session.run(feeds);
      const outputTensor = output[this.outputName!];
      const [n, c, h, w] = outputTensor.dims;
      const outputData = outputTensor.data as Float32Array;

      // create buffer for HWC order
      const outputBuffer = Buffer.alloc(h * w * c);
      for (let cIdx = 0; cIdx < c; cIdx++) {
        for (let hIdx = 0; hIdx < h; hIdx++) {
          for (let wIdx = 0; wIdx < w; wIdx++) {
            outputBuffer[hIdx * w * c + wIdx * c + cIdx] = Math.min(Math.max(outputData[cIdx * h * w + hIdx * w + wIdx] * 255, 0), 255);
          }
        }
      }

      return await sharp(outputBuffer, {
        raw: {
          width: w,
          height: h,
          channels: c as 1 | 2 | 3 | 4,
        }
      }).png().toBuffer();

    } catch (error) {
      console.error(`Image processing failed: ${error}`);
      throw error;
    } finally {
      this.processing = false;
    }
  }
}

const processor = new ImageProcessor();

// Routes
app.get('/', (req, res) => {
  const indexPath = path.join(__dirname, 'static', 'web', 'index.html');
  res.sendFile(indexPath);
});

app.get('/api/health', (req, res) => {
  res.json({
    status: 'healthy',
    model_loaded: processor['session'] !== null,
    model_name: processor['modelName']
  });
});

app.post('/api/load_model', upload.single('file'), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ error: 'No file provided' });
    }

    const modelPath = path.join(UPLOAD_FOLDER, req.file.filename);
    await processor.loadModel(modelPath);

    res.json({
      success: true,
      model: processor['modelName'],
      message: `Model '${processor['modelName']}' loaded successfully`
    });
  } catch (error) {
    console.error(`Model loading error: ${error}`);
    res.status(500).json({ error: `Failed to load model: ${error instanceof Error ? error.message : 'Unknown error'}` });
  }
});

app.post('/api/process', imageUpload.single('file'), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ error: 'No image file provided' });
    }

    if (!processor['session']) {
      return res.status(400).json({ error: 'No model loaded' });
    }

    const strength = parseFloat(req.body.strength || '0.5');
    const processedImage = await processor.processImage(req.file.buffer, strength);

    // Generate output filename
    const originalName = req.file.originalname;
    const ext = path.extname(originalName);
    const baseName = path.basename(originalName, ext);
    const outputName = `${baseName}_${processor['modelName']}.png`;

    res.set('Content-Type', 'image/png');
    res.set('Content-Disposition', `attachment; filename="${outputName}"`);
    res.send(processedImage);
  } catch (error) {
    console.error(`Image processing error: ${error}`);
    res.status(500).json({ error: `Failed to process image: ${error instanceof Error ? error.message : 'Unknown error'}` });
  }
});

// Error handling middleware
app.use((err: any, req: any, res: any, next: any) => {
  console.error(err.stack);
  res.status(500).json({ error: 'Internal server error' });
});

// Start server
app.listen(PORT, () => {
  console.log(`DeJPEG server running on port ${PORT}`);
});