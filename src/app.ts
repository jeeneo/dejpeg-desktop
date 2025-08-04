import express from 'express';
import cors from 'cors';
import multer from 'multer';
import path from 'path';
import fs from 'fs';
import { InferenceSession, Tensor } from 'onnxruntime-node';
import sharp from 'sharp';
import { randomUUID } from 'crypto';
// import { extractNodeModules, cleanupNodeModules } from './extract';

const app = express();
const PORT = 5567;

// Configuration
import os from 'os';

const UPLOAD_FOLDER = path.join(process.env.HOME || '/tmp', '.dejpeg', 'models');
const TEMP_FOLDER = path.join(process.env.HOME || '/tmp', '.dejpeg', 'temp');
const ALLOWED_EXTENSIONS = new Set(['.onnx']);
const MAX_MODEL_SIZE = 2 * 1024 * 1024 * 1024; // 2GB
const MAX_IMAGE_SIZE = 500 * 1024 * 1024; // 500MB

// Chunking configuration
const DEFAULT_CHUNK_SIZE = 1200;
const SCUNET_CHUNK_SIZE = 640;
const DEFAULT_OVERLAP = 32;
const SCUNET_OVERLAP = 128;

// Ensure upload folder exists
if (!fs.existsSync(UPLOAD_FOLDER)) {
  fs.mkdirSync(UPLOAD_FOLDER, { recursive: true });
}

if (!fs.existsSync(TEMP_FOLDER)) {
  fs.mkdirSync(TEMP_FOLDER, { recursive: true });
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

// Interfaces
interface ChunkInfo {
  x: number;
  y: number;
  width: number;
  height: number;
  chunkFile: string;
  processedFile?: string;
  originalStartX: number;
  originalStartY: number;
  row: number;
  col: number;
}

interface ModelInfo {
  imageInputName: string;
  imageInputShape: number[];
  isGrayscale: boolean;
  hasQualityInput: boolean;
  qualityInputName?: string;
}

interface ProcessingState {
  totalChunks: number;
  completedChunks: number;
  currentImage: number;
  totalImages: number;
}

// Utility functions
function clearTempDir(dir: string): void {
  if (fs.existsSync(dir)) {
    const files = fs.readdirSync(dir);
    for (const file of files) {
      const filePath = path.join(dir, file);
      fs.unlinkSync(filePath);
    }
  }
}

function getChunkSizeForModel(modelName: string): number {
  return modelName && modelName.startsWith('scunet_') ? SCUNET_CHUNK_SIZE : DEFAULT_CHUNK_SIZE;
}

function getOverlapForModel(modelName: string): number {
  return modelName && modelName.startsWith('scunet_') ? SCUNET_OVERLAP : DEFAULT_OVERLAP;
}

async function saveImageToFile(imageData: Buffer, filePath: string): Promise<void> {
  await sharp(imageData).png().toFile(filePath);
}

async function loadImageFromFile(filePath: string): Promise<{ data: Buffer; width: number; height: number; channels: number }> {
  const { data, info } = await sharp(filePath)
    .removeAlpha()
    .raw()
    .toBuffer({ resolveWithObject: true });
  
  return {
    data,
    width: info.width,
    height: info.height,
    channels: info.channels
  };
}

// Image Processor Class
class ImageProcessor {
  private session: InferenceSession | null = null;
  private modelName: string | null = null;
  private modelInfo: ModelInfo | null = null;
  private processing = false;
  private processingState: ProcessingState = {
    totalChunks: 0,
    completedChunks: 0,
    currentImage: 1,
    totalImages: 1
  };

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
      this.modelName = path.basename(modelPath, '.onnx');

      console.log(`Input names: ${this.session.inputNames.join(', ')}`);
      console.log(`Output names: ${this.session.outputNames.join(', ')}`);

      // Analyze model inputs dynamically
      this.modelInfo = await this.analyzeModelInputs();

      console.log(`Model loaded successfully: ${this.modelName}`);
      console.log(`Model info:`, this.modelInfo);
      return true;
    } catch (error) {
      console.error(`Failed to load model: ${error}`);
      this.session = null;
      this.modelName = null;
      this.modelInfo = null;
      throw error;
    }
  }

  private async analyzeModelInputs(): Promise<ModelInfo> {
    if (!this.session) {
      throw new Error('No session available');
    }

    let imageInputName: string | null = null;
    let imageInputShape: number[] = [];
    let isGrayscale = false;
    let hasQualityInput = false;
    let qualityInputName: string | undefined;

    console.log('Available input names:', this.session.inputNames);

    // Try to identify the image input by name patterns
    const imageInputCandidates = this.session.inputNames.filter(name => 
      name.toLowerCase().includes('input') || 
      name.toLowerCase().includes('image') || 
      name.toLowerCase().includes('x') ||
      name === 'input' ||
      (this.session && this.session.inputNames.length === 1) // If only one input, assume it's the image
    );

    if (imageInputCandidates.length === 0) {
      // Fallback: use the first input
      imageInputCandidates.push(this.session.inputNames[0]);
    }

    // Test with small dummy tensors to determine expected shape
    for (const candidateName of imageInputCandidates) {
      try {
        // Test with grayscale (1 channel)
        const testGrayscale = new Tensor('float32', new Float32Array(64 * 64), [1, 1, 64, 64]);
        const testFeeds: Record<string, Tensor> = {};
        testFeeds[candidateName] = testGrayscale;
        
        try {
          await this.session.run(testFeeds);
          // If successful, this is likely a grayscale model
          imageInputName = candidateName;
          imageInputShape = [1, 1, 64, 64]; // We'll update height/width dynamically
          isGrayscale = true;
          console.log(`Detected grayscale image input: ${candidateName}`);
          break;
        } catch (e1) {
          // Try color (3 channels)
          try {
            const testColor = new Tensor('float32', new Float32Array(3 * 64 * 64), [1, 3, 64, 64]);
            testFeeds[candidateName] = testColor;
            await this.session.run(testFeeds);
            // If successful, this is likely a color model
            imageInputName = candidateName;
            imageInputShape = [1, 3, 64, 64]; // We'll update height/width dynamically
            isGrayscale = false;
            console.log(`Detected color image input: ${candidateName}`);
            break;
          } catch (e2) {
            if (e2 && typeof e2 === 'object' && 'message' in e2) {
              console.log(`Failed to detect tensor format for ${candidateName}:`, (e2 as any).message);
            } else {
              console.log(`Failed to detect tensor format for ${candidateName}:`, e2);
            }
            continue;
          }
        }
      } catch (e) {
        if (e && typeof e === 'object' && 'message' in e) {
          console.log(`Error testing input ${candidateName}:`, (e as any).message);
        } else {
          console.log(`Error testing input ${candidateName}:`, e);
        }
        continue;
      }
    }

    if (!imageInputName) {
      // Final fallback - assume first input is image with color
      imageInputName = this.session.inputNames[0];
      imageInputShape = [1, 3, -1, -1]; // Dynamic height/width
      isGrayscale = false;
      console.log(`Using fallback: assuming ${imageInputName} is color image input`);
    }

    // Check for quality/strength input
    for (const inputName of this.session.inputNames) {
      if (inputName !== imageInputName) {
        // Common names for quality inputs
        if (inputName.toLowerCase().includes('qf') || 
            inputName.toLowerCase().includes('quality') ||
            inputName.toLowerCase().includes('strength')) {
          hasQualityInput = true;
          qualityInputName = inputName;
          console.log(`Detected quality input: ${inputName}`);
          break;
        }
      }
    }

    // If we have exactly 2 inputs and haven't found quality input, assume second is quality
    if (!hasQualityInput && this.session.inputNames.length === 2) {
      qualityInputName = this.session.inputNames.find(name => name !== imageInputName);
      if (qualityInputName) {
        hasQualityInput = true;
        console.log(`Assuming second input is quality: ${qualityInputName}`);
      }
    }

    return {
      imageInputName,
      imageInputShape,
      isGrayscale,
      hasQualityInput,
      qualityInputName
    };
  }

  async processImage(imageBuffer: Buffer, strength: number = 0.5, onProgress?: (state: ProcessingState) => void): Promise<Buffer> {
    if (!this.session || !this.modelInfo) {
      throw new Error('No model loaded. Please load a model first.');
    }

    if (this.processing) {
      throw new Error('Another image is currently being processed');
    }

    this.processing = true;
    this.processingState = {
      totalChunks: 0,
      completedChunks: 0,
      currentImage: 1,
      totalImages: 1
    };

    try {
      // Get image info
      const { info } = await sharp(imageBuffer).toBuffer({ resolveWithObject: true });
      const { width, height } = info;

      console.log(`Processing image: ${width}x${height}`);

      const effectiveChunkSize = getChunkSizeForModel(this.modelName!);
      const overlap = getOverlapForModel(this.modelName!);

      // Check if we need to chunk the image
      if (width > effectiveChunkSize || height > effectiveChunkSize) {
        return await this.processImageWithChunking(imageBuffer, strength, onProgress);
      } else {
        this.processingState.totalChunks = 1;
        if (onProgress) onProgress(this.processingState);

        const result = await this.processChunk(imageBuffer, strength);
        this.processingState.completedChunks = 1;
        if (onProgress) onProgress(this.processingState);

        return result;
      }
    } catch (error) {
      console.error(`Image processing failed: ${error}`);
      throw error;
    } finally {
      this.processing = false;
    }
  }

  private async processImageWithChunking(imageBuffer: Buffer, strength: number, onProgress?: (state: ProcessingState) => void): Promise<Buffer> {
    const sessionId = randomUUID();
    const chunkDir = path.join(TEMP_FOLDER, `chunks_${sessionId}`);
    const processedDir = path.join(TEMP_FOLDER, `processed_${sessionId}`);

    try {
      // Create temp directories
      if (!fs.existsSync(chunkDir)) fs.mkdirSync(chunkDir, { recursive: true });
      if (!fs.existsSync(processedDir)) fs.mkdirSync(processedDir, { recursive: true });

      // Get image dimensions
      const { info } = await sharp(imageBuffer).toBuffer({ resolveWithObject: true });
      const { width, height } = info;

      // Create chunks
      console.log('Creating chunks...');
      const chunks = await this.createChunks(imageBuffer, chunkDir);
      
      this.processingState.totalChunks = chunks.length;
      this.processingState.completedChunks = 0;
      if (onProgress) onProgress(this.processingState);

      console.log(`Processing ${chunks.length} chunks...`);

      // Process each chunk
      for (let i = 0; i < chunks.length; i++) {
        const chunk = chunks[i];
        console.log(`Processing chunk ${i + 1}/${chunks.length} at (${chunk.x}, ${chunk.y})`);

        // Load chunk
        const chunkImageData = await loadImageFromFile(chunk.chunkFile);
        const chunkBuffer = await sharp(chunkImageData.data, {
          raw: {
            width: chunkImageData.width,
            height: chunkImageData.height,
            channels: chunkImageData.channels as 1 | 2 | 3 | 4
          }
        }).png().toBuffer();

        // Process chunk
        const processedChunk = await this.processChunk(chunkBuffer, strength);

        // Save processed chunk
        chunk.processedFile = path.join(processedDir, `processed_${chunk.x}_${chunk.y}.png`);
        await saveImageToFile(processedChunk, chunk.processedFile);

        this.processingState.completedChunks++;
        if (onProgress) onProgress(this.processingState);
      }

      console.log('Reassembling chunks...');
      // Reassemble chunks with feathering
      const result = await this.reassembleChunksWithFeathering(chunks, width, height);

      return result;
    } finally {
      // Clean up temp directories
      try {
        if (fs.existsSync(chunkDir)) {
          clearTempDir(chunkDir);
          fs.rmdirSync(chunkDir);
        }
        if (fs.existsSync(processedDir)) {
          clearTempDir(processedDir);
          fs.rmdirSync(processedDir);
        }
      } catch (cleanupError) {
        console.warn('Error cleaning up temp directories:', cleanupError);
      }
    }
  }

  private async createChunks(imageBuffer: Buffer, chunkDir: string): Promise<ChunkInfo[]> {
    const chunks: ChunkInfo[] = [];
    
    const { info } = await sharp(imageBuffer).toBuffer({ resolveWithObject: true });
    const { width, height } = info;

    const effectiveChunkSize = getChunkSizeForModel(this.modelName!);
    const overlap = getOverlapForModel(this.modelName!);

    // Calculate grid dimensions
    const cols = Math.ceil(width / (effectiveChunkSize - overlap));
    const rows = Math.ceil(height / (effectiveChunkSize - overlap));

    for (let row = 0; row < rows; row++) {
      for (let col = 0; col < cols; col++) {
        // Calculate chunk position with symmetric overlap
        const startX = col * (effectiveChunkSize - overlap);
        const startY = row * (effectiveChunkSize - overlap);

        // Extend into overlap regions
        const chunkX = Math.max(0, startX - (col > 0 ? Math.floor(overlap / 2) : 0));
        const chunkY = Math.max(0, startY - (row > 0 ? Math.floor(overlap / 2) : 0));

        // Calculate actual chunk dimensions
        let chunkWidth = Math.min(effectiveChunkSize + (col > 0 ? Math.floor(overlap / 2) : 0), width - chunkX);
        let chunkHeight = Math.min(effectiveChunkSize + (row > 0 ? Math.floor(overlap / 2) : 0), height - chunkY);

        // Ensure we don't exceed image boundaries
        chunkWidth = Math.min(chunkWidth, width - chunkX);
        chunkHeight = Math.min(chunkHeight, height - chunkY);

        if (chunkWidth <= 0 || chunkHeight <= 0) continue;

        // Extract chunk
        const chunkBuffer = await sharp(imageBuffer)
          .extract({ left: chunkX, top: chunkY, width: chunkWidth, height: chunkHeight })
          .png()
          .toBuffer();

        const chunkFile = path.join(chunkDir, `chunk_${chunkX}_${chunkY}.png`);
        await saveImageToFile(chunkBuffer, chunkFile);

        chunks.push({
          x: chunkX,
          y: chunkY,
          width: chunkWidth,
          height: chunkHeight,
          chunkFile,
          originalStartX: startX,
          originalStartY: startY,
          row,
          col
        });
      }
    }

    return chunks;
  }

  private async processChunk(imageBuffer: Buffer, strength: number): Promise<Buffer> {
    if (!this.session || !this.modelInfo) {
      throw new Error('No model loaded');
    }

    // Convert image to the format expected by the model
    const { data, info } = await sharp(imageBuffer)
      .removeAlpha()
      .raw()
      .toBuffer({ resolveWithObject: true });

    const { width, height, channels } = info;

    // Create input array in CHW format
    let inputArray: Float32Array;
    
    if (this.modelInfo.isGrayscale) {
      inputArray = new Float32Array(height * width);
      for (let h = 0; h < height; h++) {
        for (let w = 0; w < width; w++) {
          const idx = h * width + w;
          const pixelIdx = idx * channels;
          // Convert to grayscale
          const gray = channels === 1 ? data[pixelIdx] : 
                      (data[pixelIdx] + data[pixelIdx + 1] + data[pixelIdx + 2]) / 3;
          inputArray[idx] = gray / 255.0;
        }
      }
    } else {
      inputArray = new Float32Array(3 * height * width);
      const actualChannels = Math.min(channels, 3);
      
      for (let c = 0; c < actualChannels; c++) {
        for (let h = 0; h < height; h++) {
          for (let w = 0; w < width; w++) {
            const inputIdx = c * height * width + h * width + w;
            const pixelIdx = (h * width + w) * channels + c;
            inputArray[inputIdx] = data[pixelIdx] / 255.0;
          }
        }
      }
      
      // Fill remaining channels if needed
      if (actualChannels < 3) {
        for (let c = actualChannels; c < 3; c++) {
          for (let h = 0; h < height; h++) {
            for (let w = 0; w < width; w++) {
              const inputIdx = c * height * width + h * width + w;
              const lastChannelIdx = (actualChannels - 1) * height * width + h * width + w;
              inputArray[inputIdx] = inputArray[lastChannelIdx];
            }
          }
        }
      }
    }

    // Create input tensor with dynamic shape
    const actualInputShape = [1, this.modelInfo.isGrayscale ? 1 : 3, height, width];
    const inputTensor = new Tensor('float32', inputArray, actualInputShape);

    // Prepare feeds
    const feeds: Record<string, Tensor> = {};
    feeds[this.modelInfo.imageInputName] = inputTensor;

    // Add quality input if needed
    if (this.modelInfo.hasQualityInput && this.modelInfo.qualityInputName) {
      feeds[this.modelInfo.qualityInputName] = new Tensor('float32', new Float32Array([strength]), [1, 1]);
    }

    // Run inference
    const output = await this.session.run(feeds);
    const outputTensor = output[this.session.outputNames[0]];
    const outputData = outputTensor.data as Float32Array;
    const [, outputChannels, outputHeight, outputWidth] = outputTensor.dims;

    // Convert output back to image format (HWC)
    const outputBuffer = Buffer.alloc(outputHeight * outputWidth * outputChannels);
    
    for (let c = 0; c < outputChannels; c++) {
      for (let h = 0; h < outputHeight; h++) {
        for (let w = 0; w < outputWidth; w++) {
          const inputIdx = c * outputHeight * outputWidth + h * outputWidth + w;
          const outputIdx = h * outputWidth * outputChannels + w * outputChannels + c;
          const value = Math.min(Math.max(outputData[inputIdx] * 255, 0), 255);
          outputBuffer[outputIdx] = Math.floor(value);
        }
      }
    }

    // Convert back to PNG
    return await sharp(outputBuffer, {
      raw: {
        width: outputWidth,
        height: outputHeight,
        channels: outputChannels as 1 | 2 | 3 | 4,
      }
    }).png().toBuffer();
  }

  private async reassembleChunksWithFeathering(chunks: ChunkInfo[], totalWidth: number, totalHeight: number): Promise<Buffer> {
    // Create base image
    let result = sharp({
      create: {
        width: totalWidth,
        height: totalHeight,
        channels: 3,
        background: { r: 0, g: 0, b: 0 }
      }
    });

    const overlap = getOverlapForModel(this.modelName!);
    const featherSize = Math.floor(overlap / 2);

    // Process chunks in order
    const compositeImages: any[] = [];

    for (const chunk of chunks) {
      if (!chunk.processedFile) continue;

      // Load processed chunk
      const processedImage = await sharp(chunk.processedFile).png().toBuffer();
      
      // Create feathered version
      const featheredChunk = await this.createFeatheredChunk(
        processedImage, 
        chunk, 
        totalWidth, 
        totalHeight, 
        featherSize
      );

      compositeImages.push({
        input: featheredChunk,
        left: chunk.x,
        top: chunk.y,
        blend: 'over'
      });
    }

    // Composite all chunks
    result = result.composite(compositeImages);

    return await result.png().toBuffer();
  }

  private async createFeatheredChunk(
    chunkBuffer: Buffer, 
    chunkInfo: ChunkInfo, 
    totalWidth: number, 
    totalHeight: number, 
    featherSize: number
  ): Promise<Buffer> {
    const { data, info } = await sharp(chunkBuffer)
      .ensureAlpha()
      .raw()
      .toBuffer({ resolveWithObject: true });

    const { width, height, channels } = info;
    const modifiedData = Buffer.from(data);

    // Apply feathering to alpha channel
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const idx = (y * width + x) * channels;
        let alpha = 1.0;

        // Left edge feathering (except for leftmost chunks)
        if (chunkInfo.x > 0 && x < featherSize) {
          alpha = Math.min(alpha, x / featherSize);
        }

        // Top edge feathering (except for topmost chunks)
        if (chunkInfo.y > 0 && y < featherSize) {
          alpha = Math.min(alpha, y / featherSize);
        }

        // Right edge feathering (except for rightmost chunks)
        if (chunkInfo.x + width < totalWidth && x >= width - featherSize) {
          alpha = Math.min(alpha, (width - x) / featherSize);
        }

        // Bottom edge feathering (except for bottommost chunks)
        if (chunkInfo.y + height < totalHeight && y >= height - featherSize) {
          alpha = Math.min(alpha, (height - y) / featherSize);
        }

        // Apply alpha
        modifiedData[idx + 3] = Math.floor(alpha * 255);
      }
    }

    return await sharp(modifiedData, {
      raw: {
        width,
        height,
        channels: channels as 1 | 2 | 3 | 4,
      }
    }).png().toBuffer();
  }

  getProcessingState(): ProcessingState {
    return { ...this.processingState };
  }

  isProcessing(): boolean {
    return this.processing;
  }
}

const processor = new ImageProcessor();
// extractNodeModules();

// was a linux thing but now i guess 'exit' for everyone
// const IS_LINUX = process.platform === 'linux';

// Routes
app.get('/', (req, res) => {
  const indexPath = path.join(__dirname, 'static', 'web', 'index.html');
  let html = fs.readFileSync(indexPath, 'utf8');
  // Inject window.isLinux
  // html = html.replace(
  //   '</head>',
  //   `<script>window.isLinux=${IS_LINUX};</script>\n</head>`
  // );
  res.send(html);
});

app.get('/api/health', (req, res) => {
  res.json({
    status: 'healthy',
    model_loaded: processor['session'] !== null,
    model_name: processor['modelName'],
    processing: processor.isProcessing(),
    processing_state: processor.getProcessingState()
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
      message: `model '${processor['modelName']}' loaded`,
      model_info: processor['modelInfo']
    });
  } catch (error) {
    console.error(`model loading error: ${error}`);
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
    
    // Set up SSE for progress updates if requested
    const useSSE = req.body.progress === 'true';
    
    if (useSSE) {
      res.writeHead(200, {
        'Content-Type': 'text/event-stream',
        'Cache-Control': 'no-cache',
        'Connection': 'keep-alive',
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Headers': 'Cache-Control'
      });

      const processedImage = await processor.processImage(req.file.buffer, strength, (state) => {
        res.write(`data: ${JSON.stringify({ type: 'progress', state })}\n\n`);
      });

      // Send final result as base64
      const base64Image = processedImage.toString('base64');
      const originalName = req.file!.originalname;
      const ext = path.extname(originalName);
      const baseName = path.basename(originalName, ext);
      const outputName = `${baseName}_${processor['modelName']}.png`;

      res.write(`data: ${JSON.stringify({ 
        type: 'complete', 
        image: base64Image,
        filename: outputName 
      })}\n\n`);
      res.end();
    } else {
      const processedImage = await processor.processImage(req.file.buffer, strength);

      // Generate output filename
      const originalName = req.file.originalname;
      const ext = path.extname(originalName);
      const baseName = path.basename(originalName, ext);
      const outputName = `${baseName}_${processor['modelName']}.png`;

      res.set('Content-Type', 'image/png');
      res.set('Content-Disposition', `attachment; filename="${outputName}"`);
      res.send(processedImage);
    }
  } catch (error) {
    console.error(`Image processing error: ${error}`);
    if (req.body.progress === 'true') {
      res.write(`data: ${JSON.stringify({ 
        type: 'error', 
        error: `Failed to process image: ${error instanceof Error ? error.message : 'Unknown error'}` 
      })}\n\n`);
      res.end();
    } else {
      res.status(500).json({ error: `Failed to process image: ${error instanceof Error ? error.message : 'Unknown error'}` });
    }
  }
});

app.get('/api/status', (req, res) => {
  res.json({
    processing: processor.isProcessing(),
    state: processor.getProcessingState()
  });
});

app.post('/api/exit', (req, res) => {
  res.json({ status: 'exiting' });
  // cleanupNodeModules();
  setTimeout(() => process.exit(0), 100);
});

// Error handling middleware
app.use((err: any, req: any, res: any, next: any) => {
  console.error(err.stack);
  res.status(500).json({ error: 'Internal server error' });
});

app.listen(PORT, '0.0.0.0', () => {
  console.log(`DeJPEG running at http://0.0.0.0:${PORT}`);

  const url = `http://localhost:${PORT}`;
  const { exec } = require('child_process');

  if (process.platform === 'linux') {
    exec(`xdg-open ${url}`);
  } else if (process.platform === 'win32') {
    exec(`start ${url}`);
  }
  // else if (process.platform === 'darwin') { // macOS untested - uncomment if needed
  //   exec(`open ${url}`);
  // }
});