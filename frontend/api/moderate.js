// Vercel serverless function: scores a batch of video frames with the
// int8 ONNX export of the VS Loss full-finetune checkpoint (IEEE SPL 2025).
//
// POST /api/moderate
//   { "frames": [{ "timestamp": 1.5, "jpegBase64": "..." }, ...] }
// →
//   { "threshold": 0.0603,
//     "results": [{ "timestamp": 1.5, "score": 0.91, "nsfw": true }, ...],
//     "summary": { "frames": 20, "flagged": 2, "maxScore": 0.91, "nsfw": true } }
//
// Preprocessing must mirror training (dataset_loader.NSFWDataset, no augment):
// squash-resize to 224x224, ImageNet mean/std — NOT Falconsai's 0.5/0.5.

const path = require('path');
const sharp = require('sharp');
const ort = require('onnxruntime-node');

// Operating threshold for the int8 VS Loss model, from its own val sweep
// (training/checkpoints/varstab_int8_probs.json). Max-F1 sits at 0.0026, but
// the score distribution is bimodal and flat from 0.005–0.1 (P .857/R .814
// either way), so 0.05 gives the same accuracy with ~25x margin against
// out-of-distribution safe frames (solid colors score up to ~2e-3, which
// made the 0.0026 threshold flag harmless synthetic videos).
const THRESHOLD = 0.05;

const SIZE = 224;
const MEAN = [0.485, 0.456, 0.406];
const STD = [0.229, 0.224, 0.225];
const MAX_FRAMES_PER_REQUEST = 32;

const MODEL_PATH = path.join(process.cwd(), 'api', '_model', 'model_varstab_full_int8.onnx');

let sessionPromise = null;
function getSession() {
  if (!sessionPromise) {
    sessionPromise = ort.InferenceSession.create(MODEL_PATH, {
      graphOptimizationLevel: 'all',
    });
  }
  return sessionPromise;
}

async function jpegToCHW(jpegBuffer) {
  const { data } = await sharp(jpegBuffer)
    .resize(SIZE, SIZE, { fit: 'fill' }) // squash, matching transforms.Resize((224,224))
    .removeAlpha()
    .raw()
    .toBuffer({ resolveWithObject: true });

  const plane = SIZE * SIZE;
  const chw = new Float32Array(3 * plane);
  for (let i = 0; i < plane; i++) {
    for (let c = 0; c < 3; c++) {
      chw[c * plane + i] = (data[i * 3 + c] / 255 - MEAN[c]) / STD[c];
    }
  }
  return chw;
}

async function scoreFrames(jpegBuffers) {
  const session = await getSession();
  const planes = await Promise.all(jpegBuffers.map(jpegToCHW));

  const n = planes.length;
  const batch = new Float32Array(n * 3 * SIZE * SIZE);
  planes.forEach((p, i) => batch.set(p, i * 3 * SIZE * SIZE));

  const input = new ort.Tensor('float32', batch, [n, 3, SIZE, SIZE]);
  const output = await session.run({ pixel_values: input });
  const logits = output.logits.data;

  return Array.from({ length: n }, (_, i) => 1 / (1 + Math.exp(-logits[i])));
}

module.exports = async (req, res) => {
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'POST only' });
  }

  const frames = req.body && req.body.frames;
  if (!Array.isArray(frames) || frames.length === 0) {
    return res.status(400).json({ error: 'body must be { frames: [{ timestamp, jpegBase64 }] }' });
  }
  if (frames.length > MAX_FRAMES_PER_REQUEST) {
    return res.status(400).json({ error: `max ${MAX_FRAMES_PER_REQUEST} frames per request` });
  }

  let buffers;
  try {
    buffers = frames.map((f) => Buffer.from(f.jpegBase64, 'base64'));
  } catch {
    return res.status(400).json({ error: 'invalid base64 in jpegBase64' });
  }

  try {
    const scores = await scoreFrames(buffers);
    const results = scores.map((score, i) => ({
      timestamp: frames[i].timestamp ?? null,
      score: Math.round(score * 1e4) / 1e4,
      nsfw: score >= THRESHOLD,
    }));
    const flagged = results.filter((r) => r.nsfw).length;
    return res.status(200).json({
      threshold: THRESHOLD,
      results,
      summary: {
        frames: results.length,
        flagged,
        maxScore: Math.max(...results.map((r) => r.score)),
        nsfw: flagged > 0,
      },
    });
  } catch (err) {
    console.error('moderation failed:', err);
    return res.status(500).json({ error: 'inference failed' });
  }
};

module.exports.scoreFrames = scoreFrames;
module.exports.THRESHOLD = THRESHOLD;
