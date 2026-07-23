# ClearFrame — Frontend & Live Demo

The React app and Vercel serverless function behind
[clearframe-tau.vercel.app](https://clearframe-tau.vercel.app) — the live demo
for the [AI Video Content Moderation](../README.md) project.

## How it works

The video you analyze **never leaves your machine**:

1. **Browser frame sampling** (`src/lib/frameSampler.js`) — the video plays in
   a hidden `<video>` element; ~1 frame per second (up to 120) is drawn onto a
   224×224 canvas and exported as a small JPEG (~15–30 KB each).
2. **Serverless scoring** (`api/moderate.js`) — frames are POSTed in batches
   of 16 (3 in flight) to a Vercel function that runs the int8-quantized ONNX
   export of the VS Loss fine-tuned ViT (`api/_model/`, 87 MB) via
   `onnxruntime-node`, with `sharp` mirroring the training preprocessing
   (224×224 squash + ImageNet normalization).
3. **Report assembly** (`src/lib/moderateClient.js`) — consecutive flagged
   frames are grouped into violation spans with severity and suggested
   actions, rendered by the timeline and report panel.

## Structure

```
api/moderate.js        serverless scoring endpoint (POST /api/moderate)
api/_model/            int8 ONNX model bundled into the function (vercel.json)
src/App.js             page + analyzer state machine
src/lib/               frame sampler + moderation client
src/components/        UploadZone, Pipeline, StageStepper, FilmStrip,
                       Timeline, ReportPanel, Benchmarks
src/index.css          OKLCH design tokens (single source of visual truth)
```

## Develop

```bash
npm install
npm start          # CRA dev server — UI only, /api routes are NOT served
npx vercel dev     # UI + the serverless function (use this to test analysis)
npm run build      # production build
```

Deploy: `npx vercel --prod` (function config — 60 s max duration, bundled
model — lives in `vercel.json`).
