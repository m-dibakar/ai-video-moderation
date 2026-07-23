// Browser-side frame sampler: decodes the video with the native <video>
// element and samples frames onto a 224x224 canvas, so only small JPEGs
// (~15-30KB each) ever leave the client. This is what lets the whole
// pipeline run on free tiers — no video upload, no server-side ffmpeg.
//
// The 224x224 squash (no letterboxing) deliberately matches the training
// transform (transforms.Resize((224,224))); do not "fix" the aspect ratio.

const SIZE = 224;

function withTimeout(promise, ms, message) {
  return Promise.race([
    promise,
    new Promise((_, reject) => setTimeout(() => reject(new Error(message)), ms)),
  ]);
}

function once(target, event, errEvent = 'error') {
  return new Promise((resolve, reject) => {
    const onOk = () => { cleanup(); resolve(); };
    const onErr = () => { cleanup(); reject(new Error(`video ${errEvent}`)); };
    const cleanup = () => {
      target.removeEventListener(event, onOk);
      target.removeEventListener(errEvent, onErr);
    };
    target.addEventListener(event, onOk, { once: true });
    target.addEventListener(errEvent, onErr, { once: true });
  });
}

/**
 * Sample frames evenly across a video file.
 *
 * NSFW moments in real videos can be sub-second shots (measured 0.25-1.0s
 * windows on trailer footage, with hard-bimodal scores and no shoulder), so
 * sampling density — not the threshold — decides whether they are caught.
 * 1fps with a 120-frame cap hits such windows reliably up to ~2min videos;
 * the old 20-frame/0.5fps default missed all of them on a 2.5min trailer.
 *
 * @param {File|Blob} file        video file from an <input> or drop zone
 * @param {object}   [opts]
 * @param {number}   [opts.maxFrames=120] hard cap on sampled frames
 * @param {number}   [opts.targetFps=1]   sampling rate for short videos
 * @param {number}   [opts.jpegQuality=0.9]
 * @param {function} [opts.onProgress]    (done, total) => void
 * @returns {Promise<{duration: number, frames: Array<{timestamp: number, jpegBase64: string}>}>}
 */
export async function sampleFrames(file, opts = {}) {
  const { maxFrames = 120, targetFps = 1, jpegQuality = 0.9, onProgress } = opts;

  const url = URL.createObjectURL(file);
  const video = document.createElement('video');
  video.muted = true;
  video.playsInline = true;
  video.preload = 'auto';
  video.src = url;
  // Keep the video in the DOM (invisibly): requestVideoFrameCallback never
  // fires for an off-DOM video in Chrome, which would leave every frame
  // waiting out the full fallback timer below.
  video.style.cssText =
    'position:fixed;left:0;top:0;width:2px;height:2px;opacity:0;pointer-events:none;';
  document.body.appendChild(video);

  try {
    await withTimeout(once(video, 'loadedmetadata'), 15000,
      'video metadata never loaded — format may be unsupported by this browser');

    // MediaRecorder-produced webm reports Infinity until forced to the end.
    if (!isFinite(video.duration)) {
      video.currentTime = Number.MAX_SAFE_INTEGER;
      await withTimeout(once(video, 'durationchange'), 10000, 'could not determine video duration');
      video.currentTime = 0;
    }
    const duration = video.duration;
    if (!isFinite(duration) || duration <= 0) {
      throw new Error('could not read video duration');
    }

    const n = Math.max(1, Math.min(maxFrames, Math.ceil(duration * targetFps)));
    // midpoints of n equal slices — avoids the (often black) first/last frame
    const timestamps = Array.from({ length: n }, (_, i) => ((i + 0.5) * duration) / n);

    const canvas = document.createElement('canvas');
    canvas.width = SIZE;
    canvas.height = SIZE;
    const ctx = canvas.getContext('2d', { willReadFrequently: false });

    const frames = [];
    for (let i = 0; i < timestamps.length; i++) {
      video.currentTime = Math.min(timestamps[i], Math.max(0, duration - 0.1));
      await withTimeout(once(video, 'seeked'), 10000,
        `seek to ${timestamps[i].toFixed(1)}s timed out`);
      // Let the decoder present the seeked frame before drawing.
      // requestVideoFrameCallback is the precise (and usually much faster)
      // signal now that the video is in the DOM, but it still only races the
      // timer — it must NOT be awaited unconditionally (browsers without it,
      // or edge cases where it stops firing, would hang sampling forever).
      await new Promise((resolve) => {
        if ('requestVideoFrameCallback' in video) video.requestVideoFrameCallback(() => resolve());
        setTimeout(resolve, 150);
      });
      ctx.drawImage(video, 0, 0, SIZE, SIZE);
      const dataUrl = canvas.toDataURL('image/jpeg', jpegQuality);
      frames.push({
        timestamp: Math.round(timestamps[i] * 100) / 100,
        jpegBase64: dataUrl.slice(dataUrl.indexOf(',') + 1),
      });
      if (onProgress) onProgress(i + 1, timestamps.length);
    }
    return { duration, frames };
  } finally {
    video.remove();
    video.removeAttribute('src');
    video.load();
    URL.revokeObjectURL(url);
  }
}
