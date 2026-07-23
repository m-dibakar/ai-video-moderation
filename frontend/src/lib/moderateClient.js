// Client for the Vercel moderation pipeline: sample frames in the browser,
// score them via /api/moderate in batches, and assemble the same report
// shape the FastAPI backend produced (backend/report_generator.py), so
// ReportPanel / Timeline render it unchanged.

import { sampleFrames } from './frameSampler';

const BATCH_SIZE = 16; // stays well under the function's 32-frame cap
const BATCH_CONCURRENCY = 3; // parallel /api/moderate calls — Vercel scales instances
const MODEL_VERSION = 'ViT-B/16 · VS Loss full fine-tune · int8 ONNX';
const LOSS_LINE = 'VS Loss (IEEE SPL 2025)';

function fmtTime(seconds) {
  const m = String(Math.floor(seconds / 60)).padStart(2, '0');
  const s = String(Math.floor(seconds % 60)).padStart(2, '0');
  return `${m}:${s}`;
}

// Mirrors backend/temporal_filter.py severity mapping.
function severityFor(avgConf) {
  if (avgConf > 0.85) return { severity: 'high', suggested_action: 'auto-blur' };
  if (avgConf > 0.65) return { severity: 'medium', suggested_action: 'flag-for-review' };
  return { severity: 'low', suggested_action: 'log-only' };
}

// Group consecutive flagged frames into violation spans. Each sampled frame
// stands in for a slice of the video, so a flagged frame covers
// [timestamp - slice/2, timestamp + slice/2].
function buildViolations(results, duration) {
  const slice = results.length ? duration / results.length : 0;
  const violations = [];
  let run = null;

  results.forEach((r, i) => {
    if (r.nsfw) {
      if (run && i === run.lastIndex + 1) {
        run.frames.push(r);
        run.lastIndex = i;
      } else {
        if (run) violations.push(run);
        run = { frames: [r], lastIndex: i };
      }
    }
  });
  if (run) violations.push(run);

  return violations.map((v, i) => {
    const start = Math.max(0, v.frames[0].timestamp - slice / 2);
    const end = Math.min(duration, v.frames[v.frames.length - 1].timestamp + slice / 2);
    const avgConf = v.frames.reduce((s, f) => s + f.score, 0) / v.frames.length;
    return {
      id: i + 1,
      start_time: Math.round(start * 100) / 100,
      end_time: Math.round(end * 100) / 100,
      duration: Math.round((end - start) * 100) / 100,
      timestamp_display: fmtTime(start),
      type: 'nsfw',
      confidence: Math.round(avgConf * 1000) / 1000,
      ...severityFor(avgConf),
    };
  });
}

/**
 * Moderate a video file end to end: browser sampling + serverless scoring.
 *
 * @param {File|Blob} file
 * @param {object}   [opts]
 * @param {number}   [opts.maxFrames=120]
 * @param {function} [opts.onProgress]  ({ phase: 'sampling'|'scoring', done, total }) => void
 * @returns {Promise<object>} report in the backend/report_generator.py shape
 */
export async function moderateVideo(file, opts = {}) {
  const { maxFrames = 120, onProgress } = opts;

  const { duration, frames } = await sampleFrames(file, {
    maxFrames,
    onProgress: (done, total) =>
      onProgress && onProgress({ phase: 'sampling', done, total }),
  });

  const batches = [];
  for (let i = 0; i < frames.length; i += BATCH_SIZE) {
    batches.push(frames.slice(i, i + BATCH_SIZE));
  }

  const batchResults = new Array(batches.length);
  let threshold = null;
  let scored = 0;
  let next = 0;
  async function worker() {
    while (next < batches.length) {
      const idx = next++;
      const res = await fetch('/api/moderate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ frames: batches[idx] }),
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.error || `moderation API failed (${res.status})`);
      }
      const data = await res.json();
      threshold = data.threshold;
      batchResults[idx] = data.results;
      scored += batches[idx].length;
      if (onProgress) {
        onProgress({ phase: 'scoring', done: scored, total: frames.length });
      }
    }
  }
  await Promise.all(
    Array.from({ length: Math.min(BATCH_CONCURRENCY, batches.length) }, worker)
  );
  const results = batchResults.flat();

  const violations = buildViolations(results, duration);
  const violationTime = violations.reduce((s, v) => s + v.duration, 0);

  return {
    video: file.name || 'video',
    duration_seconds: Math.round(duration * 100) / 100,
    model_version: MODEL_VERSION,
    loss_function_used: LOSS_LINE,
    threshold,
    frames_sampled: results.length,
    frame_scores: results, // per-frame detail for the JSON download
    summary: {
      total_violations: violations.length,
      violation_time_seconds: Math.round(violationTime * 100) / 100,
      violation_percentage: Math.round((violationTime / Math.max(duration, 1)) * 1000) / 10,
      compliance_status: violations.length ? 'FAIL' : 'PASS',
      severity_breakdown: {
        high: violations.filter((v) => v.severity === 'high').length,
        medium: violations.filter((v) => v.severity === 'medium').length,
        low: violations.filter((v) => v.severity === 'low').length,
      },
    },
    violations,
  };
}
