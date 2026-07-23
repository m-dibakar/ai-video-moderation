import React, { useState } from 'react';

// Real results — 50-epoch full fine-tune, 95:5 imbalance (see repo README).
// All four rows use the same protocol: each run's best-F1 checkpoint on the
// shared seed-42 validation split (the same checkpoint the demo deploys).
const MODELS = [
  { name: 'BCE', detail: 'baseline', precision: 0.907, recall: 0.831, f1: 0.867, fn: 10 },
  { name: 'Focal Loss', detail: 'α=0.75, γ=2', precision: 0.881, recall: 0.881, f1: 0.881, fn: 7 },
  { name: 'ASL', detail: 'γ+=0, γ−=4, m=0.05', precision: 0.793, recall: 0.780, f1: 0.786, fn: 13 },
  { name: 'VS Loss', detail: 'IEEE SPL 2025', precision: 0.879, recall: 0.864, f1: 0.872, fn: 8, star: true, live: true },
];

const METRICS = {
  precision: { label: 'Precision', axisMin: 0.7, axisMax: 0.95, fmt: (v) => v.toFixed(3), betterHigh: true },
  recall: { label: 'Recall', axisMin: 0.7, axisMax: 0.95, fmt: (v) => v.toFixed(3), betterHigh: true },
  f1: { label: 'F1', axisMin: 0.7, axisMax: 0.95, fmt: (v) => v.toFixed(3), betterHigh: true },
  fn: { label: 'Missed violations', axisMin: 0, axisMax: 16, fmt: (v) => String(v), betterHigh: false },
};

export default function Benchmarks() {
  const [metric, setMetric] = useState('precision');
  const m = METRICS[metric];
  const values = MODELS.map((r) => r[metric]);
  const best = m.betterHigh ? Math.max(...values) : Math.min(...values);

  return (
    <div className="bench">
      <div className="bench-toggle" role="tablist" aria-label="Benchmark metric">
        {Object.entries(METRICS).map(([key, cfg]) => (
          <button
            key={key}
            role="tab"
            aria-selected={metric === key}
            className={`bench-tab ${metric === key ? 'is-active' : ''}`}
            onClick={() => setMetric(key)}
          >
            {cfg.label}
          </button>
        ))}
      </div>

      <div className="bench-chart">
        {MODELS.map((row) => {
          const v = row[metric];
          const w = Math.max(4, ((v - m.axisMin) / (m.axisMax - m.axisMin)) * 100);
          const isBest = v === best;
          return (
            <div key={row.name} className={`bench-row ${isBest ? 'is-best' : ''}`}>
              <div className="bench-name">
                {row.name}
                {row.star && <span className="bench-star" title="IEEE Signal Processing Letters 2025">★</span>}
                {row.live && <span className="bench-live-tag" title="This checkpoint powers the live demo below">live demo</span>}
                <span className="bench-detail">{row.detail}</span>
              </div>
              <div className="bench-track">
                <div className="bench-bar" style={{ width: `${w}%` }} />
              </div>
              <div className="bench-value">
                {m.fmt(v)}
                {isBest && <span className="bench-best-tag">best</span>}
              </div>
            </div>
          );
        })}
        <p className="bench-axis">
          {m.betterHigh
            ? `axis ${m.axisMin.toFixed(2)} – ${m.axisMax.toFixed(2)}`
            : 'false negatives out of 59 true violations — lower is better'}
        </p>
      </div>

      <p className="bench-cite">
        50-epoch full fine-tune · best-F1 checkpoint per loss · 95:5 class imbalance · deepghs/nsfw_detect (28k images) · ★{' '}
        <em>“Variance Stabilized Loss Function for Semantic Segmentation,”</em> Rabidas, Malakar et&nbsp;al., IEEE
        Signal Processing Letters, 2025.{' '}
        <a href="https://doi.org/10.1109/LSP.2025.3625880" target="_blank" rel="noreferrer">
          DOI 10.1109/LSP.2025.3625880
        </a>
      </p>
    </div>
  );
}
