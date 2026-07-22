import React from 'react';

const SWEEP_S = 7;

// Illustrative inference pass — mirrors real model behavior (rare positives).
const FRAMES = [
  { t: '00:00', conf: 0.99, flag: false },
  { t: '00:02', conf: 0.98, flag: false },
  { t: '00:04', conf: 0.97, flag: false },
  { t: '00:06', conf: 0.99, flag: false },
  { t: '00:08', conf: 0.94, flag: true },
  { t: '00:10', conf: 0.91, flag: true },
  { t: '00:12', conf: 0.98, flag: false },
  { t: '00:14', conf: 0.99, flag: false },
  { t: '00:16', conf: 0.87, flag: true },
  { t: '00:18', conf: 0.98, flag: false },
  { t: '00:20', conf: 0.99, flag: false },
  { t: '00:22', conf: 0.97, flag: false },
];

/* Tiny abstract "scene" so each frame reads as a distinct video still. */
function FrameScene({ seed }) {
  const hillY = 26 + ((seed * 7) % 12);
  const sunX = 10 + ((seed * 13) % 36);
  const sunY = 8 + ((seed * 5) % 10);
  return (
    <svg viewBox="0 0 56 40" aria-hidden="true" focusable="false">
      <rect width="56" height="40" fill="oklch(0.93 0.006 140)" />
      <circle cx={sunX} cy={sunY} r="4" fill="oklch(0.82 0.02 140)" />
      <path
        d={`M0 ${hillY} L${14 + ((seed * 3) % 10)} ${hillY - 10} L${30 + ((seed * 11) % 8)} ${hillY + 2} L44 ${hillY - 6} L56 ${hillY + 1} V40 H0 Z`}
        fill="oklch(0.78 0.015 140)"
      />
      <rect y="34" width="56" height="6" fill="oklch(0.7 0.015 140)" />
    </svg>
  );
}

export default function FilmStrip() {
  return (
    <figure className="filmstrip" aria-label="Animated diagram: a scanline sweeps a strip of video frames and each frame receives a safe or flagged verdict">
      <div className="fs-strip">
        {FRAMES.map((f, i) => {
          const delay = ((i + 0.5) / FRAMES.length) * SWEEP_S;
          return (
            <div
              key={f.t}
              className={`fs-frame ${f.flag ? 'is-flag' : 'is-pass'}`}
              style={{ '--fs-delay': `${delay}s` }}
              data-tip={`${f.flag ? 'explicit' : 'safe'} · ${f.conf.toFixed(2)}`}
            >
              <FrameScene seed={i + 3} />
              <span className="fs-badge" aria-hidden="true">
                {f.flag ? (
                  <svg viewBox="0 0 10 10"><path d="M2 2l6 6M8 2l-6 6" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" /></svg>
                ) : (
                  <svg viewBox="0 0 10 10"><path d="M1.8 5.4l2.3 2.3 4.1-5" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" /></svg>
                )}
              </span>
              <span className="fs-time">{f.t}</span>
            </div>
          );
        })}
        <div className="fs-scanline" aria-hidden="true" />
      </div>
      <figcaption className="fs-caption">
        Model inference, illustrative — 1 frame sampled every 2 s, each scored by the fine-tuned ViT
      </figcaption>
    </figure>
  );
}
