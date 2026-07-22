import React from 'react';

// A real ordered sequence — numbering carries information here.
const STAGES = [
  {
    n: '1',
    title: 'Frame sampler',
    body: 'decord pulls one frame every 2 seconds — enough temporal resolution to catch any violation lasting 3 s or more.',
    tag: 'decord',
  },
  {
    n: '2',
    title: 'ViT classifier',
    body: 'A fine-tuned Falconsai vision transformer scores every sampled frame as safe or explicit.',
    tag: 'ViT-B/16',
  },
  {
    n: '3',
    title: 'Temporal filter',
    body: 'Detections must persist for at least 3 seconds. Single-frame false positives die here.',
    tag: '≥ 3 s',
  },
  {
    n: '4',
    title: 'Compliance report',
    body: 'Structured JSON: timestamps, severity, confidence, and a suggested action per violation.',
    tag: 'JSON',
  },
  {
    n: '5',
    title: 'Review dashboard',
    body: 'The player, violation timeline, and report you can run below — jump straight to any flagged second.',
    tag: 'React',
  },
];

export default function Pipeline() {
  return (
    <ol className="pipeline">
      {STAGES.map((s) => (
        <li key={s.n} className="pipe-stage">
          <span className="pipe-num" aria-hidden="true">{s.n}</span>
          <h3 className="pipe-title">{s.title}</h3>
          <p className="pipe-body">{s.body}</p>
          <span className="pipe-tag">{s.tag}</span>
        </li>
      ))}
    </ol>
  );
}
