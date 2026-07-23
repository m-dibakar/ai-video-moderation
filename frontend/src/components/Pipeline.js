import React from 'react';

// A real ordered sequence — numbering carries information here.
const STAGES = [
  {
    n: '1',
    title: 'Frame sampler',
    body: 'Your browser samples one frame per second (up to 120) straight onto a 224 px canvas — the video itself never leaves your machine.',
    tag: '1 fps · in-browser',
  },
  {
    n: '2',
    title: 'ViT classifier',
    body: 'A fine-tuned Falconsai vision transformer — served as an int8 ONNX model in a serverless function — scores every sampled frame as safe or explicit.',
    tag: 'ViT-B/16 · int8',
  },
  {
    n: '3',
    title: 'Violation grouping',
    body: 'Consecutive flagged frames merge into violation spans, each graded high / medium / low by average confidence.',
    tag: 'spans',
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
