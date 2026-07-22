import React from 'react';

const STAGES = ['Upload', 'Sample frames', 'Classify', 'Report'];

/**
 * activeIndex: index of the stage currently running (stages before it are done).
 * Pass activeIndex = STAGES.length when everything is complete.
 */
export default function StageStepper({ activeIndex }) {
  return (
    <ol className="stepper" aria-label="Analysis progress">
      {STAGES.map((label, i) => {
        const state = i < activeIndex ? 'done' : i === activeIndex ? 'active' : 'todo';
        return (
          <li key={label} className={`step is-${state}`} aria-current={state === 'active' ? 'step' : undefined}>
            <span className="step-dot" aria-hidden="true">
              {state === 'done' && (
                <svg viewBox="0 0 10 10">
                  <path d="M1.8 5.4l2.3 2.3 4.1-5" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" />
                </svg>
              )}
            </span>
            <span className="step-label">{label}</span>
            {i < STAGES.length - 1 && <span className="step-rail" aria-hidden="true" />}
          </li>
        );
      })}
    </ol>
  );
}
