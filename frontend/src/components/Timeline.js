import React, { useRef } from 'react';

function fmtTime(s) {
  const m = Math.floor(s / 60);
  const sec = Math.floor(s % 60);
  return `${String(m).padStart(2, '0')}:${String(sec).padStart(2, '0')}`;
}

export default function Timeline({ violations, duration, currentTime, onSeek }) {
  const trackRef = useRef(null);

  const seekFromPointer = (e) => {
    const rect = trackRef.current.getBoundingClientRect();
    const frac = Math.min(1, Math.max(0, (e.clientX - rect.left) / rect.width));
    onSeek(frac * duration);
  };

  return (
    <div className="timeline">
      <div className="tl-head">
        <span className="tl-label">Violation timeline</span>
        <span className="tl-legend">
          <span className="tl-key is-high">high</span>
          <span className="tl-key is-medium">medium</span>
          <span className="tl-key is-low">low</span>
        </span>
      </div>

      {/* eslint-disable-next-line jsx-a11y/no-noninteractive-element-interactions */}
      <div
        ref={trackRef}
        className="tl-track"
        onClick={seekFromPointer}
        role="slider"
        tabIndex={0}
        aria-label="Seek video"
        aria-valuemin={0}
        aria-valuemax={Math.round(duration)}
        aria-valuenow={Math.round(currentTime)}
        aria-valuetext={fmtTime(currentTime)}
        onKeyDown={(e) => {
          if (e.key === 'ArrowRight') onSeek(Math.min(duration, currentTime + 5));
          if (e.key === 'ArrowLeft') onSeek(Math.max(0, currentTime - 5));
        }}
      >
        <div className="tl-progress" style={{ width: `${(currentTime / duration) * 100}%` }} />
        {violations.map((v, i) => (
          <button
            key={i}
            className={`tl-marker is-${v.severity}`}
            style={{
              left: `${(v.start_time / duration) * 100}%`,
              width: `${Math.max(((v.end_time - v.start_time) / duration) * 100, 0.8)}%`,
            }}
            onClick={(e) => {
              e.stopPropagation();
              onSeek(v.start_time);
            }}
            data-tip={`${v.severity} · ${v.timestamp_display}`}
            aria-label={`Jump to ${v.severity} severity violation at ${v.timestamp_display}`}
          />
        ))}
        <div className="tl-playhead" style={{ left: `${(currentTime / duration) * 100}%` }} aria-hidden="true" />
      </div>

      <div className="tl-ends">
        <span>{fmtTime(0)}</span>
        <span>{fmtTime(currentTime)}</span>
        <span>{fmtTime(duration)}</span>
      </div>
    </div>
  );
}
