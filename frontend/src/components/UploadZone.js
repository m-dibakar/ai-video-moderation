import React, { useRef, useState } from 'react';

function fmtSize(bytes) {
  if (bytes >= 1e9) return `${(bytes / 1e9).toFixed(2)} GB`;
  if (bytes >= 1e6) return `${(bytes / 1e6).toFixed(1)} MB`;
  return `${Math.round(bytes / 1e3)} KB`;
}

export default function UploadZone({ file, onFile, disabled }) {
  const inputRef = useRef(null);
  const [dragOver, setDragOver] = useState(false);

  const pick = (f) => {
    if (f && f.type.startsWith('video/')) onFile(f);
  };

  const onDrop = (e) => {
    e.preventDefault();
    setDragOver(false);
    if (disabled) return;
    pick(e.dataTransfer.files?.[0]);
  };

  return (
    <div
      className={`dropzone ${dragOver ? 'is-over' : ''} ${file ? 'has-file' : ''} ${disabled ? 'is-disabled' : ''}`}
      onDragOver={(e) => {
        e.preventDefault();
        if (!disabled) setDragOver(true);
      }}
      onDragLeave={() => setDragOver(false)}
      onDrop={onDrop}
      onClick={() => !disabled && inputRef.current?.click()}
      onKeyDown={(e) => {
        if ((e.key === 'Enter' || e.key === ' ') && !disabled) {
          e.preventDefault();
          inputRef.current?.click();
        }
      }}
      role="button"
      tabIndex={disabled ? -1 : 0}
      aria-label={file ? `Selected video ${file.name}. Activate to choose a different file.` : 'Choose a video file to analyze'}
    >
      <input
        ref={inputRef}
        type="file"
        accept="video/*"
        hidden
        onChange={(e) => pick(e.target.files?.[0])}
      />
      <svg className="dz-glyph" viewBox="0 0 40 40" aria-hidden="true">
        <rect x="3" y="8" width="34" height="24" rx="3" fill="none" stroke="currentColor" strokeWidth="2" />
        <path d="M17 15.5v9l7.5-4.5z" fill="currentColor" />
        <path d="M3 14h34M3 26h34" stroke="currentColor" strokeWidth="1" opacity="0.35" />
      </svg>
      {file ? (
        <>
          <p className="dz-title">{file.name}</p>
          <p className="dz-sub">
            <span className="dz-mono">{fmtSize(file.size)}</span> — drop or click to replace
          </p>
        </>
      ) : (
        <>
          <p className="dz-title">Drop a video here</p>
          <p className="dz-sub">or click to browse — any format your browser can play</p>
        </>
      )}
    </div>
  );
}
