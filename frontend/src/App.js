import React, { useState, useRef, useEffect } from 'react';
import { uploadVideo, pollJob } from './api';
import './App.css';

function App() {
  const videoRef = useRef(null);
  const [videoFile, setVideoFile] = useState(null);
  const [videoURL, setVideoURL] = useState(null);
  const [jobId, setJobId] = useState(null);
  const [status, setStatus] = useState('idle');
  const [statusMsg, setStatusMsg] = useState('');
  const [report, setReport] = useState(null);
  const [duration, setDuration] = useState(0);

  const handleFileSelect = (e) => {
    const file = e.target.files[0];
    if (!file) return;
    setVideoFile(file);
    setVideoURL(URL.createObjectURL(file));
    setStatus('idle');
    setStatusMsg('');
    setReport(null);
    setJobId(null);
  };

  const handleUpload = async () => {
    if (!videoFile) return;
    try {
      setStatus('uploading');
      setStatusMsg('Uploading video...');
      const { job_id } = await uploadVideo(videoFile);
      setJobId(job_id);
      setStatus('processing');
      setStatusMsg('Processing: sampling frames...');
    } catch (err) {
      setStatus('error');
      setStatusMsg(`Upload failed: ${err.message}`);
    }
  };

  useEffect(() => {
    if (status !== 'processing' || !jobId) return;
    const interval = setInterval(async () => {
      try {
        const job = await pollJob(jobId);
        if (job.status === 'sampling_frames') {
          setStatusMsg('Processing: sampling frames...');
        } else if (job.status === 'classifying') {
          setStatusMsg('Processing: classifying frames with ViT...');
        } else if (job.status === 'completed') {
          clearInterval(interval);
          setReport(job.report);
          setStatus('done');
          const v = job.report.summary.total_violations;
          const s = job.report.summary.compliance_status;
          setStatusMsg(`${s} — ${v} violation${v !== 1 ? 's' : ''} detected`);
        } else if (job.status === 'failed') {
          clearInterval(interval);
          setStatus('error');
          setStatusMsg(`Error: ${job.error}`);
        }
      } catch (err) {
        clearInterval(interval);
        setStatus('error');
        setStatusMsg(`Polling failed: ${err.message}`);
      }
    }, 3000);
    return () => clearInterval(interval);
  }, [status, jobId]);

  const jumpTo = (seconds) => {
    if (videoRef.current) {
      videoRef.current.currentTime = seconds;
      videoRef.current.play();
    }
  };

  const onVideoLoaded = () => {
    if (videoRef.current) setDuration(videoRef.current.duration);
  };

  const isProcessing = status === 'uploading' || status === 'processing';
  const passed = report?.summary?.compliance_status === 'PASS';

  return (
    <div className="app">
      <header>
        <h1>🛡️ AI Video Moderation</h1>
        <p>Upload a video to get an AI-powered compliance report</p>
      </header>

      <main>
        {/* Upload Bar */}
        <div className="upload-bar">
          <label className="file-label">
            📁 {videoFile ? videoFile.name : 'Choose a video file'}
            <input type="file" accept="video/*" onChange={handleFileSelect} hidden />
          </label>
          <button onClick={handleUpload} disabled={!videoFile || isProcessing}>
            {isProcessing ? '⏳ Analyzing...' : '🔍 Analyze Video'}
          </button>
          {statusMsg && (
            <span className={`status-msg ${status}`}>
              {status === 'done' && (passed ? '✅ ' : '🚨 ')}
              {statusMsg}
            </span>
          )}
        </div>

        {videoURL && (
          <div className="content-grid">

            {/* Left: Video + Timeline */}
            <div className="left-panel">
              <video
                ref={videoRef}
                src={videoURL}
                controls
                className="video-player"
                onLoadedMetadata={onVideoLoaded}
              />

              {/* Violation Timeline */}
              {report && duration > 0 && (
                <div className="timeline-section">
                  <p className="timeline-label">Violation Timeline</p>
                  <div className="timeline-bar">
                    {report.violations.map((v, i) => (
                      <div
                        key={i}
                        className={`timeline-marker ${v.severity}`}
                        style={{
                          left: `${(v.start_time / duration) * 100}%`,
                          width: `${Math.max(((v.end_time - v.start_time) / duration) * 100, 1)}%`
                        }}
                        onClick={() => jumpTo(v.start_time)}
                        title={`${v.severity} | ${v.timestamp_display}`}
                      />
                    ))}
                  </div>
                  <div className="timeline-ends">
                    <span>00:00</span>
                    <span>{Math.floor(duration / 60)}:{String(Math.floor(duration % 60)).padStart(2, '0')}</span>
                  </div>
                </div>
              )}
            </div>

            {/* Right: Report Panel */}
            {report && (
              <aside className="report-panel">
                {/* Compliance Badge */}
                <div className={`compliance-badge ${passed ? 'pass' : 'fail'}`}>
                  {passed ? '✅ PASS' : '🚨 FAIL'}
                </div>

                {/* Summary Stats */}
                <div className="stats-grid">
                  <div className="stat">
                    <span className="stat-value">{report.summary.total_violations}</span>
                    <span className="stat-label">Violations</span>
                  </div>
                  <div className="stat">
                    <span className="stat-value">{report.summary.violation_percentage}%</span>
                    <span className="stat-label">Flagged</span>
                  </div>
                  <div className="stat">
                    <span className={`stat-value sev high`}>{report.summary.severity_breakdown.high}</span>
                    <span className="stat-label">High</span>
                  </div>
                  <div className="stat">
                    <span className={`stat-value sev medium`}>{report.summary.severity_breakdown.medium}</span>
                    <span className="stat-label">Medium</span>
                  </div>
                </div>

                {/* Model Info */}
                <div className="model-info">
                  <span>Model: {report.model_version}</span>
                  <span>Loss: {report.loss_function_used}</span>
                </div>

                {/* Violations Table */}
                {report.violations.length > 0 ? (
                  <div className="violations-section">
                    <h3>Violations</h3>
                    {report.violations.map((v, i) => (
                      <div
                        key={i}
                        className={`violation-card ${v.severity}`}
                        onClick={() => jumpTo(v.start_time)}
                      >
                        <div className="v-header">
                          <span className="v-time">{v.timestamp_display}</span>
                          <span className={`v-badge ${v.severity}`}>{v.severity}</span>
                          <span className="v-conf">{Math.round(v.confidence * 100)}%</span>
                        </div>
                        <div className="v-details">
                          <span>{v.type.toUpperCase()}</span>
                          <span>{v.duration}s</span>
                          <span className="v-action">{v.suggested_action}</span>
                        </div>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="no-violations">
                    <p>✅ No violations detected</p>
                    <p>This video is compliant.</p>
                  </div>
                )}
              </aside>
            )}

          </div>
        )}
      </main>
    </div>
  );
}

export default App;
