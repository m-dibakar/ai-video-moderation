import React, { useState, useRef, useCallback } from 'react';
import { moderateVideo } from './lib/moderateClient';
import FilmStrip from './components/FilmStrip';
import Benchmarks from './components/Benchmarks';
import Pipeline from './components/Pipeline';
import UploadZone from './components/UploadZone';
import StageStepper from './components/StageStepper';
import Timeline from './components/Timeline';
import ReportPanel from './components/ReportPanel';
import './App.css';

function Wordmark() {
  return (
    <span className="wordmark">
      <svg className="wordmark-glyph" viewBox="0 0 24 24" aria-hidden="true">
        <rect x="2.5" y="5" width="19" height="14" rx="2.5" fill="none" stroke="currentColor" strokeWidth="2" />
        <path d="M10 9.5v5l4.5-2.5z" fill="currentColor" />
      </svg>
      ClearFrame
    </span>
  );
}

function App() {
  const videoRef = useRef(null);
  const [videoFile, setVideoFile] = useState(null);
  const [videoURL, setVideoURL] = useState(null);
  const [status, setStatus] = useState('idle'); // idle | processing | done | error
  const [jobStage, setJobStage] = useState(null); // sampling_frames | classifying
  const [errorMsg, setErrorMsg] = useState('');
  const [report, setReport] = useState(null);
  const [duration, setDuration] = useState(0);
  const [currentTime, setCurrentTime] = useState(0);

  const handleFileSelect = (file) => {
    if (videoURL) URL.revokeObjectURL(videoURL);
    setVideoFile(file);
    setVideoURL(URL.createObjectURL(file));
    setStatus('idle');
    setJobStage(null);
    setErrorMsg('');
    setReport(null);
    setDuration(0);
    setCurrentTime(0);
  };

  const handleAnalyze = async () => {
    if (!videoFile) return;
    try {
      setStatus('processing');
      setJobStage('sampling_frames');
      setErrorMsg('');
      const result = await moderateVideo(videoFile, {
        onProgress: ({ phase }) =>
          setJobStage(phase === 'scoring' ? 'classifying' : 'sampling_frames'),
      });
      setReport(result);
      setStatus('done');
    } catch (err) {
      setStatus('error');
      setErrorMsg(
        /failed to fetch|networkerror/i.test(err.message)
          ? 'Could not reach /api/moderate. In local dev, run "npx vercel dev" — the CRA dev server does not serve API routes.'
          : `Analysis failed: ${err.message}`
      );
    }
  };

  const seekTo = useCallback((seconds) => {
    if (videoRef.current) {
      videoRef.current.currentTime = seconds;
      videoRef.current.play();
    }
  }, []);

  const isProcessing = status === 'processing';
  const stepperIndex =
    status === 'processing' ? (jobStage === 'classifying' ? 2 : 1)
    : status === 'done' ? 4
    : 0;

  return (
    <div className="app">
      <nav className="nav">
        <a href="#top" className="nav-brand"><Wordmark /></a>
        <div className="nav-links">
          <a href="#benchmarks">Benchmarks</a>
          <a href="#pipeline">Pipeline</a>
          <a href="#analyze" className="btn btn-primary btn-sm">Analyze a video</a>
        </div>
      </nav>

      <main id="top">
        {/* ---------- Hero ---------- */}
        <header className="hero">
          <h1 className="hero-title">
            Violations are rare.
            <br />
            Missing them isn&rsquo;t an option.
          </h1>
          <p className="hero-sub">
            ClearFrame screens video frame-by-frame with a fine-tuned vision transformer, trained on an
            IEEE-published variance-stabilized loss built for the 95:5 imbalance of real content libraries.
          </p>
          <div className="hero-actions">
            <a href="#analyze" className="btn btn-primary">Analyze a video</a>
            <a href="#benchmarks" className="btn btn-ghost">See the benchmarks</a>
          </div>
          <FilmStrip />
          <p className="hero-specs">ViT-B/16 · deepghs/nsfw_detect · BCE / Focal / ASL / VS Loss (IEEE SPL 2025)</p>
        </header>

        {/* ---------- Benchmarks ---------- */}
        <section id="benchmarks" className="section">
          <h2>Four loss functions.<br />One brutal imbalance.</h2>
          <p className="section-lede">
            When 95% of frames are safe, BCE learns the lazy shortcut: predict &ldquo;safe&rdquo; everywhere and
            score 98% accuracy while catching nothing. These are the real numbers from a 50-epoch full
            fine-tune — pick a metric and compare.
          </p>
          <Benchmarks />
        </section>

        {/* ---------- Pipeline ---------- */}
        <section id="pipeline" className="section section-tinted">
          <h2>From upload to verdict in five stages.</h2>
          <p className="section-lede">
            Every design choice targets the same failure mode: false negatives that let a violation ship,
            and false positives that erode reviewer trust.
          </p>
          <Pipeline />
        </section>

        {/* ---------- Analyzer ---------- */}
        <section id="analyze" className="section">
          <h2>Run it yourself.</h2>
          <p className="section-lede">
            Pick any video and the pipeline returns a frame-level compliance report with a clickable
            violation timeline. Frames are sampled in your browser and scored serverlessly — the video
            itself never leaves your machine.
          </p>

          <div className="analyzer">
            <div className="analyzer-controls">
              <UploadZone file={videoFile} onFile={handleFileSelect} disabled={isProcessing} />
              <div className="analyzer-cta">
                <button
                  className="btn btn-primary btn-wide"
                  onClick={handleAnalyze}
                  disabled={!videoFile || isProcessing}
                >
                  {isProcessing ? 'Analyzing…' : 'Run analysis'}
                </button>
                {(isProcessing || status === 'done') && <StageStepper activeIndex={stepperIndex} />}
              </div>
            </div>

            {status === 'error' && (
              <div className="alert" role="alert">
                <svg viewBox="0 0 20 20" aria-hidden="true">
                  <circle cx="10" cy="10" r="9" fill="none" stroke="currentColor" strokeWidth="1.6" />
                  <path d="M10 5.5v5.5M10 14.2v.2" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" />
                </svg>
                {errorMsg}
              </div>
            )}

            {videoURL && (
              <div className={`workbench ${report ? 'has-report' : ''}`}>
                <div className="workbench-main">
                  <div className="player-frame">
                    <video
                      ref={videoRef}
                      src={videoURL}
                      controls
                      className="video-player"
                      onLoadedMetadata={() => setDuration(videoRef.current?.duration || 0)}
                      onTimeUpdate={() => setCurrentTime(videoRef.current?.currentTime || 0)}
                    />
                  </div>
                  {report && duration > 0 && (
                    <Timeline
                      violations={report.violations}
                      duration={duration}
                      currentTime={currentTime}
                      onSeek={seekTo}
                    />
                  )}
                </div>

                {report && (
                  <ReportPanel
                    report={report}
                    currentTime={currentTime}
                    onSeek={seekTo}
                    videoName={videoFile?.name}
                  />
                )}
              </div>
            )}
          </div>
        </section>
      </main>

      <footer className="footer">
        <div className="footer-inner">
        <Wordmark />
        <p>
          A research-to-production benchmark of BCE, Focal, Asymmetric, and Variance-Stabilized loss for
          NSFW detection on imbalanced video data. VS Loss: Rabidas, Malakar et&nbsp;al., IEEE Signal Processing Letters, 2025 —{' '}
          <a href="https://doi.org/10.1109/LSP.2025.3625880" target="_blank" rel="noreferrer">
            DOI&nbsp;10.1109/LSP.2025.3625880
          </a>
        </p>
        </div>
      </footer>
    </div>
  );
}

export default App;
