import React from 'react';

export default function ReportPanel({ report, currentTime, onSeek, videoName }) {
  const passed = report.summary.compliance_status === 'PASS';

  const downloadReport = () => {
    const blob = new Blob([JSON.stringify(report, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${(videoName || 'video').replace(/\.[^.]+$/, '')}-compliance-report.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <aside className="report">
      <div className={`verdict ${passed ? 'is-pass' : 'is-fail'}`}>
        <span className="verdict-glyph" aria-hidden="true">
          {passed ? (
            <svg viewBox="0 0 20 20">
              <circle cx="10" cy="10" r="9" fill="none" stroke="currentColor" strokeWidth="1.6" />
              <path d="M6 10.4l2.6 2.6 5.4-6" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" />
            </svg>
          ) : (
            <svg viewBox="0 0 20 20">
              <path d="M10 2.2L18.6 17H1.4z" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinejoin="round" />
              <path d="M10 8v4.2M10 14.6v.2" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" />
            </svg>
          )}
        </span>
        <div>
          <p className="verdict-word">{passed ? 'Compliant' : 'Flagged'}</p>
          <p className="verdict-sub">
            {passed
              ? 'No policy violations detected'
              : `${report.summary.total_violations} violation${report.summary.total_violations !== 1 ? 's' : ''} · ${report.summary.violation_percentage}% of frames flagged`}
          </p>
        </div>
      </div>

      <dl className="readout">
        <div className="readout-cell">
          <dt>Violations</dt>
          <dd>{report.summary.total_violations}</dd>
        </div>
        <div className="readout-cell">
          <dt>Flagged</dt>
          <dd>{report.summary.violation_percentage}%</dd>
        </div>
        <div className="readout-cell">
          <dt>High</dt>
          <dd className="is-high">{report.summary.severity_breakdown.high}</dd>
        </div>
        <div className="readout-cell">
          <dt>Medium</dt>
          <dd className="is-medium">{report.summary.severity_breakdown.medium}</dd>
        </div>
      </dl>

      <p className="model-line">
        <span>{report.model_version}</span>
        <span>{report.loss_function_used}</span>
      </p>

      {report.violations.length > 0 ? (
        <ul className="v-list">
          {report.violations.map((v, i) => {
            const active = currentTime >= v.start_time && currentTime <= v.end_time;
            return (
              <li key={i}>
                <button
                  className={`v-row is-${v.severity} ${active ? 'is-active' : ''}`}
                  onClick={() => onSeek(v.start_time)}
                  aria-label={`Jump to ${v.severity} violation at ${v.timestamp_display}`}
                >
                  <span className="v-time">{v.timestamp_display}</span>
                  <span className={`v-pill is-${v.severity}`}>{v.severity}</span>
                  <span className="v-type">{v.type}</span>
                  <span className="v-conf">
                    <span className="v-conf-track" aria-hidden="true">
                      <span className="v-conf-fill" style={{ width: `${Math.round(v.confidence * 100)}%` }} />
                    </span>
                    {Math.round(v.confidence * 100)}%
                  </span>
                  <span className="v-action">{v.suggested_action}</span>
                </button>
              </li>
            );
          })}
        </ul>
      ) : (
        <div className="v-empty">
          <p>Every sampled frame classified safe.</p>
          <p>Nothing persisted past the 3-second temporal filter.</p>
        </div>
      )}

      <button className="btn btn-ghost report-dl" onClick={downloadReport}>
        Download JSON report
      </button>
    </aside>
  );
}
