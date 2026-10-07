// Shown in place of the normal chat view while a repo is being indexed - a real circular
// progress ring (based on actual chunks embedded so far, not a fake/indeterminate spinner),
// plus the file count, total size, and a rough time estimate from the backend's scan.

function formatBytes(bytes) {
  if (bytes == null) return "";
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function formatSeconds(s) {
  if (s == null) return "";
  if (s < 60) return `${Math.round(s)}s`;
  const m = Math.floor(s / 60);
  const rem = Math.round(s % 60);
  return rem > 0 ? `${m}m ${rem}s` : `${m}m`;
}

function CircularProgress({ percent, size = 120, stroke = 10 }) {
  const radius = (size - stroke) / 2;
  const circumference = 2 * Math.PI * radius;
  const offset = circumference - (Math.max(0, Math.min(100, percent)) / 100) * circumference;
  const center = size / 2;
  return (
    <svg width={size} height={size} className="circular-progress" role="img" aria-label={`${Math.round(percent)}% indexed`}>
      <circle className="track" cx={center} cy={center} r={radius} strokeWidth={stroke} fill="none" />
      <circle
        className="indicator"
        cx={center}
        cy={center}
        r={radius}
        strokeWidth={stroke}
        fill="none"
        strokeDasharray={circumference}
        strokeDashoffset={offset}
        strokeLinecap="round"
        transform={`rotate(-90 ${center} ${center})`}
      />
      <text x="50%" y="50%" textAnchor="middle" dy="0.35em" className="percent-text">
        {Math.round(percent)}%
      </text>
    </svg>
  );
}

function IndeterminateSpinner({ size = 120, stroke = 10 }) {
  const radius = (size - stroke) / 2;
  const circumference = 2 * Math.PI * radius;
  const center = size / 2;
  return (
    <svg width={size} height={size} className="circular-progress indeterminate" role="img" aria-label="Loading">
      <circle className="track" cx={center} cy={center} r={radius} strokeWidth={stroke} fill="none" />
      <circle
        className="indicator"
        cx={center}
        cy={center}
        r={radius}
        strokeWidth={stroke}
        fill="none"
        strokeDasharray={`${circumference * 0.25} ${circumference * 0.75}`}
        strokeLinecap="round"
      />
    </svg>
  );
}

export default function IndexingView({ indexing }) {
  const { url, fileCount, totalBytes, chunksTotal, chunksDone, estimatedSeconds, stage, complete, error } = indexing;

  if (error) {
    return (
      <div className="indexing-view">
        <h2>Indexing {url} failed</h2>
        <p className="indexing-stats" style={{ color: "var(--brick)" }}>{error}</p>
        <p className="indexing-stats">You can try adding it again from the left, or try a different repository.</p>
      </div>
    );
  }

  if (complete) {
    return (
      <div className="indexing-view">
        <h2>Indexing of {url} is complete</h2>
      </div>
    );
  }

  if (stage === "loading_model") {
    // Deliberately no file count / size / estimate / chunk line here - those describe the
    // EMBEDDING step specifically, which hasn't started yet at this point.
    return (
      <div className="indexing-view">
        <IndeterminateSpinner />
        <h2>Indexing {url}</h2>
        <p className="indexing-stats">Loading the embedding model&hellip; (can take longer the first time)</p>
      </div>
    );
  }

  const percent = chunksTotal ? (chunksDone / chunksTotal) * 100 : 0;
  return (
    <div className="indexing-view">
      <CircularProgress percent={percent} />
      <h2>Indexing {url}</h2>
      <p className="indexing-stats">
        {fileCount} file{fileCount === 1 ? "" : "s"} &middot; {formatBytes(totalBytes)} total
        {estimatedSeconds != null && <> &middot; estimated time: {formatSeconds(estimatedSeconds)}</>}
      </p>
      {chunksTotal > 0 && (
        <p className="indexing-stats">{chunksDone} / {chunksTotal} chunks embedded</p>
      )}
    </div>
  );
}
