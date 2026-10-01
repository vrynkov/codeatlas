export default function TraceLog({ entries, working }) {
  return (
    <div className="log">
      <div className="section-label">Field log</div>
      {entries.length === 0 && !working && (
        <div className="hint">Ask a question to watch the team work.</div>
      )}
      {entries.map((e, i) => (
        <div className="log-entry" key={i}>
          <span className={`tag ${e.type === "stopped" ? "stopped" : e.agent}`}>
            {e.type === "stopped" ? "stopped" : e.agent}
          </span>
          <span className="text">{e.text}</span>
        </div>
      ))}
      {working && <div className="spinner-line">…still investigating</div>}
    </div>
  );
}
