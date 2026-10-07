import { useState } from "react";
import { addRepo, getIndexStatus } from "./api.js";

const POLL_INTERVAL_MS = 1500;
const COMPLETE_MESSAGE_MS = 2000; // how long "indexing complete" shows before switching to the normal chat view

export default function RepoRail({ repos, activeRepo, onSelect, onAdded, onIndexingStarted, onIndexingProgress }) {
  const [url, setUrl] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  function pollUntilDone(name, submittedUrl) {
    const poll = async () => {
      let status;
      try {
        status = await getIndexStatus(name);
      } catch (err) {
        if (err.status === 404) {
          // The job genuinely doesn't exist anymore (most likely: the backend process
          // restarted mid-indexing, e.g. a dev-server auto-reload) - retrying forever would
          // never resolve this, and WAS silently doing exactly that before this fix, which is
          // why this used to look like "stuck with repeated 404s in the server log."
          onIndexingProgress({
            name,
            error: "Lost track of this indexing job (the server may have restarted). Please try adding it again.",
          });
          return;
        }
        setTimeout(poll, POLL_INTERVAL_MS); // a genuine transient blip is fine to just retry
        return;
      }
      if (status.stage === "done") {
        // Show a clear "complete" message for a couple of seconds in the MAIN panel (where
        // the user is actually looking), rather than switching to the normal chat view the
        // instant polling detects completion.
        onIndexingProgress({ name, chunksDone: status.chunks_done, chunksTotal: status.chunks_total, complete: true });
        setTimeout(() => onAdded({ name, url: status.url || submittedUrl, commit: status.commit }), COMPLETE_MESSAGE_MS);
        return;
      }
      if (status.stage === "error") {
        // Show the error prominently in the MAIN panel, not just as small sidebar text that's
        // easy to miss while looking at where the progress view used to be.
        onIndexingProgress({ name, error: status.error || "Indexing failed." });
        return;
      }
      onIndexingProgress({ name, stage: status.stage, chunksDone: status.chunks_done, chunksTotal: status.chunks_total });
      setTimeout(poll, POLL_INTERVAL_MS);
    };
    poll();
  }

  async function handleAdd(e) {
    e.preventDefault();
    const submittedUrl = url.trim();
    if (!submittedUrl) return;
    setBusy(true);
    setError("");
    try {
      const started = await addRepo(submittedUrl);
      setUrl("");
      onIndexingStarted({
        name: started.repo,
        url: submittedUrl,
        fileCount: started.file_count,
        totalBytes: started.total_bytes,
        chunksTotal: started.chunks_total,
        estimatedSeconds: started.estimated_seconds,
      });
      pollUntilDone(started.repo, submittedUrl);
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="rail">
      <div className="brand">
        <span className="mark">&#9670;</span>
        <h1>CodeAtlas</h1>
      </div>

      <div className="section-label">Charted repositories</div>
      {repos.length === 0 && <div className="hint">None indexed yet — add one below.</div>}
      {repos.map((r) => (
        <button
          key={r.name}
          className={`repo-item${r.name === activeRepo ? " active" : ""}`}
          onClick={() => onSelect(r.name)}
          title={r.url || r.name}
        >
          {r.name}
        </button>
      ))}

      <div className="section-label">Chart a new one</div>
      <form className="add-repo" onSubmit={handleAdd}>
        <input
          type="text"
          placeholder="https://github.com/psf/requests"
          value={url}
          onChange={(e) => setUrl(e.target.value)}
          disabled={busy}
        />
        <button type="submit" disabled={busy}>
          {busy ? "Starting…" : "Index repository"}
        </button>
        {error && <div className="hint" style={{ color: "var(--brick)" }}>{error}</div>}
      </form>
    </div>
  );
}
