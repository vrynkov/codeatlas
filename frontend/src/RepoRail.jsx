import { useState } from "react";
import { addRepo } from "./api.js";

export default function RepoRail({ repos, activeRepo, onSelect, onAdded }) {
  const [url, setUrl] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  async function handleAdd(e) {
    e.preventDefault();
    if (!url.trim()) return;
    setBusy(true);
    setError("");
    try {
      const { repo, commit } = await addRepo(url.trim());
      setUrl("");
      onAdded({ name: repo, url: url.trim(), commit }); // URL we already had; commit comes from the response
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
          {busy ? "Indexing…" : "Index repository"}
        </button>
        {busy && <div className="hint">First run downloads the embedding model — can take a minute.</div>}
        {error && <div className="hint" style={{ color: "var(--brick)" }}>{error}</div>}
      </form>
    </div>
  );
}
