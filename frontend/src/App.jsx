import { useEffect, useState } from "react";
import ChatPanel from "./ChatPanel.jsx";
import RepoRail from "./RepoRail.jsx";
import TraceLog from "./TraceLog.jsx";
import { listRepos } from "./api.js";

export default function App() {
  const [repos, setRepos] = useState([]); // [{name, url, commit}]
  const [activeRepo, setActiveRepo] = useState(null); // just the name, used for identity/API calls
  const [trace, setTrace] = useState([]);
  const [working, setWorking] = useState(false);
  const [loadError, setLoadError] = useState("");
  // While a repo is being indexed: {name, url, fileCount, totalBytes, chunksTotal,
  // estimatedSeconds, chunksDone} - null once indexing finishes or isn't happening.
  const [indexing, setIndexing] = useState(null);

  useEffect(() => {
    listRepos()
      .then((r) => {
        setRepos(r);
        if (r.length > 0) setActiveRepo(r[0].name);
      })
      .catch(() => setLoadError("Could not reach the CodeAtlas backend. Is it running on :8000?"));
  }, []);

  function handleAdded({ name, url, commit }) {
    setRepos((r) => (r.some((x) => x.name === name) ? r : [...r, { name, url, commit }]));
    setActiveRepo(name);
    setIndexing(null); // indexing really is done now - reveal the normal chat view
  }

  function handleIndexingStarted(info) {
    setIndexing({ ...info, chunksDone: 0 });
    setActiveRepo(info.name); // switch focus to it immediately, even though it's not in `repos` yet
  }

  function handleIndexingProgress(progress) {
    // Merges in new fields (chunksDone/chunksTotal, or an error, or the "complete" flag) over
    // whatever's already known (fileCount, totalBytes, ...) - the error and complete states
    // both still render IN the indexing view (main panel), not as a silent disappearance.
    setIndexing((prev) => (prev ? { ...prev, ...progress } : prev));
  }

  const activeRepoEntry = repos.find((r) => r.name === activeRepo);
  const isIndexingActive = indexing && indexing.name === activeRepo;

  if (loadError) {
    return (
      <div style={{ padding: 40, color: "var(--text)" }}>
        <h2>{loadError}</h2>
        <p className="hint">Start it with: uvicorn app.main:app --reload (from the backend folder)</p>
      </div>
    );
  }

  return (
    <div className="shell">
      <RepoRail
        repos={repos}
        activeRepo={activeRepo}
        onSelect={setActiveRepo}
        onAdded={handleAdded}
        onIndexingStarted={handleIndexingStarted}
        onIndexingProgress={handleIndexingProgress}
      />
      {isIndexingActive ? (
        <ChatPanel repo={activeRepo} indexing={indexing} onTrace={setTrace} onWorking={setWorking} />
      ) : activeRepo ? (
        <ChatPanel
          repo={activeRepo}
          repoUrl={activeRepoEntry?.url}
          repoCommit={activeRepoEntry?.commit}
          onTrace={setTrace}
          onWorking={setWorking}
        />
      ) : (
        <div className="chat-col">
          <div className="empty-state" style={{ margin: "auto" }}>
            <h2>No repository charted yet</h2>
            <p>Add one on the left to begin.</p>
          </div>
        </div>
      )}
      <TraceLog entries={trace} working={working} />
    </div>
  );
}
