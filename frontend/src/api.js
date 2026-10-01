// Thin wrapper around the FastAPI backend's endpoints (see backend/app/api/routes.py).
const BASE = "/api";

export async function listRepos() {
  const r = await fetch(`${BASE}/repos`);
  if (!r.ok) throw new Error("Could not load repos");
  return r.json(); // [{name, url}]
}

export async function addRepo(url) {
  const r = await fetch(`${BASE}/repos`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ url }),
  });
  const data = await r.json();
  if (!r.ok) throw new Error(data.detail || "Could not index that repo");
  return data;
}

export function sendFeedback(repo, question, files, helpful) {
  return fetch(`${BASE}/feedback`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ repo, question, files, helpful }),
  });
}

// Streams agent progress events live via Server-Sent Events, then calls onDone with the
// final answer text. Returns { requestId, stop } - stop() tells the SERVER to actually halt
// work (see /api/ask/stop), not just close the browser's connection.
export function askStream(repo, question, { onEvent, onDone, onStopped, onError }) {
  const requestId = crypto.randomUUID();
  const url = `${BASE}/ask?repo=${encodeURIComponent(repo)}&question=${encodeURIComponent(question)}&request_id=${requestId}`;
  const source = new EventSource(url);
  source.onmessage = (e) => {
    const ev = JSON.parse(e.data);
    onEvent(ev);
    if (ev.agent === "team" && ev.type === "answer") {
      source.close();
      onDone(ev.text);
    } else if (ev.type === "stopped") {
      source.close();
      onStopped();
    } else if (ev.agent === "error") {
      source.close();
      onError(ev.text);
    }
  };
  source.onerror = () => {
    source.close();
    onError("Lost connection to the server.");
  };
  const stop = () => {
    source.close();
    fetch(`${BASE}/ask/stop`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ request_id: requestId }),
    }).catch(() => {}); // best-effort - the stream will time out on its own if this fails
  };
  return { requestId, stop };
}
