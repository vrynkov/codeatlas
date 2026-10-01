import { useEffect, useRef, useState } from "react";
import { askStream, sendFeedback } from "./api.js";

// Turns markdown-lite answer text into styled JSX: fenced ```code blocks```, **bold** headers,
// `inline code`, and file:line citations. The Explainer's answers use all of these, so without
// this, every answer would show literal asterisks and backticks instead of formatting.
function renderAnswer(text) {
  const blocks = text.split(/(```[\s\S]*?```)/g);
  return blocks.map((block, bi) => {
    if (block.startsWith("```")) {
      const code = block.replace(/^```\w*\n?/, "").replace(/```$/, "");
      return (
        <pre key={bi} className="code-block">
          <code>{code}</code>
        </pre>
      );
    }
    const parts = block.split(/(\*\*[^*]+\*\*|`[^`]+`|[\w./-]+\.\w{1,4}:\d+(?:-\d+)?)/g);
    return (
      <span key={bi}>
        {parts.map((p, i) => {
          if (/^\*\*[^*]+\*\*$/.test(p)) {
            return <strong key={i}>{p.slice(2, -2).replace(/`([^`]+)`/g, "$1")}</strong>;
          }
          if (/^[\w./-]+\.\w{1,4}:\d/.test(p) || /^`.*`$/.test(p)) {
            return <code key={i}>{p.replace(/`/g, "")}</code>;
          }
          return <span key={i}>{p}</span>;
        })}
      </span>
    );
  });
}

export default function ChatPanel({ repo, repoUrl, onTrace, onWorking }) {
  const [messages, setMessages] = useState([]); // {role, text, files?}
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const stopRef = useRef(null);
  const scrollRef = useRef(null);
  const traceRef = useRef([]); // shared between ask() (which fills it) and stop() (which can also append to it)

  useEffect(() => {
    setMessages([]);
    onTrace([]);
  }, [repo]);

  useEffect(() => {
    scrollRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  function ask(e) {
    e.preventDefault();
    const question = input.trim();
    if (!question || busy) return;
    setMessages((m) => [...m, { role: "user", text: question }]);
    setInput("");
    setBusy(true);
    onWorking(true);
    traceRef.current = [];
    onTrace([]);
    stopRef.current = askStream(repo, question, {
      onEvent: (ev) => {
        traceRef.current.push(ev);
        onTrace([...traceRef.current]);
      },
      onDone: (answer) => {
        const files = [...new Set([...answer.matchAll(/([\w./-]+\.\w{1,4}):\d+/g)].map((m) => m[1]))];
        setMessages((m) => [...m, { role: "assistant", text: answer, question, files, rated: null }]);
        setBusy(false);
        onWorking(false);
      },
      onStopped: () => {
        traceRef.current.push({ agent: "team", type: "stopped", text: "Stopped before finishing." });
        onTrace([...traceRef.current]);
        setBusy(false);
        onWorking(false);
      },
      onError: (err) => {
        setMessages((m) => [...m, { role: "assistant", text: `Something went wrong: ${err}`, error: true }]);
        setBusy(false);
        onWorking(false);
      },
    });
  }

  function stop() {
    // Closing the connection ourselves (below) means the server's own "stopped" confirmation
    // can never reach us - we cut the line before it could arrive. So the UI resets itself
    // immediately on click, rather than waiting for a server message that's now unreachable.
    traceRef.current.push({ agent: "team", type: "stopped", text: "Stopped by user." });
    onTrace([...traceRef.current]);
    setBusy(false);
    onWorking(false);
    stopRef.current?.stop();
    stopRef.current = null;
  }

  function rate(i, helpful) {
    const msg = messages[i];
    sendFeedback(repo, msg.question, msg.files || [], helpful);
    setMessages((m) => m.map((mm, idx) => (idx === i ? { ...mm, rated: helpful } : mm)));
  }

  useEffect(() => () => stopRef.current?.stop(), []);  // stop cleanly if the user switches repos mid-answer

  return (
    <div className="chat-col">
      <div className="chat-header">
        <h2>
          Surveying <span className="repo-name">{repoUrl || repo}</span>
        </h2>
      </div>

      <div className="messages">
        {messages.length === 0 && (
          <div className="empty-state">
            <h2>Ask about anything in this repository</h2>
            <p>Try "How are retries handled?" or "Where is authentication added to a request?"</p>
          </div>
        )}
        {messages.map((m, i) => (
          <div className={`msg ${m.role}`} key={i}>
            <div className="bubble">{m.role === "assistant" ? renderAnswer(m.text) : m.text}</div>
            {m.role === "assistant" && !m.error && (
              <div className="feedback-row">
                <button className={m.rated === true ? "given" : ""} onClick={() => rate(i, true)}>
                  Helpful
                </button>
                <button className={m.rated === false ? "given" : ""} onClick={() => rate(i, false)}>
                  Not quite
                </button>
              </div>
            )}
          </div>
        ))}
        <div ref={scrollRef} />
      </div>

      <form className="composer" onSubmit={ask}>
        <input
          placeholder={busy ? "The team is investigating…" : "Ask about this repository…"}
          value={input}
          onChange={(e) => setInput(e.target.value)}
          disabled={busy}
        />
        {busy ? (
          <button type="button" className="stop-btn" onClick={stop}>
            Stop
          </button>
        ) : (
          <button type="submit" disabled={!input.trim()}>
            Ask
          </button>
        )}
      </form>
    </div>
  );
}
