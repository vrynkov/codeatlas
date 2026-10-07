import { useEffect, useRef, useState } from "react";
import { askStream, sendFeedback } from "./api.js";

// Turns markdown-lite answer text into styled JSX: fenced ```code blocks```, **bold** headers,
// `inline code`, and file:line citations. The Explainer's answers use all of these, so without
// this, every answer would show literal asterisks and backticks instead of formatting.
//
// Citations ALSO become real links straight to the exact file and line on GitHub, pinned to the
// exact commit that was actually indexed (not just the branch, which can move after the fact) -
// so clicking one takes you to precisely the code CodeAtlas read, not a stale/wrong line.
// Some models emit a file-citation marker like 【README.md†L9-L13】 as literal visible text
// (an OpenAI-style retrieval citation annotation) instead of it being specially rendered.
// This normalizes it into the "`file:N-M`" form the rest of renderAnswer already understands.
function normalizeBracketCitations(text) {
  return text.replace(
    /【([\w./-]+\.\w{1,4})†L(\d+)(?:[-\u2011\u2013\u2014]L(\d+))?】/g,
    (_, path, start, end) => `\`${path}:${start}${end ? `-${end}` : ""}\``
  );
}

// Some answers pack several ranges into one citation ("`file.py:1-5, 38-43, 45-53`"), or name
// a file once in backticks and then describe its ranges separately in prose ("`file.py` lines
// 34-45", sometimes several such mentions chained with "and" for the same file). This expands
// either shape into one clean "`file:N-M`" citation per range, so each becomes its own link
// instead of the whole thing falling back to plain, unlinked text.
function normalizeMultiRangeCitations(text) {
  text = text.replace(
    /`([\w./-]+\.\w{1,4}):((?:\d+(?:[-\u2011\u2013\u2014]\d+)?\s*,\s*)+\d+(?:[-\u2011\u2013\u2014]\d+)?)`/g,
    (whole, path, rangesStr) => rangesStr.split(",").map((s) => `\`${path}:${s.trim()}\``).join(", ")
  );
  text = text.replace(
    /`([\w./-]+\.\w{1,4})`((?:\s*,?\s*(?:and\s+)?\(?\s*lines?\s+\d+(?:[-\u2011\u2013\u2014]\d+)?\s*\)?(?:\s*\([^)]*\))?)+)/gi,
    (whole, path, block) => {
      const mentions = [...block.matchAll(/lines?\s+(\d+)(?:[-\u2011\u2013\u2014](\d+))?/gi)];
      return mentions.length === 0
        ? whole
        : mentions.map(([, s, e]) => `\`${path}:${s}${e ? `-${e}` : ""}\``).join(", ");
    }
  );
  return text;
}

function renderAnswer(text, repoUrl, repoCommit) {
  text = normalizeBracketCitations(text);
  text = normalizeMultiRangeCitations(text);
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
    // The range separator accepts a plain hyphen AND the "fancy" dash characters models
    // sometimes write instead (non-breaking hyphen U+2011, en dash, em dash) - without this,
    // a citation like "file.py:11‑15" (U+2011) wouldn't be recognized as a citation at all.
    const parts = block.split(/(\*\*[^*]+\*\*|`[^`]+`|[\w./-]+\.\w{1,4}:\d+(?:[-\u2011\u2013\u2014]\d+)?)/g);
    return (
      <span key={bi}>
        {parts.map((p, i) => {
          if (/^\*\*[^*]+\*\*$/.test(p)) {
            return <strong key={i}>{p.slice(2, -2).replace(/`([^`]+)`/g, "$1")}</strong>;
          }
          // A citation is usually written wrapped in a single pair of backticks (the common
          // real case) but sometimes appears bare - unwrap one layer before checking either way.
          const inner = /^`([^`]*)`$/.test(p) ? p.slice(1, -1) : p;
          const citation = inner.match(/^([\w./-]+\.\w{1,4}):(\d+)(?:[-\u2011\u2013\u2014](\d+))?$/);
          if (citation) {
            const [, path, start, end] = citation;
            // ?plain=1 forces GitHub's raw/code view instead of its rendered-HTML preview.
            // Without it, line-number anchors (#L55) are silently ignored for any file type
            // GitHub auto-renders (Markdown, reStructuredText, ...) - only .py and similar
            // "plain code" files worked without it, which is exactly the gap that was reported.
            const href = repoUrl && repoCommit
              ? `${repoUrl.replace(/\.git$/, "").replace(/\/$/, "")}/blob/${repoCommit}/${path}` +
                `?plain=1#L${start}${end ? `-L${end}` : ""}`
              : null;
            return href ? (
              <a key={i} href={href} target="_blank" rel="noreferrer" className="citation-link">
                <code>{inner}</code>
              </a>
            ) : (
              <code key={i}>{inner}</code> // older repo, indexed before this feature - no link, same as before
            );
          }
          // A bare file path with no line numbers at all (e.g. "app/tools/search.py") still gets
          // linked - just to the file itself, no #Lxx anchor. Requiring a "/" is deliberate: it
          // safely excludes things that merely LOOK path-like ("Node.js", "e.g.") but aren't
          // real repo-relative paths, without needing a list of known extensions to check against.
          if (inner.includes("/") && /^[\w./-]+\.\w{1,4}$/.test(inner)) {
            const href = repoUrl && repoCommit
              ? `${repoUrl.replace(/\.git$/, "").replace(/\/$/, "")}/blob/${repoCommit}/${inner}`
              : null;
            return href ? (
              <a key={i} href={href} target="_blank" rel="noreferrer" className="citation-link">
                <code>{inner}</code>
              </a>
            ) : (
              <code key={i}>{inner}</code>
            );
          }
          if (/^`.*`$/.test(p)) {
            return <code key={i}>{p.replace(/`/g, "")}</code>;
          }
          return <span key={i}>{p}</span>;
        })}
      </span>
    );
  });
}

export default function ChatPanel({ repo, repoUrl, repoCommit, onTrace, onWorking }) {
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
            <div className="bubble">
              {m.role === "assistant" ? renderAnswer(m.text, repoUrl, repoCommit) : m.text}
            </div>
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
