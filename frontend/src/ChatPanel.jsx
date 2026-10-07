import { useEffect, useRef, useState } from "react";
import { askStream, sendFeedback } from "./api.js";
import IndexingView from "./IndexingView.jsx";

// A curated set of genuine source/doc/config extensions, used to recognize a bare file
// citation with no line numbers (e.g. "`hello.py`"). More precise than requiring a "/" in the
// path (which wrongly excludes a real file sitting at a repo's root, with no subdirectory at
// all - a confirmed real case): correctly includes "hello.py" while still excluding generic
// prose like "e.g." or "i.e." (whose trailing letters aren't real extensions). One accepted,
// narrow tradeoff: a proper noun that happens to end in a genuine extension (e.g. "Node.js")
// could still become a link - judged a better tradeoff than missing real citations entirely.
const KNOWN_FILE_EXTENSIONS = new Set([
  "py", "js", "jsx", "ts", "tsx", "java", "go", "rs", "c", "cpp", "cs", "rb", "php",
  "md", "rst", "txt", "json", "yaml", "yml", "toml", "cfg", "ini", "sh", "env",
  "html", "css", "scss", "sql", "xml",
]);

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
// A bracket citation with a bare NUMBER instead of a real filename (e.g. "【0†L1-L6】") is an
// unresolved internal reference index leaking through as literal text, not an actual file path
// - there's genuinely nothing to link to, so the best outcome is removing it cleanly rather
// than showing cryptic symbols or attempting a broken link.
function stripUnresolvedBracketCitations(text) {
  return text.replace(/\s*【\d+†L\d+(?:[-\u2011\u2013\u2014]L\d+)?】/g, "");
}

function normalizeBracketCitations(text) {
  return text.replace(
    /【([\w ./-]+\.\w{1,4})†L(\d+)(?:[-\u2011\u2013\u2014]L(\d+))?】/g,
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
    /`([\w ./-]+\.\w{1,4}):((?:\d+(?:[-\u2011\u2013\u2014]\d+)?\s*,\s*)+\d+(?:[-\u2011\u2013\u2014]\d+)?)`/g,
    (whole, path, rangesStr) => rangesStr.split(",").map((s) => `\`${path}:${s.trim()}\``).join(", ")
  );
  text = text.replace(
    /`([\w ./-]+\.\w{1,4})`((?:\s*,?\s*(?:and\s+)?\(?\s*lines?\s+\d+(?:[-\u2011\u2013\u2014]\d+)?\s*\)?(?:\s*\([^)]*\))?)+)/gi,
    (whole, path, block) => {
      const mentions = [...block.matchAll(/lines?\s+(\d+)(?:[-\u2011\u2013\u2014](\d+))?/gi)];
      return mentions.length === 0
        ? whole
        : mentions.map(([, s, e]) => `\`${path}:${s}${e ? `-${e}` : ""}\``).join(", ");
    }
  );
  return text;
}

// Some answers mention the line range BEFORE the filename, both inside one parenthetical
// ("(lines 43-68, `file.ext`)") - the reverse of the usual "`file.ext` lines 43-68" order.
function normalizeLinesBeforeFileCitations(text) {
  return text.replace(
    /\(\s*lines?\s+(\d+)(?:[-\u2011\u2013\u2014](\d+))?\s*,\s*`([\w ./-]+\.\w{1,4})`\s*\)/gi,
    (whole, start, end, path) => `\`${path}:${start}${end ? `-${end}` : ""}\``
  );
}

// Some answers cite several ranges in the SAME file as separate backtick spans, but only name
// the file in the FIRST one and drop it from the rest ("`file.py:70-80`, `:55-67`, `:33-52`"),
// assuming the reader infers the file from context. Stateful: scans forward, remembering the
// most recently cited real file, and fills it into any later "bare" `:N-M` citation. Must run
// AFTER the other normalizers above, since they're what put real citations into this exact
// "`file:N-M`" form in the first place - this pass needs to see them already in that shape.
function normalizeImplicitFileCitations(text) {
  let lastPath = null;
  return text.replace(
    /`(?:([\w ./-]+\.\w{1,4}))?:(\d+)(?:[-\u2011\u2013\u2014](\d+))?`/g,
    (whole, path, start, end) => {
      if (path) {
        lastPath = path;
        return whole; // already complete and correct - leave unchanged
      }
      return lastPath ? `\`${lastPath}:${start}${end ? `-${end}` : ""}\`` : whole;
    }
  );
}

function renderAnswer(text, repoUrl, repoCommit) {
  text = stripUnresolvedBracketCitations(text);
  text = normalizeBracketCitations(text);
  text = normalizeMultiRangeCitations(text);
  text = normalizeLinesBeforeFileCitations(text);
  text = normalizeImplicitFileCitations(text);
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
          // Path chars include a literal space - real repos can have space-containing file or
          // folder names (common in documentation-style projects) - safe to allow here since
          // `inner` only ever holds content from within a matched pair of backticks (or a bare
          // token that structurally can't contain a space in the first place; see the split
          // regex above, which is deliberately NOT widened the same way).
          const citation = inner.match(/^([\w ./-]+\.\w{1,4}):(\d+)(?:[-\u2011\u2013\u2014](\d+))?$/);
          if (citation) {
            const [, path, start, end] = citation;
            // ?plain=1 forces GitHub's raw/code view instead of its rendered-HTML preview.
            // Without it, line-number anchors (#L55) are silently ignored for any file type
            // GitHub auto-renders (Markdown, reStructuredText, ...) - only .py and similar
            // "plain code" files worked without it, which is exactly the gap that was reported.
            // encodeURI escapes a space (and similar) into a valid URL - needed now that path
            // can contain one.
            const href = repoUrl && repoCommit
              ? `${repoUrl.replace(/\.git$/, "").replace(/\/$/, "")}/blob/${repoCommit}/${encodeURI(path)}` +
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
          // A bare file path with no line numbers at all (e.g. "app/tools/search.py", or just
          // "hello.py" sitting at the repo root) still gets linked - just to the file itself,
          // no #Lxx anchor. See KNOWN_FILE_EXTENSIONS above for why this checks the extension
          // rather than requiring a "/".
          const bareMatch = inner.match(/^([\w ./-]+)\.(\w{1,4})$/);
          if (bareMatch && KNOWN_FILE_EXTENSIONS.has(bareMatch[2].toLowerCase())) {
            const href = repoUrl && repoCommit
              ? `${repoUrl.replace(/\.git$/, "").replace(/\/$/, "")}/blob/${repoCommit}/${encodeURI(inner)}`
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

export default function ChatPanel({ repo, repoUrl, repoCommit, indexing, onTrace, onWorking }) {
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

  if (indexing) {
    // All hooks above still ran, in the same order, regardless of this branch - safe.
    return (
      <div className="chat-col">
        <IndexingView indexing={indexing} />
      </div>
    );
  }

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
            <h2>Ask anything about this repository: {repoUrl || repo}</h2>
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
        {busy && (
          <div className="msg assistant">
            <div className="bubble thinking-bubble">
              <span className="spinner" aria-hidden="true" />
              The team is investigating&hellip;
            </div>
          </div>
        )}
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
