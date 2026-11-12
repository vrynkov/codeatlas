"""Settings, loaded from the .env file (never commit that file!)."""
import os
from pathlib import Path

from dotenv import load_dotenv

BACKEND_DIR = Path(__file__).resolve().parents[1]  # the backend/ folder, wherever we are launched from
load_dotenv(BACKEND_DIR / ".env")

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
# .strip() matters here specifically: a trailing newline in a Secret Manager value (easy to
# introduce accidentally when the secret file was created, e.g. via a text editor that adds
# one on save) makes the Authorization header itself technically invalid HTTP, which fails
# INSTANTLY, before any network request is even attempted - surfacing as a generic
# "Connection error." that's very easy to mistake for an actual connectivity problem.
# If Groq retires this model, pick another tool-capable one from https://console.groq.com/docs/models
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

DATA_DIR = BACKEND_DIR / "data"
REPOS_DIR = DATA_DIR / "repos"
DB_DIR = DATA_DIR / "chroma"

MAX_AGENT_STEPS = 10      # safety limit: stops endless loops (and protects your free quota)
MAX_TOOL_OUTPUT_CHARS = 3000  # keep tool results small: Groq's free tier has a tokens-per-minute cap
# The ReAct loop resends the WHOLE conversation on every step, so old tool results get paid for
# again and again (measured: ~18K tokens per team question, which kept tripping the per-minute
# limit and pushing answers onto smaller backup models). Only the latest few results are sent in
# full; older ones are cut to a short preview. The full text stays in the agent's state, so the
# Explainer, the Critic and the forced answer still see all the evidence.
KEEP_FULL_TOOL_RESULTS = 2
TRIMMED_TOOL_RESULT_CHARS = 400

# --- Multi-agent team settings ---
MAX_ATTEMPTS = 2          # research attempts per question (1 first try + 1 retry after the Critic's feedback)
PASS_SCORE = 4            # the Critic gives 1-5; this score or higher = answer accepted
MAX_EVIDENCE_CHARS = 9000 # how much code the Explainer/Critic get to see (keeps us under free-tier token limits)

# --- Self-improvement (lessons memory) ---
LESSON_TOP_K = 3            # how many past lessons to look at per question
MAX_LESSON_DISTANCE = 0.75  # cosine distance: smaller = more similar. Tune this in config.py if recall feels off.

# --- Public-server safety limits (all overridable from .env / Cloud Run env vars) ---
def _int_env(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except ValueError:
        return default

MAX_REPO_MB = _int_env("MAX_REPO_MB", 50)           # refuse repos bigger than this (download size)
MAX_REPO_CHUNKS = _int_env("MAX_REPO_CHUNKS", 3000)  # embedding is the slow, CPU-heavy part: cap it
CLONE_TIMEOUT_S = _int_env("CLONE_TIMEOUT_S", 90)    # a clone taking longer than this is given up
MAX_QUESTION_CHARS = _int_env("MAX_QUESTION_CHARS", 1000)

# Rate limits PER VISITOR (per IP address). They live in memory, which is fine ONLY because the
# app runs as a single instance (--max-instances=1). They reset when the instance restarts.
ASK_PER_MINUTE = _int_env("ASK_PER_MINUTE", 3)
ASK_PER_DAY = _int_env("ASK_PER_DAY", 15)
ADD_REPO_PER_HOUR = _int_env("ADD_REPO_PER_HOUR", 3)
ADD_REPO_PER_DAY = _int_env("ADD_REPO_PER_DAY", 6)
FEEDBACK_PER_MINUTE = _int_env("FEEDBACK_PER_MINUTE", 10)
# How many proxies in front of the app add themselves to X-Forwarded-For. Cloud Run's front end
# appends the real visitor IP as the LAST entry, so 1 = "use the last entry". Anything to the left
# of it was written by the visitor and can be faked, so it is never trusted.
TRUSTED_PROXY_HOPS = _int_env("TRUSTED_PROXY_HOPS", 1)
