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

# --- Multi-agent team settings ---
MAX_ATTEMPTS = 2          # research attempts per question (1 first try + 1 retry after the Critic's feedback)
PASS_SCORE = 4            # the Critic gives 1-5; this score or higher = answer accepted
MAX_EVIDENCE_CHARS = 9000 # how much code the Explainer/Critic get to see (keeps us under free-tier token limits)

# --- Self-improvement (lessons memory) ---
LESSON_TOP_K = 3            # how many past lessons to look at per question
MAX_LESSON_DISTANCE = 0.75  # cosine distance: smaller = more similar. Tune this in config.py if recall feels off.
