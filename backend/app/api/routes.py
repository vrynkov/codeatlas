"""The HTTP API the React frontend talks to. Thin wrappers around the same functions the
CLI, the eval scripts, and the MCP server already use - no logic lives here twice."""
import json
import threading
import time
import uuid
from collections import OrderedDict

import anyio
from fastapi import APIRouter, BackgroundTasks, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from ..agents.team import cited_files, run_team, submit_feedback
from ..config import (ADD_REPO_PER_DAY, ADD_REPO_PER_HOUR, ASK_PER_DAY, ASK_PER_MINUTE,
                      FEEDBACK_PER_MINUTE, MAX_QUESTION_CHARS)
from ..indexing.indexer import CloneError, IndexBusyError, get_progress, start_indexing_job
from ..logging_setup import get_logger
from ..mcp_server import _indexed_repos  # the same "which repos exist" logic, reused here
from .limiter import enforce

router = APIRouter(prefix="/api")
log = get_logger(__name__)

# Explicit stop signals for in-progress /ask streams, keyed by a request_id the frontend
# makes up per question. This is deliberately NOT based on detecting a dropped connection:
# that depends on network/browser timing and is hard to make reliable or testable. A user
# clicking "Stop" sets a flag here directly, and the stream checks it between agent steps.
_stop_flags: dict[str, bool] = {}

# Answers this server actually gave, by request_id, so feedback can only be about a REAL answer.
# The repo, question and cited files are taken from HERE, never from the feedback request - so
# nobody can POST made-up "feedback" for a question that was never asked, with made-up files.
# Bounded (oldest dropped first) so it can't grow forever. In memory: fine for one instance.
_ANSWERS: "OrderedDict[str, dict]" = OrderedDict()
_ANSWERS_LOCK = threading.Lock()
MAX_REMEMBERED_ANSWERS = 1000


def _remember_answer(rid: str, repo: str, question: str, answer: str) -> None:
    with _ANSWERS_LOCK:
        _ANSWERS[rid] = {"repo": repo, "question": question, "files": cited_files(answer), "rated": False}
        _ANSWERS.move_to_end(rid)
        while len(_ANSWERS) > MAX_REMEMBERED_ANSWERS:
            _ANSWERS.popitem(last=False)


class AddRepoBody(BaseModel):
    url: str = Field(max_length=300)


class FeedbackBody(BaseModel):
    request_id: str = Field(max_length=64)
    helpful: bool
    note: str = Field("", max_length=1000)  # saved for the maintainer to READ; never put in a prompt


class StopBody(BaseModel):
    request_id: str = Field(max_length=64)


@router.get("/repos")
def list_repos() -> list[dict]:
    from ..indexing.indexer import repo_source_commit, repo_source_url
    return [{"name": name, "url": repo_source_url(name), "commit": repo_source_commit(name)}
            for name in _indexed_repos()]


@router.post("/repos")
async def add_repo(body: AddRepoBody, background_tasks: BackgroundTasks, request: Request) -> dict:
    """Clones, scans, and chunks a repo (all fast - no embedding model involved yet), then
    returns IMMEDIATELY with enough info for the UI to show a real "indexing..." state (file
    count, size, a rough time estimate) - rather than one long blocking request with zero
    feedback. The genuinely slow part (embedding every chunk) runs afterward in the
    background; the frontend polls GET /repos/{name}/index-status for live progress."""
    enforce(request, "add_repo", [(ADD_REPO_PER_HOUR, 3600), (ADD_REPO_PER_DAY, 86400)])
    try:
        job = await anyio.to_thread.run_sync(start_indexing_job, body.url)
    except IndexBusyError as e:
        raise HTTPException(429, str(e)) from e
    except CloneError as e:
        log.warning("add_repo_failed", url=body.url, reason=str(e))
        raise HTTPException(400, str(e)) from e
    except Exception as e:  # noqa: BLE001
        log.exception("add_repo_failed", url=body.url)
        # The details are in the server log; the visitor gets a generic message (an internal error
        # text can reveal file paths or other server details to the public).
        raise HTTPException(400, "Could not index that repo. Please check the URL and try again.") from e
    background_tasks.add_task(job["work"])
    return {
        "repo": job["name"],
        "file_count": job["file_count"],
        "total_bytes": job["total_bytes"],
        "chunks_total": job["chunks_total"],
        "estimated_seconds": job["estimated_seconds"],
    }


@router.get("/repos/{name}/index-status")
def index_status(name: str) -> dict:
    """Polled by the frontend while a repo is indexing, to show live progress - which chunk
    batch has been embedded so far, and the final commit/url once the background job finishes."""
    progress = get_progress(name)
    if progress is None:
        raise HTTPException(404, f"No indexing job found for '{name}'.")
    result = dict(progress)
    if result["stage"] == "done":
        from ..indexing.indexer import repo_source_commit, repo_source_url
        result["url"] = repo_source_url(name)
        result["commit"] = repo_source_commit(name)
    return result


@router.get("/ask")
async def ask(request: Request,
              repo: str = Query(max_length=100),
              question: str = Query(min_length=1, max_length=MAX_QUESTION_CHARS),
              request_id: str | None = Query(None, max_length=64)) -> StreamingResponse:
    """Server-Sent Events: streams each agent's progress live, then the final answer.
    A GET (not POST) so the browser's built-in EventSource can be used as-is on the frontend.
    `request_id` is optional (older/simpler callers still work) but is what lets POST /ask/stop
    cancel this specific stream."""
    if repo not in _indexed_repos():
        raise HTTPException(404, f"Unknown repo '{repo}'.")
    # Every question costs ~25K tokens of the shared free Groq quota, so one visitor must not be
    # able to burn the whole day's quota for everyone else.
    try:
        enforce(request, "ask", [(ASK_PER_MINUTE, 60), (ASK_PER_DAY, 86400)])
    except HTTPException as e:
        # The browser's EventSource can't read the text of an error response (it only sees "the
        # connection failed"), so the limit message is sent as one normal error EVENT instead -
        # the UI then shows the real reason rather than "Lost connection to the server."
        msg = json.dumps({"agent": "error", "type": "error", "text": e.detail})
        return StreamingResponse(iter([f"data: {msg}\n\n"]), media_type="text/event-stream",
                                 headers=e.headers)

    rid = request_id or str(uuid.uuid4())
    _stop_flags[rid] = False
    # Only the length is logged, not the question itself: the app is public, and the full text is
    # in the LangSmith trace anyway (found by the same request_id).
    log.info("team_run_started", repo=repo, request_id=rid, question_len=len(question))

    async def event_stream():
        events = run_team(repo, question, request_id=rid)
        done = object()
        start = time.perf_counter()
        outcome, event_count = "disconnected", 0  # stays this if the browser goes away mid-stream
        try:
            while True:
                if _stop_flags.get(rid):
                    outcome = "stopped"
                    log.info("team_run_stopped", repo=repo, request_id=rid)
                    yield f"data: {json.dumps({'agent': 'team', 'type': 'stopped', 'text': 'Stopped.'})}\n\n"
                    break
                # Honest limitation: a single agent step already in flight (one Groq call)
                # can't be interrupted mid-call - Python can't forcibly stop a running thread.
                # We check the stop flag BETWEEN steps, so no further work starts once the
                # user has clicked Stop, but the current step finishes on its own first.
                ev = await anyio.to_thread.run_sync(next, events, done)
                if ev is done:
                    outcome = "finished"
                    break
                event_count += 1
                if ev.get("agent") in ("team", "chat") and ev.get("type") == "answer":
                    _remember_answer(rid, repo, question, ev.get("text", ""))
                yield f"data: {json.dumps(ev)}\n\n"
        except Exception as e:  # noqa: BLE001  surface the error to the browser instead of hanging
            # str(e) alone can be a generic, unhelpful single line (e.g. openai/groq's
            # APIConnectionError defaults to just "Connection error." with no further detail) -
            # log the real exception TYPE and full traceback server-side so a failure is
            # actually diagnosable from Cloud Run's logs, not just visible as that one vague
            # line in the browser.
            outcome = "error"
            log.exception("team_run_failed", repo=repo, request_id=rid, question_len=len(question),
                          error_type=type(e).__name__)
            yield f"data: {json.dumps({'agent': 'error', 'type': 'error', 'text': str(e)})}\n\n"
        finally:
            events.close()  # let the underlying agent graph clean up promptly
            _stop_flags.pop(rid, None)
            log.info("team_run_finished", repo=repo, request_id=rid, outcome=outcome,
                     events=event_count, duration_ms=round((time.perf_counter() - start) * 1000))

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@router.post("/ask/stop")
def stop_ask(body: StopBody) -> dict:
    """Called by the Stop button. Idempotent: fine to call even if the stream already
    finished or the id is unknown (e.g. the user double-clicked Stop)."""
    if body.request_id in _stop_flags:
        _stop_flags[body.request_id] = True
        return {"stopped": True}
    return {"stopped": False, "reason": "already finished or unknown request_id"}


@router.post("/feedback")
def feedback(body: FeedbackBody, request: Request) -> dict:
    enforce(request, "feedback", [(FEEDBACK_PER_MINUTE, 60)])
    with _ANSWERS_LOCK:
        answer = _ANSWERS.get(body.request_id)
        if answer is None:
            raise HTTPException(404, "That answer is unknown or too old to rate.")
        if answer["rated"]:
            raise HTTPException(409, "This answer was already rated.")
        answer["rated"] = True
    msg = submit_feedback(answer["repo"], answer["question"], answer["files"], body.helpful, body.note)
    return {"message": msg}
