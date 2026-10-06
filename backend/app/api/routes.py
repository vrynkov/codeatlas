"""The HTTP API the React frontend talks to. Thin wrappers around the same functions the
CLI, the eval scripts, and the MCP server already use - no logic lives here twice."""
import json
import sys
import uuid

import anyio
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from ..agents.team import run_team, submit_feedback
from ..indexing.indexer import clone_repo, index_repo
from ..mcp_server import _indexed_repos  # the same "which repos exist" logic, reused here

router = APIRouter(prefix="/api")

# Explicit stop signals for in-progress /ask streams, keyed by a request_id the frontend
# makes up per question. This is deliberately NOT based on detecting a dropped connection:
# that depends on network/browser timing and is hard to make reliable or testable. A user
# clicking "Stop" sets a flag here directly, and the stream checks it between agent steps.
_stop_flags: dict[str, bool] = {}


class AddRepoBody(BaseModel):
    url: str


class FeedbackBody(BaseModel):
    repo: str
    question: str
    files: list[str] = []
    helpful: bool
    note: str = ""


class StopBody(BaseModel):
    request_id: str


@router.get("/repos")
def list_repos() -> list[dict]:
    from ..indexing.indexer import repo_source_commit, repo_source_url
    return [{"name": name, "url": repo_source_url(name), "commit": repo_source_commit(name)}
            for name in _indexed_repos()]


@router.post("/repos")
async def add_repo(body: AddRepoBody) -> dict:
    """Clones + indexes a repo. Runs the (slow, CPU-bound) indexer in a worker thread so it
    doesn't block other requests while it works."""
    try:
        name, _path = await anyio.to_thread.run_sync(clone_repo, body.url)  # cheap: no-ops if already cloned
        chunks = await anyio.to_thread.run_sync(index_repo, body.url)
        from ..indexing.indexer import repo_source_commit
        commit = await anyio.to_thread.run_sync(repo_source_commit, name)
        return {"repo": name, "chunks": chunks, "commit": commit}
    except Exception as e:  # noqa: BLE001
        raise HTTPException(400, f"Could not index that repo: {e}") from e


@router.get("/ask")
async def ask(repo: str, question: str, request_id: str | None = None) -> StreamingResponse:
    """Server-Sent Events: streams each agent's progress live, then the final answer.
    A GET (not POST) so the browser's built-in EventSource can be used as-is on the frontend.
    `request_id` is optional (older/simpler callers still work) but is what lets POST /ask/stop
    cancel this specific stream."""
    if repo not in _indexed_repos():
        raise HTTPException(404, f"Unknown repo '{repo}'.")

    rid = request_id or str(uuid.uuid4())
    _stop_flags[rid] = False

    async def event_stream():
        events = run_team(repo, question)
        done = object()
        try:
            while True:
                if _stop_flags.get(rid):
                    yield f"data: {json.dumps({'agent': 'team', 'type': 'stopped', 'text': 'Stopped.'})}\n\n"
                    break
                # Honest limitation: a single agent step already in flight (one Groq call)
                # can't be interrupted mid-call - Python can't forcibly stop a running thread.
                # We check the stop flag BETWEEN steps, so no further work starts once the
                # user has clicked Stop, but the current step finishes on its own first.
                ev = await anyio.to_thread.run_sync(next, events, done)
                if ev is done:
                    break
                yield f"data: {json.dumps(ev)}\n\n"
        except Exception as e:  # noqa: BLE001  surface the error to the browser instead of hanging
            # str(e) alone can be a generic, unhelpful single line (e.g. openai/groq's
            # APIConnectionError defaults to just "Connection error." with no further detail) -
            # log the real exception TYPE and full traceback server-side so a failure is
            # actually diagnosable from Cloud Run's logs, not just visible as that one vague
            # line in the browser.
            import traceback
            print(f"[ERROR] {type(e).__name__}: {e}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            yield f"data: {json.dumps({'agent': 'error', 'type': 'error', 'text': str(e)})}\n\n"
        finally:
            events.close()  # let the underlying agent graph clean up promptly
            _stop_flags.pop(rid, None)

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
def feedback(body: FeedbackBody) -> dict:
    msg = submit_feedback(body.repo, body.question, body.files, body.helpful, body.note)
    return {"message": msg}
