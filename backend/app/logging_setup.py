"""Structured (JSON) logging for CodeAtlas, so another tool (e.g. PagePal) can read and analyze it.

Why JSON? A plain print() line is just text: "rate limit hit; waiting 20 seconds". A JSON line is
data: {"message": "rate_limited", "model": "openai/gpt-oss-120b", "wait_s": 20, "severity": "WARNING"}.
A program can filter it ("all WARNINGs for this model") without guessing at the wording.

Where do logs go? To STDERR, never stdout. The MCP server (mcp_server.py) uses stdout to talk to
its client, so a log line on stdout would break it. Cloud Run collects stderr just the same.

Cloud Logging understands a few special field names, so we use them:
  severity                          -> the log level (INFO, WARNING, ERROR...)
  message                           -> the short event name shown in the log list
  logging.googleapis.com/trace      -> links this log line to its trace in Cloud Trace
  logging.googleapis.com/spanId     -> ...and to the exact step (span) inside that trace

Settings (environment variables, all optional):
  LOG_FORMAT=json|console   default: json on Cloud Run, readable console text on your laptop
  LOG_LEVEL=DEBUG|INFO|...  default: INFO
  GOOGLE_CLOUD_PROJECT      your GCP project id; needed to build the trace link field

Use it like this:
    from ..logging_setup import get_logger
    log = get_logger(__name__)
    log.info("indexing_done", repo="requests", chunks=1071, seconds=42.5)
"""
import contextvars
import logging
import os
import sys

import structlog

# Cloud Run sends every request with an "X-Cloud-Trace-Context: TRACE_ID/SPAN_ID;o=1" header.
# If OpenTelemetry isn't switched on, we still use this so log lines can be grouped per request.
_cloud_trace_header: contextvars.ContextVar[str | None] = contextvars.ContextVar("cloud_trace_header", default=None)

_configured = False
_project_id: str | None = None


def _on_cloud_run() -> bool:
    return bool(os.getenv("K_SERVICE"))  # Cloud Run always sets this


def _find_project_id() -> str | None:
    project = os.getenv("GOOGLE_CLOUD_PROJECT", "").strip()
    if project or not _on_cloud_run():
        return project or None
    try:  # on Cloud Run, ask Google's metadata server (google-auth comes with the trace exporter)
        import google.auth
        return google.auth.default()[1]
    except Exception:  # noqa: BLE001  logging must never crash the app
        return None


def _current_trace() -> tuple[str, str, bool] | None:
    """(trace_id, span_id, sampled) for the request being handled right now, or None."""
    try:
        from opentelemetry import trace
        ctx = trace.get_current_span().get_span_context()
        if ctx.is_valid:
            return format(ctx.trace_id, "032x"), format(ctx.span_id, "016x"), ctx.trace_flags.sampled
    except Exception:  # noqa: BLE001  OpenTelemetry not installed or not set up - fall back below
        pass
    header = _cloud_trace_header.get()
    if header:
        trace_id, _, rest = header.partition("/")
        span_id = rest.split(";")[0]
        if span_id.isdigit():  # this header gives the span id in decimal; Cloud Logging wants 16 hex digits
            span_id = format(int(span_id), "016x")
        return trace_id, span_id, "o=1" in rest
    return None


def _add_trace(_, __, event_dict: dict) -> dict:
    found = _current_trace()
    if found:
        trace_id, span_id, sampled = found
        event_dict["trace_id"] = trace_id  # plain copy, handy for PagePal and for local runs
        if _project_id:
            event_dict["logging.googleapis.com/trace"] = f"projects/{_project_id}/traces/{trace_id}"
            event_dict["logging.googleapis.com/spanId"] = span_id
            event_dict["logging.googleapis.com/trace_sampled"] = sampled
    return event_dict


def _add_severity(_, method_name: str, event_dict: dict) -> dict:
    """Cloud Logging wants `severity` in capitals. log.exception() counts as ERROR."""
    level = event_dict.pop("level", method_name)
    event_dict["severity"] = "ERROR" if level == "exception" else level.upper()
    return event_dict


def setup_logging() -> None:
    """Safe to call many times; only the first call does anything."""
    global _configured, _project_id
    if _configured:
        return
    _configured = True
    _project_id = _find_project_id()

    fmt = os.getenv("LOG_FORMAT", "json" if _on_cloud_run() else "console").lower()
    level = getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO)

    processors = [
        structlog.contextvars.merge_contextvars,  # adds request_id, repo... bound for this request
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True, key="time"),
        _add_trace,
    ]
    if fmt == "json":
        processors += [
            _add_severity,
            structlog.processors.format_exc_info,  # the traceback becomes an "exception" text field
            structlog.processors.EventRenamer("message"),
            structlog.processors.JSONRenderer(),
        ]
    else:
        processors.append(structlog.dev.ConsoleRenderer(colors=False))

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(level),
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str):
    setup_logging()
    return structlog.get_logger().bind(logger=name)


def bind_request(request_id: str, cloud_trace_header: str | None = None) -> None:
    """Called once per HTTP request (see main.py). Every log line written while handling that
    request - in any file, even in worker threads - then carries the same request_id."""
    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(request_id=request_id)
    _cloud_trace_header.set(cloud_trace_header)
