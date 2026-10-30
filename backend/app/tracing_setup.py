"""Tracing with OpenTelemetry, sent to Google Cloud Trace (free up to 2.5 million spans a month).

A TRACE is the full story of one request. It is made of SPANS, one per step, each with a start
time and a duration, e.g.:
    GET /api/ask                    48.2s
      llm_call openai/gpt-oss-120b   3.1s
      llm_call openai/gpt-oss-120b   4.7s
      ...
So you can see WHERE the time went, and which step failed.

Off by default (your laptop has no Cloud Trace). Switch it on with CLOUD_TRACE_ENABLED=true.
Optional: CLOUD_TRACE_SAMPLE_RATE=0.1 keeps only 10% of traces (useful during load tests).

This is set up in code, not with the `opentelemetry-instrument` command, so it stays off locally
and nothing in the Dockerfile has to change.

LangSmith is separate and needs no code at all: LangGraph sends its own traces there when
LANGSMITH_TRACING=true and LANGSMITH_API_KEY are set (see the README).
"""
import os

from .logging_setup import get_logger

log = get_logger(__name__)


def setup_tracing(app) -> None:
    if os.getenv("CLOUD_TRACE_ENABLED", "").lower() not in ("1", "true", "yes"):
        return
    try:
        from opentelemetry import propagate, trace
        from opentelemetry.exporter.cloud_trace import CloudTraceSpanExporter
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
        from opentelemetry.propagate import composite
        from opentelemetry.propagators.cloud_trace_propagator import CloudTraceFormatPropagator
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from opentelemetry.sdk.trace.sampling import TraceIdRatioBased
        from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

        rate = float(os.getenv("CLOUD_TRACE_SAMPLE_RATE", "1.0"))
        provider = TracerProvider(
            resource=Resource.create({"service.name": os.getenv("K_SERVICE", "codeatlas")}),
            # Not "parent based" on purpose: Cloud Run itself only samples a few requests, and we
            # want to decide for ourselves how many get traced.
            sampler=TraceIdRatioBased(rate),
        )
        provider.add_span_processor(BatchSpanProcessor(CloudTraceSpanExporter()))
        trace.set_tracer_provider(provider)
        # Read Cloud Run's X-Cloud-Trace-Context header, so our spans join the request's trace.
        propagate.set_global_textmap(composite.CompositePropagator(
            [CloudTraceFormatPropagator(), TraceContextTextMapPropagator()]))
        # The UI polls index-status every second while indexing - not worth a trace each time.
        FastAPIInstrumentor.instrument_app(app, excluded_urls="index-status")
        log.info("tracing_enabled", exporter="cloud_trace", sample_rate=rate)
    except Exception:  # noqa: BLE001  tracing is a nice-to-have; the app must still start
        log.exception("tracing_setup_failed")
