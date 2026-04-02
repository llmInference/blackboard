"""Minimal trace stub for local OpenTelemetry imports."""

from __future__ import annotations

from contextlib import contextmanager


class SpanContext:
    """Tiny stand-in for an OpenTelemetry SpanContext."""


class Span:
    """Tiny no-op span."""

    def get_span_context(self) -> SpanContext:
        return SpanContext()

    def record_exception(self, _exc: Exception) -> None:
        return None

    def set_status(self, _status) -> None:
        return None

    def set_attribute(self, _key, _value) -> None:
        return None


class Link:
    """Tiny stand-in for an OpenTelemetry Link."""

    def __init__(self, context: SpanContext):
        self.context = context


class SpanKind:
    PRODUCER = "producer"
    CONSUMER = "consumer"
    CLIENT = "client"
    INTERNAL = "internal"


class _Tracer:
    @contextmanager
    def start_as_current_span(
        self,
        name,
        context=None,
        kind=None,
        attributes=None,
        links=None,
        start_time=None,
        record_exception=True,
        set_status_on_exception=True,
        end_on_exit=True,
    ):
        del (
            name,
            context,
            kind,
            attributes,
            links,
            start_time,
            record_exception,
            set_status_on_exception,
            end_on_exit,
        )
        yield Span()


Tracer = _Tracer


class StatusCode:
    ERROR = "error"


class Status:
    def __init__(self, status_code, description: str | None = None):
        self.status_code = status_code
        self.description = description


class TracerProvider:
    """Tiny tracer provider."""

    def get_tracer(self, _name: str):
        return _Tracer()


class NoOpTracerProvider(TracerProvider):
    """No-op tracer provider."""


_GLOBAL_TRACER_PROVIDER = NoOpTracerProvider()


def get_tracer_provider():
    return _GLOBAL_TRACER_PROVIDER


def get_current_span(_context=None):
    return Span()


def get_tracer(_name: str):
    return _Tracer()


def set_span_in_context(span, context=None):
    del span, context
    return None
