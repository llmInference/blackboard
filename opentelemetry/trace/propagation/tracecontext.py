"""Minimal tracecontext propagator stub."""


class TraceContextTextMapPropagator:
    """Tiny stand-in for TraceContextTextMapPropagator."""

    def inject(self, carrier):
        carrier.setdefault("traceparent", "00-00000000000000000000000000000000-0000000000000000-00")

