"""Minimal propagate stub for local OpenTelemetry imports."""

from opentelemetry.context import Context


def extract(_carrier):
    return Context()

