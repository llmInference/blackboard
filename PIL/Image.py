"""Minimal PIL.Image stub for AutoGen source imports."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path


class Image:
    """Tiny stand-in for PIL.Image.Image used only for import-time compatibility."""

    def __init__(self, payload: bytes | None = None):
        self._payload = payload or b""

    def convert(self, _mode: str) -> "Image":
        return self

    def save(self, fp, format: str | None = None) -> None:  # noqa: A002
        del format
        data = self._payload or b""
        if hasattr(fp, "write"):
            fp.write(data)


def open(source) -> Image:  # noqa: A001
    if isinstance(source, (str, Path)):
        return Image()
    if isinstance(source, BytesIO):
        return Image(source.getvalue())
    if hasattr(source, "read"):
        return Image(source.read())
    return Image()
