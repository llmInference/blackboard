"""Minimal protobuf.message stub for local AutoGen source imports."""


class _Descriptor:
    full_name = "google.protobuf.Message"


class Message:
    """Tiny stand-in for protobuf Message used only for import-time compatibility."""

    DESCRIPTOR = _Descriptor()

    def ParseFromString(self, payload: bytes) -> None:
        self._payload = payload

    def SerializeToString(self) -> bytes:
        return getattr(self, "_payload", b"")

