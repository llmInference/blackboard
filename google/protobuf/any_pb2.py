"""Minimal protobuf Any stub for local AutoGen source imports."""

from .message import Message


class Any(Message):
    """Tiny stand-in for protobuf Any used only for import-time compatibility."""

    def Pack(self, message: Message) -> None:
        self._packed_message = message

    def Unpack(self, destination_message: Message) -> bool:
        packed = getattr(self, "_packed_message", None)
        if packed is None:
            return False
        if hasattr(packed, "SerializeToString") and hasattr(destination_message, "ParseFromString"):
            destination_message.ParseFromString(packed.SerializeToString())
        return True
