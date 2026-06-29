"""Lazy package exports for the POLT runtime."""

__all__ = ["POLT"]


def __getattr__(name):
    """Load heavyweight runtime objects only when explicitly requested."""
    if name == "POLT":
        from .runtime import POLT

        return POLT
    raise AttributeError(name)
