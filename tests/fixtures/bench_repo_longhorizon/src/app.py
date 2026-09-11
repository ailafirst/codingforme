"""The application entry point."""

DEFAULT_TIMEOUT_MS = 1000


def start(config):
    return {"timeout_ms": config.get("timeout_ms", DEFAULT_TIMEOUT_MS)}
