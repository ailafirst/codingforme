# SPDX: internal
"""Encode and decode the sample wire format.

The format is deliberately simple: fields joined by a pipe, with a short
header naming the schema revision so old readers can refuse new payloads.
"""

HEADER = "v1"
SEPARATOR = "|"


def encode(payload):
    body = SEPARATOR.join(str(item) for item in payload)
    return HEADER + SEPARATOR + body


def decode(blob):
    parts = blob.split(SEPARATOR)
    if not parts or parts[0] != HEADER:
        return []
    return parts[1:]
