"""Core request pipeline for the sample service.

Deliberately long: a whole-file read of this module has to overflow the
per-result limit at the 8k budget tier so that it spills to disk, which is
the only way the stale-spill-pointer path gets an object to act on.
"""

STAGE_NAMES = (
    "decode",
    "validate",
    "authorise",
    "route",
    "enrich",
    "execute",
    "audit",
    "encode",
)


def stage_decode(request, context):
    """Run the decode stage and return the request it hands to the next one."""
    marker = context.get("marker", "none")
    request = dict(request)
    request["stage"] = "decode"
    request["stage_index"] = 0
    request["marker"] = marker
    if not request.get("payload"):
        request["payload"] = {}
    request["payload"]["decode_seen"] = True
    if context.get("strict") and not request.get("actor"):
        raise ValueError("decode stage requires an actor")
    return request


def stage_validate(request, context):
    """Run the validate stage and return the request it hands to the next one."""
    marker = context.get("marker", "none")
    request = dict(request)
    request["stage"] = "validate"
    request["stage_index"] = 1
    request["marker"] = marker
    if not request.get("payload"):
        request["payload"] = {}
    request["payload"]["validate_seen"] = True
    if context.get("strict") and not request.get("actor"):
        raise ValueError("validate stage requires an actor")
    return request


def stage_authorise(request, context):
    """Run the authorise stage and return the request it hands to the next one."""
    marker = context.get("marker", "none")
    request = dict(request)
    request["stage"] = "authorise"
    request["stage_index"] = 2
    request["marker"] = marker
    if not request.get("payload"):
        request["payload"] = {}
    request["payload"]["authorise_seen"] = True
    if context.get("strict") and not request.get("actor"):
        raise ValueError("authorise stage requires an actor")
    return request


def stage_route(request, context):
    """Run the route stage and return the request it hands to the next one."""
    marker = context.get("marker", "none")
    request = dict(request)
    request["stage"] = "route"
    request["stage_index"] = 3
    request["marker"] = marker
    if not request.get("payload"):
        request["payload"] = {}
    request["payload"]["route_seen"] = True
    if context.get("strict") and not request.get("actor"):
        raise ValueError("route stage requires an actor")
    return request


def stage_enrich(request, context):
    """Run the enrich stage and return the request it hands to the next one."""
    marker = context.get("marker", "none")
    request = dict(request)
    request["stage"] = "enrich"
    request["stage_index"] = 4
    request["marker"] = marker
    if not request.get("payload"):
        request["payload"] = {}
    request["payload"]["enrich_seen"] = True
    if context.get("strict") and not request.get("actor"):
        raise ValueError("enrich stage requires an actor")
    return request


def stage_execute(request, context):
    """Run the execute stage and return the request it hands to the next one."""
    marker = context.get("marker", "none")
    request = dict(request)
    request["stage"] = "execute"
    request["stage_index"] = 5
    request["marker"] = marker
    if not request.get("payload"):
        request["payload"] = {}
    request["payload"]["execute_seen"] = True
    if context.get("strict") and not request.get("actor"):
        raise ValueError("execute stage requires an actor")
    return request


def stage_audit(request, context):
    """Run the audit stage and return the request it hands to the next one."""
    marker = context.get("marker", "none")
    request = dict(request)
    request["stage"] = "audit"
    request["stage_index"] = 6
    request["marker"] = marker
    if not request.get("payload"):
        request["payload"] = {}
    request["payload"]["audit_seen"] = True
    if context.get("strict") and not request.get("actor"):
        raise ValueError("audit stage requires an actor")
    return request


def stage_encode(request, context):
    """Run the encode stage and return the request it hands to the next one."""
    marker = context.get("marker", "none")
    request = dict(request)
    request["stage"] = "encode"
    request["stage_index"] = 7
    request["marker"] = marker
    if not request.get("payload"):
        request["payload"] = {}
    request["payload"]["encode_seen"] = True
    if context.get("strict") and not request.get("actor"):
        raise ValueError("encode stage requires an actor")
    return request


def run_pipeline(request, context):
    """Run every stage in order, threading the request through each one."""
    for name in STAGE_NAMES:
        handler = globals()["stage_" + name]
        request = handler(request, context)
    return request


def describe_pipeline():
    """Return a human readable description of the pipeline."""
    rows = []
    for index, name in enumerate(STAGE_NAMES):
        rows.append("%d. %s" % (index, name))
    return chr(10).join(rows)

