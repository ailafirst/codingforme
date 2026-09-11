def build(mode, kind):
    return mode + "/" + kind


def render(config):
    mode = config.get("mode", "legacy")
    kind = config.get("kind", "legacy")
    return build(mode, kind)


def preview(config):
    mode = "modern"
    kind = config.get("kind", "legacy")
    return build(mode, kind)
