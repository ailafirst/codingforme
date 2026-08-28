"""Project-local configuration helpers."""

import os
import re
from pathlib import Path


ENV_KEY_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _strip_quotes(value):
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def _parse_env_line(line):
    line = line.strip()
    if not line or line.startswith("#"):
        return None
    if line.startswith("export "):
        line = line[len("export "):].strip()
    if "=" not in line:
        raise ValueError(f"invalid .env line: {line}")
    name, value = line.split("=", 1)
    name = name.strip()
    if not ENV_KEY_PATTERN.match(name):
        raise ValueError(f"invalid .env variable name: {name}")
    return name, _strip_quotes(value)


def project_root():
    """codingforme 自身所在的仓库根目录（`codingforme/` 包的上一级）。

    `.env` 一律从这里找，**不看工作区、也不看进程 cwd**。原因是踩过坑：
    此前 cli 用的是 `load_project_env(workspace.repo_root)`，而工作区经常在仓库
    外面（拿 agent 去改别的项目就是这种情况），往上走永远找不到本仓库的 `.env`，
    实测载入 0 个键。它看上去能用，只是因为 `import litellm` 会顺手 `load_dotenv()`
    从进程 cwd 读一遍——配置实际是被第三方的副作用喂进来的。于是从仓库根目录以外
    的地方启动就报缺凭证，而且哪天 litellm 去掉那行，`.env` 就彻底读不到了。
    """
    return Path(__file__).resolve().parent.parent


def find_project_env(start):
    current = Path(start).resolve()
    if current.is_file():
        current = current.parent
    for path in (current, *current.parents):
        env_path = path / ".env"
        if env_path.exists():
            return env_path
    return None


def load_project_env(start, override=True):
    env_path = find_project_env(start)
    if env_path is None:
        return {}
    loaded = {}
    for line in env_path.read_text(encoding="utf-8-sig").splitlines():
        parsed = _parse_env_line(line)
        if parsed is None:
            continue
        name, value = parsed
        loaded[name] = value
        if override or name not in os.environ:
            os.environ[name] = value
    return loaded


def provider_env(name, legacy_names=(), default=""):
    for env_name in (name, *legacy_names):
        value = os.environ.get(env_name)
        if value:
            return value
    return default
