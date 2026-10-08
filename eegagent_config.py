"""Explicit project API settings; never read another application's credentials."""
import os
from pathlib import Path


def api_settings(prefix, config_file=".env"):
    """Read KEY=value settings; environment variables override the optional file."""
    values = {}
    if config_file is not None and Path(config_file).is_file():
        for line in Path(config_file).read_text(encoding="utf-8-sig").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                raise ValueError("API configuration requires KEY=value lines")
            name, value = line.split("=", 1)
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
                value = value[1:-1]
            values[name.strip()] = value
    keys = {field: prefix + suffix for field, suffix in
            (("base_url", "_BASE_URL"), ("model", "_MODEL"), ("key", "_KEY"))}
    settings = {field: os.environ.get(name, values.get(name, "")).strip()
                for field, name in keys.items()}
    missing = [keys[field] for field, value in settings.items() if not value]
    if missing:
        raise ValueError("Configure these API settings: " + ", ".join(missing))
    settings["base_url"] = settings["base_url"].rstrip("/")
    return settings
