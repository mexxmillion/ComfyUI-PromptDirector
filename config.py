from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "config.json"


@dataclass(frozen=True)
class LlamaProfile:
    profile_id: str
    label: str
    model_path: Path
    mmproj_path: Path | None
    alias: str
    kv_type: str
    extra_server_args: tuple[str, ...]


@dataclass(frozen=True)
class DirectorConfig:
    server_path: str
    default_profile: str
    profiles: tuple[LlamaProfile, ...]


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return data


def _resolve_path(value: str, base: Path) -> Path:
    path = Path(os.path.expandvars(value)).expanduser()
    return path if path.is_absolute() else base / path


def _catalog_profiles(path: Path) -> list[tuple[dict, Path]]:
    data = _read_json(path)
    base = path.parent.parent if path.parent.name.casefold() == "configs" else path.parent
    return [(entry, base) for entry in data.get("profiles", []) if not entry.get("remote")]


def load_config(path: Path = CONFIG_PATH) -> DirectorConfig:
    if not path.is_file():
        return DirectorConfig("llama-server", "", ())

    data = _read_json(path)
    entries: list[tuple[dict, Path]] = []
    catalog_value = str(data.get("launcher_catalog", "")).strip()
    if catalog_value:
        catalog_path = _resolve_path(catalog_value, path.parent)
        entries.extend(_catalog_profiles(catalog_path))
    entries.extend((entry, path.parent) for entry in data.get("profiles", []))

    common_args = tuple(str(arg) for arg in data.get("extra_server_args", []))
    profiles: list[LlamaProfile] = []
    seen: set[str] = set()
    for entry, base in entries:
        profile_id = str(entry.get("id", "")).strip()
        model = str(entry.get("model", "")).strip()
        if not profile_id or not model or profile_id in seen:
            continue
        seen.add(profile_id)
        mmproj = str(entry.get("mmproj", "")).strip()
        profile_args = tuple(str(arg) for arg in entry.get("extra_server_args", []))
        profiles.append(
            LlamaProfile(
                profile_id=profile_id,
                label=str(entry.get("label") or profile_id),
                model_path=_resolve_path(model, base),
                mmproj_path=_resolve_path(mmproj, base) if mmproj else None,
                alias=str(entry.get("alias") or profile_id),
                kv_type=str(entry.get("kv") or "q8_0"),
                extra_server_args=common_args + profile_args,
            )
        )

    return DirectorConfig(
        server_path=str(data.get("llama_server_path") or "llama-server"),
        default_profile=str(data.get("default_profile") or ""),
        profiles=tuple(profiles),
    )


def profile_options(config: DirectorConfig | None = None) -> tuple[list[str], dict[str, LlamaProfile]]:
    config = config or load_config()
    mapping = {profile.label: profile for profile in config.profiles}
    return list(mapping) or ["Configure config.json"], mapping


def resolve_profile(selected: str, config: DirectorConfig | None = None) -> LlamaProfile:
    config = config or load_config()
    for profile in config.profiles:
        if selected in (profile.label, profile.profile_id):
            return profile
    raise ValueError("No local LLM profile is configured. Copy config.example.json to config.json and edit its paths.")
