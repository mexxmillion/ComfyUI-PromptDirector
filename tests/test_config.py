import json

from config import load_config, profile_options, resolve_profile


def test_external_catalog_resolves_paths_from_catalog_root(tmp_path):
    root = tmp_path / "llm"
    configs = root / "configs"
    configs.mkdir(parents=True)
    catalog = configs / "profiles.json"
    catalog.write_text(
        json.dumps(
            {
                "profiles": [
                    {
                        "id": "local",
                        "label": "Local Qwen",
                        "model": "models/qwen.gguf",
                        "mmproj": "models/mmproj.gguf",
                        "alias": "qwen",
                        "kv": "q4_0",
                    },
                    {"id": "remote", "remote": True, "model": "ignored.gguf"},
                ]
            }
        ),
        encoding="utf-8",
    )
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "launcher_catalog": str(catalog),
                "llama_server_path": "llama-server",
                "default_profile": "local",
                "extra_server_args": ["--jinja"],
            }
        ),
        encoding="utf-8",
    )

    config = load_config(config_path)
    labels, mapping = profile_options(config)

    assert labels == ["Local Qwen"]
    assert mapping["Local Qwen"].model_path == root / "models/qwen.gguf"
    assert resolve_profile("local", config).mmproj_path == root / "models/mmproj.gguf"
    assert resolve_profile("local", config).extra_server_args == ("--jinja",)


def test_missing_config_has_actionable_placeholder(tmp_path):
    config = load_config(tmp_path / "missing.json")
    labels, mapping = profile_options(config)
    assert labels == ["Configure config.json"]
    assert mapping == {}
