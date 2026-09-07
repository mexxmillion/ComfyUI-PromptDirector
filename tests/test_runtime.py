from pathlib import Path

from runtime import build_command


def test_build_command_adds_projector_only_when_present():
    without = build_command("llama-server", Path("model.gguf"), "qwen", 50000, 4096, "q8_0", None)
    with_vision = build_command(
        "llama-server",
        Path("model.gguf"),
        "qwen",
        50000,
        8192,
        "q4_0",
        Path("mmproj.gguf"),
        ("--jinja",),
    )
    assert "--mmproj" not in without
    assert with_vision[-3:] == ["--mmproj", "mmproj.gguf", "--jinja"]
    assert with_vision[with_vision.index("-c") + 1] == "8192"
