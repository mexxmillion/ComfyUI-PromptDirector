from pathlib import Path

import json

from runtime import (
    _chat_payload,
    build_command,
    endpoint_chat_url,
    resolve_openrouter_key,
    run_remote_endpoint,
)


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


def test_endpoint_chat_url_accepts_base_or_full_path():
    assert endpoint_chat_url("http://127.0.0.1:1234/v1") == "http://127.0.0.1:1234/v1/chat/completions"
    full = "http://127.0.0.1:1234/v1/chat/completions"
    assert endpoint_chat_url(full) == full


def test_openrouter_key_uses_environment(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "from-env")
    assert resolve_openrouter_key("") == "from-env"
    assert resolve_openrouter_key("from-node") == "from-node"


def test_chat_payload_keeps_media_tags_and_default_reasoning(monkeypatch):
    monkeypatch.setattr("runtime._image_data_url", lambda value: "data:image/jpeg;base64,image")
    monkeypatch.setattr("runtime._video_data_url", lambda value: "data:video/mp4;base64,video")
    monkeypatch.setattr("runtime._audio_part", lambda value: {"type": "input_audio", "input_audio": {"data": "audio", "format": "wav"}})

    class Tensor:
        ndim = 4

        def __getitem__(self, index):
            return "frame"

    payload = json.loads(_chat_payload(
        "google/gemini-3-flash-preview", "system", "user", [Tensor()], 100, 0.4, 1,
        videos=[object()], audios=[object()], reasoning_effort="medium", native_video=True,
    ))
    content = payload["messages"][1]["content"]
    assert payload["reasoning"] == {"effort": "medium"}
    assert [part.get("text") for part in content if part["type"] == "text"] == [
        "user", "image_reference <Picture 1>:", "video_reference <Video 1>:", "audio_reference <Audio 1>:",
    ]
    assert [part["type"] for part in content] == [
        "text", "text", "image_url", "text", "video_url", "text", "input_audio",
    ]


def test_chat_payload_can_leave_completion_uncapped_for_openrouter():
    payload = json.loads(_chat_payload(
        "google/gemini-3-flash-preview", "system", "user", [], None, 0.4, 1,
        reasoning_effort="medium", disable_thinking=False,
    ))
    assert "max_tokens" not in payload


def test_remote_endpoint_sends_native_media_without_llama_template_options(monkeypatch):
    calls = []

    def complete(*args, **kwargs):
        calls.append((args, kwargs))
        return "finished prompt"

    monkeypatch.setattr("runtime.request_completion_interruptible", complete)
    outputs, lifecycle = run_remote_endpoint(
        chat_url="http://127.0.0.1:1234/v1/chat/completions",
        api_key="local-key",
        model="vision-model",
        system_prompt="system",
        user_prompt="user",
        images=["image"],
        videos=["video"],
        audios=["audio"],
        max_tokens=100,
        temperature=0.4,
        seed=1,
        max_attempts=2,
        validator=lambda value: [],
    )
    assert outputs == ["finished prompt"]
    assert "Endpoint request completed" in lifecycle
    assert calls[0][1]["native_video"] is True
    assert calls[0][1]["disable_thinking"] is False
    assert calls[0][1]["videos"] == ["video"]
    assert calls[0][1]["audios"] == ["audio"]


def test_custom_endpoint_can_disable_hidden_thinking(monkeypatch):
    calls = []
    monkeypatch.setattr("runtime.request_completion_interruptible", lambda *args, **kwargs: (calls.append(kwargs), "done")[1])
    run_remote_endpoint(
        chat_url="http://127.0.0.1:8080/v1/chat/completions", api_key="", model="model",
        system_prompt="system", user_prompt="user", images=[], videos=[], audios=[],
        max_tokens=1500, temperature=0.4, seed=1, max_attempts=1,
        validator=lambda value: [], disable_thinking=True,
    )
    assert calls[0]["disable_thinking"] is True


def test_remote_endpoint_uses_portable_signed_32_bit_seeds(monkeypatch):
    calls = []

    def complete(*args, **kwargs):
        calls.append(args)
        return "done"

    monkeypatch.setattr("runtime.request_completion_interruptible", complete)
    run_remote_endpoint(
        chat_url="https://openrouter.ai/api/v1/chat/completions", api_key="key", model="gemini",
        system_prompt="system", user_prompt="user", images=[], videos=[], audios=[],
        max_tokens=None, temperature=0.4, seed=3493760752, max_attempts=2,
        validator=lambda value: ["retry"] if len(calls) == 1 else [],
    )
    assert [call[7] for call in calls] == [1346277104, 1346277105]
    assert all(0 <= call[7] <= 2147483647 for call in calls)
