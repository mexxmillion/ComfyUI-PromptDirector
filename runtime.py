from __future__ import annotations

import base64
import gc
import io
import json
import logging
import os
import random
import shutil
import socket
import subprocess
import sys
import time
import wave
from collections import deque
from pathlib import Path
from threading import Thread
from typing import Callable
from urllib import error, request


logger = logging.getLogger(__name__)
STARTUP_TIMEOUT = 180
REQUEST_TIMEOUT = 600
OPENROUTER_CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODEL = "google/gemini-3-flash-preview"
REASONING_EFFORTS = ("none", "low", "medium", "high")
REMOTE_SEED_RANGE = 2**31


def _check_interrupt() -> None:
    try:
        import comfy.model_management
    except ImportError:
        return
    comfy.model_management.throw_exception_if_processing_interrupted()


def free_comfy_memory() -> None:
    try:
        import comfy.model_management as model_management
        import torch
    except ImportError:
        return
    model_management.unload_all_models()
    model_management.soft_empty_cache(force=True)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def resolve_executable(value: str) -> str:
    expanded = os.path.expandvars(value.strip())
    if not expanded:
        raise FileNotFoundError("llama_server_path is empty")
    if "/" not in expanded and "\\" not in expanded:
        found = shutil.which(expanded)
        if not found:
            raise FileNotFoundError(f"llama-server was not found on PATH: {expanded}")
        return found
    path = Path(expanded).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"llama-server was not found: {path}")
    return str(path)


def build_command(
    server_path: str,
    model_path: Path,
    alias: str,
    port: int,
    context_size: int,
    kv_type: str,
    mmproj_path: Path | None,
    extra_args: tuple[str, ...] = (),
) -> list[str]:
    command = [
        server_path,
        "-m",
        str(model_path),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--no-ui",
        "--no-warmup",
        "-c",
        str(context_size),
        "-np",
        "1",
        "--cache-type-k",
        kv_type,
        "--cache-type-v",
        kv_type,
        "--alias",
        alias,
    ]
    if mmproj_path is not None:
        command.extend(("--mmproj", str(mmproj_path)))
    command.extend(extra_args)
    return command


def _drain_output(stream, output_tail: deque[str]) -> None:
    try:
        for line in iter(stream.readline, ""):
            output_tail.append(line.rstrip())
    except (OSError, ValueError):
        pass


def _wait_for_server(base_url: str, process: subprocess.Popen) -> None:
    deadline = time.monotonic() + STARTUP_TIMEOUT
    while time.monotonic() < deadline:
        _check_interrupt()
        if process.poll() is not None:
            raise RuntimeError(f"llama-server exited during startup with code {process.returncode}")
        try:
            with request.urlopen(f"{base_url}/health", timeout=1) as response:
                if response.status == 200:
                    return
        except (error.URLError, OSError):
            pass
        time.sleep(0.25)
    raise TimeoutError(f"llama-server did not become ready within {STARTUP_TIMEOUT} seconds")


def stop_server(process: subprocess.Popen | None) -> None:
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=8)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def _image_data_urls(images: list, max_side: int = 1024) -> list[tuple[str, str]]:
    from PIL import Image

    urls: list[tuple[str, str]] = []
    for reference_index, tensor in enumerate(images, 1):
        frames = tensor if tensor.ndim == 4 else tensor.unsqueeze(0)
        for frame_index, frame in enumerate(frames, 1):
            pixels = frame.clamp(0, 1).mul(255).byte().cpu().numpy()
            image = Image.fromarray(pixels)
            if max(image.size) > max_side:
                image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
            buffer = io.BytesIO()
            image.convert("RGB").save(buffer, format="JPEG", quality=90)
            encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
            label = f"Reference input {reference_index}, chronological frame {frame_index} of {len(frames)}:"
            urls.append((label, f"data:image/jpeg;base64,{encoded}"))
    return urls


def _image_data_url(tensor, max_side: int = 1536) -> str:
    from PIL import Image

    pixels = tensor.detach().clamp(0, 1).mul(255).byte().cpu().numpy()
    image = Image.fromarray(pixels)
    if max(image.size) > max_side:
        image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=90)
    return f"data:image/jpeg;base64,{base64.b64encode(buffer.getvalue()).decode('ascii')}"


def _video_data_url(frames, fps: int = 24) -> str:
    try:
        import av
    except ImportError as exc:
        raise RuntimeError("Sending video references requires PyAV from the ComfyUI environment.") from exc

    buffer = io.BytesIO()
    with av.open(buffer, mode="w", format="mp4", options={"movflags": "use_metadata_tags"}) as container:
        stream = container.add_stream("h264", rate=fps, options={"crf": "28", "preset": "veryfast"})
        stream.width = int(frames.shape[2])
        stream.height = int(frames.shape[1])
        stream.pix_fmt = "yuv420p"
        for tensor in frames:
            pixels = tensor.detach().clamp(0, 1).mul(255).byte().cpu().numpy()
            frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return f"data:video/mp4;base64,{base64.b64encode(buffer.getvalue()).decode('ascii')}"


def _audio_part(audio: dict) -> dict:
    waveform = audio["waveform"].detach().float().cpu().numpy()
    if waveform.ndim == 3:
        waveform = waveform[0]
    if waveform.ndim == 1:
        waveform = waveform[None, :]
    pcm = (waveform.clip(-1, 1) * 32767).astype("<i2")
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as output:
        output.setnchannels(int(pcm.shape[0]))
        output.setsampwidth(2)
        output.setframerate(int(audio["sample_rate"]))
        output.writeframes(pcm.T.reshape(-1).tobytes())
    return {
        "type": "input_audio",
        "input_audio": {"data": base64.b64encode(buffer.getvalue()).decode("ascii"), "format": "wav"},
    }


def _media_content(user_prompt: str, images: list, videos: list, audios: list, native_video: bool) -> str | list[dict]:
    if not images and not videos and not audios:
        return user_prompt

    content: list[dict] = [{"type": "text", "text": user_prompt}]
    for index, tensor in enumerate(images, 1):
        frame = tensor[0] if tensor.ndim == 4 else tensor
        content.append({"type": "text", "text": f"image_reference <Picture {index}>:"})
        content.append({"type": "image_url", "image_url": {"url": _image_data_url(frame)}})
    for index, frames in enumerate(videos, 1):
        content.append({"type": "text", "text": f"video_reference <Video {index}>:"})
        if native_video:
            content.append({"type": "video_url", "video_url": {"url": _video_data_url(frames)}})
        else:
            count = min(6, len(frames))
            indices = [round(i * (len(frames) - 1) / max(count - 1, 1)) for i in range(count)]
            for frame_index in indices:
                content.append({"type": "image_url", "image_url": {"url": _image_data_url(frames[frame_index])}})
    for index, audio in enumerate(audios, 1):
        content.append({"type": "text", "text": f"audio_reference <Audio {index}>:"})
        content.append(_audio_part(audio))
    return content


def _chat_payload(
    alias: str,
    system_prompt: str,
    user_prompt: str,
    images: list,
    max_tokens: int | None,
    temperature: float,
    seed: int,
    videos: list | None = None,
    audios: list | None = None,
    reasoning_effort: str = "none",
    native_video: bool = False,
    disable_thinking: bool = True,
) -> bytes:
    content = _media_content(user_prompt, images, videos or [], audios or [], native_video)
    payload = {
        "model": alias,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ],
        "temperature": temperature,
        "top_p": 0.9,
        "seed": seed,
    }
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens
    if disable_thinking:
        payload["chat_template_kwargs"] = {"enable_thinking": False}
    if reasoning_effort in REASONING_EFFORTS and reasoning_effort != "none":
        payload["reasoning"] = {"effort": reasoning_effort}
    return json.dumps(payload).encode("utf-8")


def _request_completion(
    base_url: str,
    alias: str,
    system_prompt: str,
    user_prompt: str,
    images: list,
    max_tokens: int,
    temperature: float,
    seed: int,
    *,
    api_key: str = "local",
    videos: list | None = None,
    audios: list | None = None,
    reasoning_effort: str = "none",
    native_video: bool = False,
    disable_thinking: bool = True,
) -> str:
    body = _chat_payload(
        alias, system_prompt, user_prompt, images, max_tokens, temperature, seed,
        videos, audios, reasoning_effort, native_video, disable_thinking,
    )
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    http_request = request.Request(
        base_url,
        data=body,
        headers=headers,
        method="POST",
    )
    try:
        with request.urlopen(http_request, timeout=REQUEST_TIMEOUT) as response:
            result = json.loads(response.read().decode("utf-8"))
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        if api_key:
            detail = detail.replace(api_key, "***")
        raise RuntimeError(f"prompt endpoint returned HTTP {exc.code}: {detail[:600]}") from exc
    try:
        return str(result["choices"][0]["message"]["content"] or "").strip()
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError("prompt endpoint returned an unexpected chat response") from exc


def endpoint_chat_url(api_base: str) -> str:
    base = api_base.strip().rstrip("/")
    if not base:
        raise ValueError("Local endpoint is selected but api_base is empty.")
    if not base.startswith(("http://", "https://")):
        raise ValueError("api_base must start with http:// or https://")
    return base if base.endswith("/chat/completions") else f"{base}/chat/completions"


def resolve_openrouter_key(value: str) -> str:
    key = value.strip() or os.environ.get("OPENROUTER_API_KEY", "").strip() or os.environ.get("LLM_KEY", "").strip()
    if not key:
        raise ValueError("OpenRouter needs a key in the node or OPENROUTER_API_KEY / LLM_KEY.")
    return key


def run_remote_endpoint(
    *, chat_url: str, api_key: str, model: str, system_prompt: str, user_prompt: str,
    images: list, videos: list, audios: list, max_tokens: int | None, temperature: float,
    seed: int, max_attempts: int, validator: Callable[[str], list[str]],
    reasoning_effort: str = "none",
    disable_thinking: bool = False,
) -> tuple[list[str], str]:
    if not model.strip():
        raise ValueError("The selected endpoint needs a model id.")
    outputs: list[str] = []
    initial_user_prompt = user_prompt
    started = time.monotonic()
    for attempt in range(max_attempts):
        output = request_completion_interruptible(
            chat_url, model.strip(), system_prompt, user_prompt, images, max_tokens,
            temperature, (seed + attempt) % REMOTE_SEED_RANGE, api_key=api_key,
            videos=videos, audios=audios, reasoning_effort=reasoning_effort,
            native_video=True, disable_thinking=disable_thinking,
        )
        outputs.append(output)
        errors = validator(output)
        if not errors:
            break
        user_prompt = (
            f"{initial_user_prompt}\n\nYour previous answer is copied below. Repair only the listed "
            "validation errors and return the complete final prompt again.\n\nValidation errors:\n- "
            + "\n- ".join(errors) + f"\n\nPrevious answer:\n{output}"
        )
    elapsed = time.monotonic() - started
    return outputs, f"Endpoint request completed after {elapsed:.1f}s"


def request_completion_interruptible(*args, **kwargs) -> str:
    result: dict[str, object] = {}

    def run() -> None:
        try:
            result["text"] = _request_completion(*args, **kwargs)
        except BaseException as exc:
            result["error"] = exc

    thread = Thread(target=run, daemon=True)
    thread.start()
    while thread.is_alive():
        thread.join(timeout=0.25)
        _check_interrupt()
    if "error" in result:
        raise result["error"]
    return str(result.get("text", ""))


def run_managed_server(
    server_path: str,
    model_path: Path,
    alias: str,
    context_size: int,
    kv_type: str,
    mmproj_path: Path | None,
    extra_args: tuple[str, ...],
    system_prompt: str,
    user_prompt: str,
    images: list,
    max_tokens: int,
    temperature: float,
    seed: int,
    max_attempts: int,
    validator: Callable[[str], list[str]],
    disable_thinking: bool = True,
) -> tuple[list[str], str]:
    executable = resolve_executable(server_path)
    model_path = model_path.resolve()
    if not model_path.is_file():
        raise FileNotFoundError(f"GGUF model was not found: {model_path}")
    if mmproj_path is not None:
        mmproj_path = mmproj_path.resolve()
        if not mmproj_path.is_file():
            raise FileNotFoundError(f"Vision projector was not found: {mmproj_path}")

    _check_interrupt()
    free_comfy_memory()
    port = find_free_port()
    command = build_command(
        executable,
        model_path,
        alias,
        port,
        context_size,
        kv_type,
        mmproj_path,
        extra_args,
    )
    flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
    output_tail: deque[str] = deque(maxlen=60)
    process: subprocess.Popen | None = None
    started = time.monotonic()
    outputs: list[str] = []
    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            creationflags=flags,
        )
        Thread(target=_drain_output, args=(process.stdout, output_tail), daemon=True).start()
        base_url = f"http://127.0.0.1:{port}"
        _wait_for_server(base_url, process)
        initial_user_prompt = user_prompt
        for attempt in range(max_attempts):
            output = request_completion_interruptible(
                f"{base_url}/v1/chat/completions",
                alias,
                system_prompt,
                user_prompt,
                images,
                max_tokens,
                temperature,
                (seed + attempt) % (2**32),
                disable_thinking=disable_thinking,
            )
            outputs.append(output)
            errors = validator(output)
            if not errors:
                break
            error_list = "\n- ".join(errors)
            user_prompt = (
                f"{initial_user_prompt}\n\n"
                "Your previous answer is copied below. Repair only the listed validation errors and return the complete final prompt again.\n\n"
                f"Validation errors:\n- {error_list}\n\n"
                f"Previous answer:\n{output}"
            )
    except BaseException as exc:
        tail = "\n".join(output_tail)
        if tail:
            logger.error("llama-server output before failure:\n%s", tail)
        raise exc
    finally:
        stop_server(process)

    elapsed = time.monotonic() - started
    return outputs, f"Managed llama-server released after {elapsed:.1f}s"


def random_seed(seed: int) -> int:
    return random.getrandbits(32) if seed < 0 else seed % (2**32)
