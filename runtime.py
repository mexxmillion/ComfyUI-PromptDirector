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
from collections import deque
from pathlib import Path
from threading import Thread
from typing import Callable
from urllib import error, request


logger = logging.getLogger(__name__)
STARTUP_TIMEOUT = 180
REQUEST_TIMEOUT = 600


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


def _image_data_urls(images: list, max_side: int = 1024) -> list[str]:
    from PIL import Image

    urls: list[str] = []
    for tensor in images:
        if tensor.ndim == 4:
            tensor = tensor[0]
        pixels = tensor.clamp(0, 1).mul(255).byte().cpu().numpy()
        image = Image.fromarray(pixels)
        if max(image.size) > max_side:
            image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
        buffer = io.BytesIO()
        image.convert("RGB").save(buffer, format="JPEG", quality=90)
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        urls.append(f"data:image/jpeg;base64,{encoded}")
    return urls


def _chat_payload(
    alias: str,
    system_prompt: str,
    user_prompt: str,
    images: list,
    max_tokens: int,
    temperature: float,
    seed: int,
) -> bytes:
    if images:
        content: str | list[dict] = [{"type": "text", "text": user_prompt}]
        for index, data_url in enumerate(_image_data_urls(images), 1):
            content.append({"type": "text", "text": f"Attached reference image {index}:"})
            content.append({"type": "image_url", "image_url": {"url": data_url}})
    else:
        content = user_prompt
    payload = {
        "model": alias,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": 0.9,
        "seed": seed,
        "chat_template_kwargs": {"enable_thinking": False},
    }
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
) -> str:
    body = _chat_payload(alias, system_prompt, user_prompt, images, max_tokens, temperature, seed)
    http_request = request.Request(
        f"{base_url}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json", "Authorization": "Bearer local"},
        method="POST",
    )
    try:
        with request.urlopen(http_request, timeout=REQUEST_TIMEOUT) as response:
            result = json.loads(response.read().decode("utf-8"))
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"llama-server returned HTTP {exc.code}: {detail[:600]}") from exc
    try:
        return str(result["choices"][0]["message"]["content"] or "").strip()
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError("llama-server returned an unexpected chat response") from exc


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
                base_url,
                alias,
                system_prompt,
                user_prompt,
                images,
                max_tokens,
                temperature,
                (seed + attempt) % (2**32),
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
