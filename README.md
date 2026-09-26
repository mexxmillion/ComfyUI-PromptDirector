# ComfyUI Prompt Director

Multimodal prompt enhancement for Krea 2 and MiniMax H3. The normal node surface is the request, one system-prompt preset, references, and one inference mode. Select `local llama` to start a configured GGUF, `openrouter` for the hosted API, `custom endpoint` for an existing OpenAI-compatible server, or `none` to pass the request through unchanged. Only fields required by the selected inference mode are shown.

## System prompt presets

| Mode | Output contract |
| --- | --- |
| Krea 2 — Text to Image | One faithful natural-language image prompt |
| Krea 2 — Image Edit | Requested modifications followed by explicit preservation instructions |
| MiniMax H3 — Text to Video | H3's three audiovisual fields |
| MiniMax H3 — Image to Video | Exact first-frame instruction followed by H3's three audiovisual fields |
| MiniMax H3 — First and Last Frame to Video | H3's three audiovisual fields, with distinct first/last frame roles or no frames for text-to-video |
| MiniMax H3 — Reference to Video | H3's six-section full-reference format with stable labels |
| MiniMax H3 — Reference Manager | The bundled MiniMax Reference Manager instructions |
| Custom system prompt file | A `.md` or `.txt` file under `system_prompts` |
| Custom system prompt | Direct multiline system-prompt text |

The node has incremental groups for up to nine images, three videos, and three audio clips. OpenRouter and existing local endpoints receive the native media content. The managed llama-server path sends video frames through its vision projector and leaves audio out of the LLM request. All media is passed through unchanged on H3-ready outputs in either case.

`local llama` shows the configured GGUF choices plus a thinking toggle. OpenRouter shows its model ID, reasoning level, retry count, and key. `custom endpoint` defaults to the local `http://127.0.0.1:8080/v1` Bonsai preset and shows URL, model ID, thinking control, retries, and optional key. It never inherits the OpenRouter environment key. The report names the route and model actually used. Older saved provider values are still accepted.

The optional **stop on enhancement failure** setting makes a workflow fail if the provider call fails or the enhanced prompt does not pass validation. It is off by default for existing graphs; Pocket AI Generator enables it for FL2V so a failed OpenRouter call cannot silently produce an unenhanced render.

The system-prompt preset also defines the expected output contract, so there is no separate output-target field. Choosing a custom preset reveals only its filename or multiline text input. Custom presets do not inherit a hidden Krea or H3 output contract.

`Video Frames for VL` remains available when a sampled-frame view and timestamp context are preferable to sending a full clip.

The Krea 2 instructions follow the official [Krea 2 expansion guidance](https://github.com/krea-ai/krea-2/blob/main/docs/expansion.txt). H3 modes follow MiniMax's official [base-mode](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/docs/VIDEO_PROMPT_WRITING_GUIDE_base_en.md) and [full-reference](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/docs/VIDEO_PROMPT_WRITING_GUIDE_ref_en.md) guides.

Every MiniMax H3 system-prompt path—including bundled, Reference Manager, file, and inline overrides—adds renderer guards that forbid facial or body skin-color changes associated with blushing/flushing and unreliable tongue-out, lip-licking, or lip-biting expressions. Emotion and intent are expressed through reliable performance direction instead. These guards are H3-specific and do not alter Krea prompts.

## Why an external process?

Loading a local LLM and an image/video model together can exceed VRAM. Prompt Director uses this sequence:

```text
unload ComfyUI-managed models
→ start an owned llama-server process
→ generate and validate the prompt
→ stop llama-server and wait for exit
→ return the prompt to Krea 2 or H3 conditioning
```

The node owns only the server process it starts. It does not scan for or terminate other llama.cpp instances.

## Install

Clone the repository into `custom_nodes`:

```powershell
cd C:\path\to\ComfyUI\custom_nodes
git clone https://github.com/mexxmillion/ComfyUI-PromptDirector.git
```

Copy the example configuration and edit the paths:

```powershell
cd ComfyUI-PromptDirector
Copy-Item config.example.json config.json
```

`config.json` is ignored by Git so machine-specific model paths stay local.

Prompt Director has no additional Python dependencies. It uses Pillow and PyTorch from ComfyUI only when reference images are connected.

## Configuration

You can define profiles directly:

```json
{
  "llama_server_path": "E:/LLM/llama.cpp/llama-server.exe",
  "default_profile": "qwen-9b-vision",
  "extra_server_args": ["-ngl", "99", "-fa", "on", "--jinja", "--reasoning", "off"],
  "profiles": [
    {
      "id": "qwen-9b-vision",
      "label": "Qwen 9B Vision",
      "model": "E:/LLM/models/qwen-9b-q4_k_m.gguf",
      "mmproj": "E:/LLM/models/mmproj-f16.gguf",
      "alias": "qwen-9b",
      "kv": "q8_0"
    }
  ]
}
```

Or reuse an existing launcher catalog whose entries contain `id`, `label`, `model`, optional `mmproj`, `alias`, and `kv` fields:

```json
{
  "launcher_catalog": "E:/LLM/configs/launcher-profiles.json",
  "llama_server_path": "E:/LLM/llama.cpp/llama-server.exe",
  "default_profile": "qwen9b-vision-160k",
  "extra_server_args": ["-ngl", "99", "-fa", "on", "--jinja", "--reasoning", "off"]
}
```

Relative model paths in a catalog under a directory named `configs` are resolved from the catalog directory's parent. Remote-only profiles are excluded because this node guarantees cleanup only for a local process it owns.

Prompt Director chooses a smaller context for each mode instead of inheriting a chat profile's potentially huge context:

| Mode | Context | Maximum output |
| --- | ---: | ---: |
| Krea 2 T2I | 4,096 | 420 tokens |
| Krea 2 Edit | 8,192 | 360 tokens |
| H3 T2V | 12,288 | 900 tokens |
| H3 I2V | 16,384 | 1,000 tokens |
| H3 R2V | 20,480 | 1,500 tokens |

## Workflow

```text
text idea ────────────┐
images/videos/audio ──┼→ Prompt Director ──→ enhanced_prompt → H3 text encode
                      └────────────────────→ image_N/video_N/audio_N → H3 media inputs
```

Connect Prompt Director's outputs to the corresponding MiniMax H3 groups: `image_1` to the first `ref_image` socket, `video_1` to the first `ref_video` socket, and `audio_1` to the first `ref_audio` socket. The tensors and audio dictionaries pass through unchanged, so the tags used in the generated prompt stay aligned with downstream conditioning.

For a video reference, connect a 24 fps `IMAGE` frame batch directly:

```text
Load Video IMAGE → Prompt Director reference_video_1 → video_1 → H3 first ref_video socket
```

`Video Frames for VL` is still useful with the managed llama-server path when you want explicit timestamp metadata or a smaller sampled batch. OpenRouter and existing local endpoints receive direct video inputs as an inline MP4.

The `report` output states which profile ran, how many attempts were used, whether structural validation passed, how many images reached the LLM, and whether the managed server was released.

## Prompt design

The prompt library is deliberately small. It combines:

1. A shared intent and reference-fidelity contract.
2. One target-native Krea 2 or H3 grammar.
3. One mode overlay.
4. A `Faithful`, `Balanced`, or `Cinematic` creative direction.

H3 validation checks required sections, ordering, shot labels, the exact I2V first-frame instruction, minimum useful length, and Picture references in R2V. The mechanical H3 rule that Shot 1 has no zero timestamp is normalized automatically. A failed result is sent back to the same running LLM once by default with precise repair instructions.

## Development

Run the isolated tests without starting ComfyUI or loading a model:

```powershell
python -m pytest -q
```

## Acknowledgements

The grammar-as-data approach was inspired by [Prompt Director](https://github.com/marcoaiwithfefe-hub/prompt-director). The owned-process VRAM handoff was informed by community llama.cpp integrations including [ComfyUI-PromptEnhancer](https://github.com/MaxKruse/ComfyUI-PromptEnhancer) and [ComfyUI llama.cpp Suite](https://github.com/Setmaster/comfyui-llamacpp). This repository contains its own implementation and prompt text.

## License

MIT
