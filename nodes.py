from __future__ import annotations

import logging

from comfy_api.latest import ComfyExtension, io
from typing_extensions import override

try:
    from .config import load_config, profile_options, resolve_profile
    from .prompts import (
        CREATIVITY,
        H3_I2V,
        H3_R2V,
        KREA_EDIT,
        MODE_SPECS,
        SYSTEM_PROMPT_SOURCES,
        build_system_prompt,
        build_user_prompt,
        clean_output,
        ensure_h3_skin_color_artifact_guard,
        normalize_output,
        validate_output,
    )
    from .runtime import (
        OPENROUTER_CHAT_URL,
        OPENROUTER_MODEL,
        REASONING_EFFORTS,
        endpoint_chat_url,
        random_seed,
        resolve_openrouter_key,
        run_managed_server,
        run_remote_endpoint,
    )
    from .video import evenly_spaced_indices, sample_video_frames
except ImportError:
    from config import load_config, profile_options, resolve_profile
    from prompts import (
        CREATIVITY,
        H3_I2V,
        H3_R2V,
        KREA_EDIT,
        MODE_SPECS,
        SYSTEM_PROMPT_SOURCES,
        build_system_prompt,
        build_user_prompt,
        clean_output,
        ensure_h3_skin_color_artifact_guard,
        normalize_output,
        validate_output,
    )
    from runtime import (
        OPENROUTER_CHAT_URL,
        OPENROUTER_MODEL,
        REASONING_EFFORTS,
        endpoint_chat_url,
        random_seed,
        resolve_openrouter_key,
        run_managed_server,
        run_remote_endpoint,
    )
    from video import evenly_spaced_indices, sample_video_frames


logger = logging.getLogger(__name__)
PROMPT_PROVIDERS = ("local llama", "openrouter", "custom endpoint", "none")
REFERENCE_MANAGER_PRESET = "MiniMax H3 — Reference Manager"
CUSTOM_SYSTEM_PROMPT_FILE_PRESET = "Custom system prompt file"
CUSTOM_SYSTEM_PROMPT_PRESET = "Custom system prompt"
SYSTEM_PROMPT_PRESETS = (
    *MODE_SPECS,
    REFERENCE_MANAGER_PRESET,
    CUSTOM_SYSTEM_PROMPT_FILE_PRESET,
    CUSTOM_SYSTEM_PROMPT_PRESET,
)
LEGACY_PROVIDERS = {
    "local": "local llama",
    "managed llama-server": "local llama",
    "local endpoint": "custom endpoint",
}


def _profile_default(config, labels: list[str]) -> str:
    for profile in config.profiles:
        if profile.profile_id == config.default_profile:
            return profile.label
    return labels[0]


class PromptDirector(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        config = load_config()
        profile_labels, _ = profile_options(config)
        return io.Schema(
            node_id="PromptDirector",
            display_name="Prompt Director",
            category="prompt/director",
            description=(
                "Enhances prompts with a local GGUF model, OpenRouter, or a custom OpenAI-compatible "
                "endpoint, with H3 media pass-through."
            ),
            inputs=[
                io.String.Input("prompt", multiline=True, dynamic_prompts=True, default="", display_name="request"),
                io.Combo.Input("mode", options=list(SYSTEM_PROMPT_PRESETS), default=next(iter(MODE_SPECS)),
                               display_name="system prompt preset"),
                io.Combo.Input(
                    "profile",
                    options=profile_labels,
                    default=_profile_default(config, profile_labels),
                    display_name="local vision model",
                    tooltip="Used only for Local Llama. GGUF models from config.json or its launcher catalog.",
                ),
                io.Combo.Input("creativity", options=list(CREATIVITY), default="Balanced", advanced=True),
                io.Int.Input(
                    "duration",
                    default=8,
                    min=4,
                    max=15,
                    step=1,
                    tooltip="Used by MiniMax H3 modes. Ignored by Krea 2 modes.",
                ),
                io.Int.Input(
                    "seed",
                    default=0,
                    min=-1,
                    max=0xFFFFFFFF,
                    control_after_generate=io.ControlAfterGenerate.randomize,
                    advanced=True,
                ),
                io.Int.Input("max_attempts", default=2, min=1, max=4, display_name="retries"),
                io.String.Input(
                    "reference_context",
                    default="",
                    multiline=True,
                    optional=True,
                    advanced=True,
                    display_name="video frame metadata",
                    tooltip="Optional timeline metadata from Video Frames for VL.",
                ),
                io.Autogrow.Input(
                    "reference_images",
                    optional=True,
                    template=io.Autogrow.TemplatePrefix(
                        input=io.Image.Input(
                            "reference_image",
                            tooltip="Visual context for the LLM. Connect the same media to the downstream generation node.",
                        ),
                        prefix="reference_image_",
                        min=0,
                        max=9,
                    ),
                ),
                io.Autogrow.Input(
                    "reference_videos",
                    optional=True,
                    template=io.Autogrow.TemplatePrefix(
                        input=io.Image.Input("reference_video", tooltip="Reference video frames at 24 fps."),
                        prefix="reference_video_",
                        min=0,
                        max=3,
                    ),
                ),
                io.Autogrow.Input(
                    "reference_audios",
                    optional=True,
                    template=io.Autogrow.TemplatePrefix(
                        input=io.Audio.Input("reference_audio", tooltip="Standalone reference audio."),
                        prefix="reference_audio_",
                        min=0,
                        max=3,
                    ),
                ),
                io.Combo.Input("prompt_provider", options=list(PROMPT_PROVIDERS), default=PROMPT_PROVIDERS[0],
                               display_name="inference mode",
                               tooltip="Local Llama starts the selected GGUF; OpenRouter uses its API; Custom Endpoint calls your URL; None returns the request unchanged."),
                io.String.Input("openrouter_api_key", default="", optional=True, display_name="OpenRouter API key",
                                tooltip="Blank uses OPENROUTER_API_KEY or LLM_KEY."),
                io.String.Input("openrouter_model", default=OPENROUTER_MODEL, optional=True,
                                display_name="OpenRouter model preset / ID",
                                tooltip="Used only for OpenRouter. Defaults to Gemini 3 Flash Preview."),
                io.Combo.Input("reasoning_effort", options=list(REASONING_EFFORTS), default="medium", optional=True,
                               display_name="OpenRouter reasoning",
                               tooltip="Used only for OpenRouter."),
                io.String.Input("api_base", default="http://127.0.0.1:8080/v1", optional=True, display_name="endpoint URL preset / custom",
                                tooltip="Custom endpoint URL, for example http://127.0.0.1:8080/v1."),
                io.String.Input("local_model_slug", default="Bonsai-2-27B-Ternary-CRACK", optional=True,
                                display_name="endpoint model preset / ID",
                                tooltip="Used only for custom endpoint. Model id reported by that server."),
                io.String.Input("local_api_key", default="", optional=True,
                                display_name="endpoint API key",
                                tooltip="Used only for custom endpoint. Optional key; OpenRouter environment keys are never reused here."),
                io.Boolean.Input("disable_endpoint_thinking", default=True, optional=True,
                                 display_name="disable endpoint thinking",
                                 tooltip="Custom endpoint only. Sends chat_template_kwargs.enable_thinking=false to llama.cpp-compatible servers. Turn off if your server rejects this field."),
                io.Combo.Input("system_prompt_source", options=list(SYSTEM_PROMPT_SOURCES),
                               default=SYSTEM_PROMPT_SOURCES[0], optional=True,
                               display_name="system prompt mode"),
                io.String.Input("system_prompt_file", default="", optional=True,
                                display_name="system prompt file (.md/.txt)",
                                tooltip="Legacy compatibility field. Use System Prompt Override for new workflows."),
                io.String.Input("custom_system_prompt", default="", multiline=True, optional=True,
                                display_name="system prompt override (text)",
                                tooltip="Optional direct system prompt for local, OpenRouter, or custom endpoint. Non-blank text replaces the selected preset."),
                io.Boolean.Input("local_llama_thinking", default=True, optional=True,
                                 display_name="local thinking",
                                 tooltip="Use the selected local model's thinking mode. Turn off for a faster direct answer."),
                io.Boolean.Input("fail_on_error", default=False, optional=True, advanced=True,
                                 display_name="stop on enhancement failure",
                                 tooltip="When enabled, fail the workflow rather than quietly using the original prompt."),
            ],
            outputs=[
                io.String.Output("enhanced_prompt", display_name="enhanced_prompt"),
                io.String.Output("report", display_name="report"),
                *[io.Image.Output(f"image_{index}") for index in range(1, 10)],
                *[io.Image.Output(f"video_{index}") for index in range(1, 4)],
                *[io.Audio.Output(f"audio_{index}") for index in range(1, 4)],
            ],
        )

    @classmethod
    def execute(
        cls,
        prompt: str,
        mode: str,
        profile: str,
        creativity: str,
        duration: int,
        seed: int,
        max_attempts: int,
        local_llama_thinking: bool = True,
        reference_context: str = "",
        reference_images: dict[str, object] | None = None,
        reference_videos: dict[str, object] | None = None,
        reference_audios: dict[str, object] | None = None,
        prompt_provider: str = PROMPT_PROVIDERS[0],
        openrouter_api_key: str = "",
        openrouter_model: str = OPENROUTER_MODEL,
        reasoning_effort: str = "medium",
        api_base: str = "",
        local_model_slug: str = "",
        local_api_key: str = "",
        disable_endpoint_thinking: bool = True,
        system_prompt_source: str = SYSTEM_PROMPT_SOURCES[0],
        system_prompt_file: str = "",
        custom_system_prompt: str = "",
        fail_on_error: bool = False,
    ) -> io.NodeOutput:
        images = [reference_images[key] for key in sorted(reference_images or {}) if reference_images[key] is not None]
        videos = [reference_videos[key] for key in sorted(reference_videos or {}) if reference_videos[key] is not None]
        audios = [reference_audios[key] for key in sorted(reference_audios or {}) if reference_audios[key] is not None]
        passthrough = images + [None] * (9 - len(images)) + videos + [None] * (3 - len(videos)) + audios + [None] * (3 - len(audios))

        original = prompt.strip()
        if not original:
            return io.NodeOutput("", "Prompt Director bypassed an empty prompt.", *passthrough)

        prompt_provider = LEGACY_PROVIDERS.get(prompt_provider, prompt_provider)
        if prompt_provider not in PROMPT_PROVIDERS:
            return io.NodeOutput(original, f"Unknown Prompt Director provider: {prompt_provider}", *passthrough)
        if prompt_provider == "none":
            return io.NodeOutput(original, "Inference mode: none; request returned unchanged.", *passthrough)

        preset = mode
        custom_preset = preset in (CUSTOM_SYSTEM_PROMPT_FILE_PRESET, CUSTOM_SYSTEM_PROMPT_PRESET)
        if preset == REFERENCE_MANAGER_PRESET:
            mode = H3_R2V
            system_prompt_source = "MiniMax Reference Manager"
        elif custom_preset:
            mode = H3_R2V
            system_prompt_source = "custom file" if preset == CUSTOM_SYSTEM_PROMPT_FILE_PRESET else "custom text"

        config = load_config()
        selected = None
        if prompt_provider == PROMPT_PROVIDERS[0]:
            try:
                selected = resolve_profile(profile, config)
            except ValueError as exc:
                return io.NodeOutput(original, f"Prompt Director configuration error: {exc}", *passthrough)

        warnings: list[str] = []
        visual_images = list(images)
        mmproj_path = selected.mmproj_path if selected is not None and (images or videos) else None
        if selected is not None and (images or videos) and mmproj_path is None:
            warnings.append("The selected profile has no vision projector; reference images were not sent to the LLM.")
            visual_images = []
        if not custom_preset and mode in (KREA_EDIT, H3_I2V, H3_R2V) and not (images or videos):
            warnings.append("This mode normally uses at least one reference image.")
        if selected is not None and videos and mmproj_path is not None:
            visual_images += [video[evenly_spaced_indices(len(video), 6)] for video in videos]
            warnings.append("Managed llama-server receives video references as sampled frames.")
        if selected is not None and audios:
            warnings.append("Managed llama-server does not receive audio references; they are still passed through.")

        spec = MODE_SPECS[mode]
        inline_system_prompt = custom_system_prompt.strip()
        if not inline_system_prompt and system_prompt_source == "custom file":
            candidate = system_prompt_file.strip()
            if candidate and not candidate.casefold().endswith((".md", ".txt")):
                inline_system_prompt = candidate
        if inline_system_prompt:
            system_prompt = inline_system_prompt
        else:
            try:
                system_prompt = build_system_prompt(mode, creativity, system_prompt_source, system_prompt_file)
            except (OSError, ValueError) as exc:
                return io.NodeOutput(original, f"Prompt Director system prompt error: {exc}", *passthrough)
        system_prompt = ensure_h3_skin_color_artifact_guard(mode, system_prompt)
        legacy_video_frames = not videos and len(images) == 1 and "<Video 1>" in reference_context
        if custom_preset:
            labels = [f"<Picture {index}>" for index in range(1, len(images) + 1)]
            labels += [f"<Video {index}>" for index in range(1, len(videos) + 1)]
            labels += [f"<Audio {index}>" for index in range(1, len(audios) + 1)]
            user_prompt = original
            if labels:
                user_prompt += f"\n\nAttached reference labels: {', '.join(labels)}"
        else:
            user_prompt = build_user_prompt(
                mode,
                original,
                duration,
                0 if legacy_video_frames else len(images),
                video_count=1 if legacy_video_frames else len(videos),
                audio_count=len(audios),
            )
        if reference_context.strip():
            user_prompt += f"\n\nReference sequence metadata:\n{reference_context.strip()}"
        reference_labels = ("<Video 1>",) if legacy_video_frames else tuple(
            [f"<Picture {index}>" for index in range(1, len(images) + 1)]
            + [f"<Video {index}>" for index in range(1, len(videos) + 1)]
            + [f"<Audio {index}>" for index in range(1, len(audios) + 1)]
        )
        validator = lambda value: [] if custom_preset else validate_output(
            mode, normalize_output(mode, value), len(images), reference_labels,
        )
        temperature = {"Faithful": 0.2, "Balanced": 0.45, "Cinematic": 0.65}.get(creativity, 0.45)

        try:
            if selected is not None:
                outputs, lifecycle = run_managed_server(
                    server_path=config.server_path, model_path=selected.model_path, alias=selected.alias,
                    context_size=spec.context_size, kv_type=selected.kv_type, mmproj_path=mmproj_path,
                    extra_args=selected.extra_server_args, system_prompt=system_prompt, user_prompt=user_prompt,
                    images=visual_images, max_tokens=spec.max_tokens, temperature=temperature,
                    seed=random_seed(seed), max_attempts=max_attempts, validator=validator,
                    disable_thinking=not bool(local_llama_thinking),
                )
                provider_report = f"Provider: local (managed llama-server); model: {selected.label}."
            else:
                is_openrouter = prompt_provider == "openrouter"
                outputs, lifecycle = run_remote_endpoint(
                    chat_url=OPENROUTER_CHAT_URL if is_openrouter else endpoint_chat_url(api_base),
                    api_key=resolve_openrouter_key(openrouter_api_key) if is_openrouter else local_api_key.strip(),
                    model=openrouter_model if is_openrouter else local_model_slug,
                    system_prompt=system_prompt, user_prompt=user_prompt, images=images, videos=videos,
                    audios=audios, max_tokens=None if is_openrouter else spec.max_tokens, temperature=temperature,
                    seed=random_seed(seed), max_attempts=max_attempts, validator=validator,
                    reasoning_effort=reasoning_effort if is_openrouter else "none",
                    disable_thinking=bool(disable_endpoint_thinking) if not is_openrouter else False,
                )
                provider_report = f"Provider: {'openrouter' if is_openrouter else 'custom endpoint (' + endpoint_chat_url(api_base) + ')'}; model: {openrouter_model if is_openrouter else local_model_slug}."
        except Exception as exc:
            logger.exception("Prompt Director failed")
            detail = str(exc)
            for secret in (openrouter_api_key, local_api_key):
                if secret and secret in detail:
                    detail = detail.replace(secret, "***")
            report = f"Prompt enhancement failed; original prompt returned. {type(exc).__name__}: {detail}"
            if fail_on_error:
                raise RuntimeError(f"Prompt Director enhancement failed: {type(exc).__name__}: {detail}") from None
            return io.NodeOutput(original, report, *passthrough)

        final = (clean_output(outputs[-1]) if custom_preset else normalize_output(mode, outputs[-1])) if outputs else ""
        errors = ([] if custom_preset else validate_output(mode, final, len(images), reference_labels)) if final else ["The LLM returned no text."]
        if errors and fail_on_error:
            raise RuntimeError("Prompt Director output failed validation: " + " ".join(errors))
        status = "passed validation" if not errors else "returned with validation warnings"
        report_parts = [
            f"{preset}: {status} after {len(outputs)} attempt(s).",
            provider_report,
            f"References sent to LLM: {len(images)} image(s), {len(videos)} video(s), {len(audios)} audio clip(s).",
            lifecycle + ".",
        ]
        if errors:
            report_parts.append("Validation: " + " ".join(errors))
        report_parts.extend(warnings)
        return io.NodeOutput(final or original, "\n".join(report_parts), *passthrough)


class VideoFramesForVL(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="VideoFramesForVL",
            display_name="Video Frames for VL",
            category="prompt/director",
            description="Evenly sample a VHS video frame batch for a vision-language model.",
            inputs=[
                io.Image.Input("video_frames", tooltip="Connect IMAGE from a VHS Load Video node."),
                io.Int.Input("sample_count", default=8, min=1, max=32, step=1),
                io.Custom("VHS_VIDEOINFO").Input(
                    "video_info",
                    optional=True,
                    tooltip="Connect video_info from VHS to include relative timestamps.",
                ),
            ],
            outputs=[
                io.Image.Output("sampled_frames"),
                io.String.Output("reference_context"),
                io.Int.Output("sampled_count"),
            ],
        )

    @classmethod
    def execute(cls, video_frames, sample_count: int, video_info: dict | None = None) -> io.NodeOutput:
        sampled, context, count = sample_video_frames(video_frames, sample_count, video_info)
        return io.NodeOutput(sampled, context, count)


NODE_CLASS_MAPPINGS = {"PromptDirector": PromptDirector, "VideoFramesForVL": VideoFramesForVL}
NODE_DISPLAY_NAME_MAPPINGS = {
    "PromptDirector": "Prompt Director",
    "VideoFramesForVL": "Video Frames for VL",
}


class PromptDirectorExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [PromptDirector, VideoFramesForVL]


async def comfy_entrypoint() -> PromptDirectorExtension:
    return PromptDirectorExtension()
