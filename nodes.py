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
        build_system_prompt,
        build_user_prompt,
        normalize_output,
        validate_output,
    )
    from .runtime import random_seed, run_managed_server
    from .video import sample_video_frames
except ImportError:
    from config import load_config, profile_options, resolve_profile
    from prompts import (
        CREATIVITY,
        H3_I2V,
        H3_R2V,
        KREA_EDIT,
        MODE_SPECS,
        build_system_prompt,
        build_user_prompt,
        normalize_output,
        validate_output,
    )
    from runtime import random_seed, run_managed_server
    from video import sample_video_frames


logger = logging.getLogger(__name__)


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
                "Enhances prompts for Krea 2 and MiniMax H3 with an isolated local llama-server. "
                "The server is stopped before downstream image or video models run."
            ),
            inputs=[
                io.String.Input("prompt", multiline=True, dynamic_prompts=True, default=""),
                io.Combo.Input("mode", options=list(MODE_SPECS), default=next(iter(MODE_SPECS))),
                io.Combo.Input(
                    "profile",
                    options=profile_labels,
                    default=_profile_default(config, profile_labels),
                    tooltip="Local GGUF profile from config.json or its linked launcher catalog.",
                ),
                io.Combo.Input("creativity", options=list(CREATIVITY), default="Balanced"),
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
                ),
                io.Int.Input("max_attempts", default=2, min=1, max=4, advanced=True),
                io.String.Input(
                    "reference_context",
                    default="",
                    multiline=True,
                    optional=True,
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
            ],
            outputs=[
                io.String.Output("enhanced_prompt", display_name="enhanced_prompt"),
                io.String.Output("report", display_name="report"),
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
        reference_context: str = "",
        reference_images: dict[str, object] | None = None,
    ) -> io.NodeOutput:
        original = prompt.strip()
        if not original:
            return io.NodeOutput("", "Prompt Director bypassed an empty prompt.")

        config = load_config()
        try:
            selected = resolve_profile(profile, config)
        except ValueError as exc:
            return io.NodeOutput(original, f"Prompt Director configuration error: {exc}")

        images = []
        if reference_images:
            images = [reference_images[key] for key in sorted(reference_images) if reference_images[key] is not None]

        warnings: list[str] = []
        visual_images = images
        mmproj_path = selected.mmproj_path if images else None
        if images and mmproj_path is None:
            warnings.append("The selected profile has no vision projector; reference images were not sent to the LLM.")
            visual_images = []
        if mode in (KREA_EDIT, H3_I2V, H3_R2V) and not images:
            warnings.append("This mode normally uses at least one reference image.")

        spec = MODE_SPECS[mode]
        system_prompt = build_system_prompt(mode, creativity)
        user_prompt = build_user_prompt(mode, original, duration, len(images))
        if reference_context.strip():
            user_prompt += f"\n\nReference sequence metadata:\n{reference_context.strip()}"
        reference_labels = ("<Video 1>",) if "<Video 1>" in reference_context else ()
        validator = lambda value: validate_output(
            mode,
            normalize_output(mode, value),
            len(images),
            reference_labels,
        )
        temperature = {"Faithful": 0.2, "Balanced": 0.45, "Cinematic": 0.65}.get(creativity, 0.45)

        try:
            outputs, lifecycle = run_managed_server(
                server_path=config.server_path,
                model_path=selected.model_path,
                alias=selected.alias,
                context_size=spec.context_size,
                kv_type=selected.kv_type,
                mmproj_path=mmproj_path,
                extra_args=selected.extra_server_args,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                images=visual_images,
                max_tokens=spec.max_tokens,
                temperature=temperature,
                seed=random_seed(seed),
                max_attempts=max_attempts,
                validator=validator,
            )
        except Exception as exc:
            logger.exception("Prompt Director failed")
            report = f"Prompt enhancement failed; original prompt returned. {type(exc).__name__}: {exc}"
            return io.NodeOutput(original, report)

        final = normalize_output(mode, outputs[-1]) if outputs else ""
        errors = validate_output(mode, final, len(images), reference_labels) if final else ["The LLM returned no text."]
        status = "passed validation" if not errors else "returned with validation warnings"
        report_parts = [
            f"{mode}: {status} after {len(outputs)} attempt(s).",
            f"Profile: {selected.label}.",
            f"References sent to LLM: {len(visual_images)}.",
            lifecycle + ".",
        ]
        if errors:
            report_parts.append("Validation: " + " ".join(errors))
        report_parts.extend(warnings)
        return io.NodeOutput(final or original, "\n".join(report_parts))


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
