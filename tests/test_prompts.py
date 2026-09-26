import prompts
from prompts import (
    H3_BASE_FIELDS,
    H3_FL2V,
    H3_I2V,
    H3_R2V,
    H3_REFERENCE_FIELDS,
    I2V_PREFIX,
    KREA_EDIT,
    KREA_T2I,
    MODE_SPECS,
    build_system_prompt,
    build_user_prompt,
    clean_output,
    ensure_h3_skin_color_artifact_guard,
    normalize_output,
    validate_output,
)


def test_exact_six_modes():
    assert list(MODE_SPECS) == [
        KREA_T2I,
        KREA_EDIT,
        "MiniMax H3 — Text to Video",
        H3_I2V,
        H3_FL2V,
        H3_R2V,
    ]


def test_krea_system_prompt_is_mode_specific():
    prompt = build_system_prompt(KREA_EDIT, "Faithful")
    assert "MODE: Image Edit" in prompt
    assert "Preserve every unspecified" in prompt
    assert "intent-locked" in prompt
    assert "Image 2's pose and framing" in prompt


def test_every_builtin_h3_prompt_has_absolute_skin_color_artifact_guard():
    for mode in ("MiniMax H3 — Text to Video", H3_I2V, H3_FL2V, H3_R2V):
        prompt = build_system_prompt(mode, "Balanced")
        assert "MINIMAX H3 SKIN-COLOR ARTIFACT GUARD:" in prompt
        assert "Never repeat such wording from the user's request." in prompt
        assert "absolute renderer-compatibility rule" in prompt


def test_h3_guard_wraps_custom_prompts_but_not_krea():
    custom = "A custom system prompt."
    guarded = ensure_h3_skin_color_artifact_guard(H3_FL2V, custom)
    assert guarded.startswith(custom)
    assert "SKIN-COLOR ARTIFACT GUARD" in guarded
    assert ensure_h3_skin_color_artifact_guard(KREA_EDIT, custom) == custom


def test_custom_system_prompt_is_loaded_from_the_prompt_directory(tmp_path, monkeypatch):
    prompt_dir = tmp_path / "system_prompts"
    prompt_dir.mkdir()
    (prompt_dir / "my_h3.md").write_text("My custom H3 system prompt", encoding="utf-8")
    monkeypatch.setattr(prompts, "CUSTOM_SYSTEM_PROMPT_DIR", prompt_dir)
    result = prompts.build_system_prompt(H3_R2V, "Balanced", "custom file", "my_h3.md")
    assert result.startswith("My custom H3 system prompt")
    assert "MINIMAX H3 SKIN-COLOR ARTIFACT GUARD:" in result


def test_custom_system_prompt_cannot_escape_the_prompt_directory(tmp_path, monkeypatch):
    prompt_dir = tmp_path / "system_prompts"
    prompt_dir.mkdir()
    monkeypatch.setattr(prompts, "CUSTOM_SYSTEM_PROMPT_DIR", prompt_dir)
    try:
        prompts.build_system_prompt(H3_R2V, "Balanced", "custom file", "../secret.txt")
    except ValueError as exc:
        assert "must be inside" in str(exc)
    else:
        raise AssertionError("path traversal was accepted")


def test_reference_manager_system_prompt_can_be_selected(tmp_path, monkeypatch):
    manager_prompt = tmp_path / "system_prompt.md"
    manager_prompt.write_text("Reference Manager instructions", encoding="utf-8")
    monkeypatch.setattr(prompts, "REFERENCE_MANAGER_PROMPT_CANDIDATES", (manager_prompt,))
    result = prompts.build_system_prompt(H3_R2V, "Balanced", "MiniMax Reference Manager")
    assert result.startswith("Reference Manager instructions")
    assert "MINIMAX H3 SKIN-COLOR ARTIFACT GUARD:" in result


def test_repository_contains_the_reference_manager_prompt_clone():
    path = prompts.CUSTOM_SYSTEM_PROMPT_DIR / "minimax_reference_manager.md"
    assert path.is_file()
    assert "subject_definitions:" in path.read_text(encoding="utf-8")
    assert "Transfer only Picture 2's pose and framing" in path.read_text(encoding="utf-8")


def test_clean_output_removes_reasoning_and_fences():
    raw = "<think>private plan</think>\n```text\nA finished prompt.\n```"
    assert clean_output(raw) == "A finished prompt."


def test_h3_i2v_contract_accepts_valid_shape():
    body = " ".join(["visible motion and synchronized sound"] * 30)
    text = (
        f"{I2V_PREFIX}\n\n"
        f"{H3_BASE_FIELDS[0]} [Shot 1] {body}\n\n"
        f"{H3_BASE_FIELDS[1]} {body}\n\n"
        f"{H3_BASE_FIELDS[2]} N/A"
    )
    assert validate_output(H3_I2V, text, 1) == []


def test_h3_fl2v_prompt_preserves_frame_roles():
    prompt = build_system_prompt(H3_FL2V, "Balanced")
    assert "A last frame is the literal closing image" in prompt
    assert "With no images" in prompt
    body = " ".join(["continuous camera movement and natural sound"] * 30)
    output = "\n".join(
        [
            f"{H3_BASE_FIELDS[0]} [Shot 1] {body}",
            f"{H3_BASE_FIELDS[1]} {body}",
            f"{H3_BASE_FIELDS[2]} N/A",
        ]
    )
    assert validate_output(H3_FL2V, output, 2) == []


def test_h3_r2v_reports_missing_picture():
    body = " ".join(["reference detail"] * 60)
    text = "\n".join(f"{field} {body}" for field in H3_REFERENCE_FIELDS)
    errors = validate_output(H3_R2V, text, 2)
    assert "<Picture 1> is not referenced." in errors
    assert "<Picture 2> is not referenced." in errors


def test_h3_r2v_accepts_video_reference_label():
    body = " ".join(["reference detail"] * 60)
    text = "\n".join(f"{field} [Shot 1] <Video 1> {body}" for field in H3_REFERENCE_FIELDS)
    errors = validate_output(H3_R2V, text, 1, ("<Video 1>",))
    assert "<Video 1> is not referenced." not in errors


def test_krea_rejects_known_bad_prompt_tokens():
    text = "A photorealistic portrait with the sound of rain " + "visible detail " * 30
    errors = validate_output(KREA_T2I, text, 0)
    assert any("photorealistic" in error for error in errors)
    assert any("visible content" in error for error in errors)


def test_h3_rejects_timestamp_on_first_shot():
    body = " ".join(["visible motion and synchronized sound"] * 30)
    text = "\n".join(
        [
            f"{H3_BASE_FIELDS[0]} [Shot 1] At 00:00.000, {body}",
            f"{H3_BASE_FIELDS[1]} {body}",
            f"{H3_BASE_FIELDS[2]} N/A",
        ]
    )
    errors = validate_output("MiniMax H3 — Text to Video", text, 0)
    assert "Shot 1 must not have a timestamp." in errors


def test_h3_requires_first_shot_label():
    body = " ".join(["visible motion and synchronized sound"] * 30)
    text = "\n".join(
        [
            f"{H3_BASE_FIELDS[0]} {body}",
            f"{H3_BASE_FIELDS[1]} {body}",
            f"{H3_BASE_FIELDS[2]} N/A",
        ]
    )
    errors = validate_output("MiniMax H3 — Text to Video", text, 0)
    assert "The timeline must begin with [Shot 1]." in errors


def test_h3_normalizes_first_shot_zero_timestamp():
    text = "integrated_multimodal_description: [Shot 1] At 00:00.000, opening frame"
    normalized = normalize_output("MiniMax H3 — Text to Video", text)
    assert normalized == "integrated_multimodal_description: [Shot 1] opening frame"


def test_user_prompt_names_each_media_group_with_h3_tags():
    prompt = build_user_prompt(H3_R2V, "make a scene", 8, 2, video_count=1, audio_count=2)
    assert "<Picture 1>, <Picture 2>, <Video 1>, <Audio 1>, <Audio 2>" in prompt
