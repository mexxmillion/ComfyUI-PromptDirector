from prompts import (
    H3_BASE_FIELDS,
    H3_I2V,
    H3_R2V,
    H3_REFERENCE_FIELDS,
    I2V_PREFIX,
    KREA_EDIT,
    KREA_T2I,
    MODE_SPECS,
    build_system_prompt,
    clean_output,
    normalize_output,
    validate_output,
)


def test_exact_five_modes():
    assert list(MODE_SPECS) == [
        KREA_T2I,
        KREA_EDIT,
        "MiniMax H3 — Text to Video",
        H3_I2V,
        H3_R2V,
    ]


def test_krea_system_prompt_is_mode_specific():
    prompt = build_system_prompt(KREA_EDIT, "Faithful")
    assert "MODE: Image Edit" in prompt
    assert "Preserve every unspecified" in prompt
    assert "intent-locked" in prompt


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
