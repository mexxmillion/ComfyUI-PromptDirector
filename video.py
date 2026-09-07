from __future__ import annotations


def evenly_spaced_indices(frame_count: int, sample_count: int) -> list[int]:
    if frame_count < 1:
        raise ValueError("The input video contains no frames.")
    count = min(frame_count, max(1, sample_count))
    if count == 1:
        return [0]
    return [round(index * (frame_count - 1) / (count - 1)) for index in range(count)]


def sample_video_frames(frames, sample_count: int, video_info: dict | None = None):
    if frames.ndim != 4:
        raise ValueError(f"Expected a VHS IMAGE batch in BHWC layout, received shape {tuple(frames.shape)}.")
    indices = evenly_spaced_indices(len(frames), sample_count)
    sampled = frames[indices]

    fps = 0.0
    if isinstance(video_info, dict):
        fps = float(video_info.get("loaded_fps") or 0.0)
    if fps > 0:
        points = ", ".join(f"Frame {position} = +{frame_index / fps:.3f}s" for position, frame_index in enumerate(indices, 1))
        duration = (len(frames) - 1) / fps
        context = (
            f"The attached images are {len(indices)} chronological frames sampled from one video reference "
            f"spanning approximately {duration:.3f}s. {points}. Treat them as one continuous <Video 1>, not as separate pictures."
        )
    else:
        points = ", ".join(f"Frame {position} = source frame {frame_index}" for position, frame_index in enumerate(indices, 1))
        context = (
            f"The attached images are {len(indices)} chronological frames sampled from one video reference. "
            f"{points}. Treat them as one continuous <Video 1>, not as separate pictures."
        )
    return sampled, context, len(indices)
