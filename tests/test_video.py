from video import evenly_spaced_indices, sample_video_frames


class Frames:
    ndim = 4
    shape = (10, 64, 64, 3)

    def __len__(self):
        return 10

    def __getitem__(self, indices):
        return indices


def test_evenly_spaced_indices_include_both_ends():
    assert evenly_spaced_indices(10, 4) == [0, 3, 6, 9]
    assert evenly_spaced_indices(3, 8) == [0, 1, 2]


def test_video_sampler_builds_timestamp_context():
    sampled, context, count = sample_video_frames(Frames(), 4, {"loaded_fps": 3.0})
    assert sampled == [0, 3, 6, 9]
    assert count == 4
    assert "Frame 4 = +3.000s" in context
    assert "one continuous <Video 1>" in context
