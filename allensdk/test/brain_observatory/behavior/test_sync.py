from allensdk.brain_observatory.behavior.sync import frame_time_offset


def test_frame_time_offset():
    mock_data = {
        "items": {
            "behavior": {
                "trial_log":
                    [
                        {
                            "events":
                                [
                                    ["trial_start", 2, 1],
                                    ["trial_end", 3, 2],
                                ],
                        },
                        {
                            "events":
                                [
                                    ["trial_start", 4, 3],
                                    ["trial_start", 5, 4],
                                ],
                        },
                    ],
                },
            },
        }
    actual = frame_time_offset(mock_data)
    assert 1. == actual
