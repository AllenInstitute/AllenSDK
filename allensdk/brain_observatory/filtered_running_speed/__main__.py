from allensdk.brain_observatory.\
    filtered_running_speed.multi_stimulus_running_speed import (
        MultiStimulusRunningSpeed
    )


DEFAULT_ZSCORE_THRESHOLD = 10.0
USE_LOWPASS_FILTER = True

if __name__ == "__main__":
    multi_stimulus_running_speed = MultiStimulusRunningSpeed()
    multi_stimulus_running_speed.process()
