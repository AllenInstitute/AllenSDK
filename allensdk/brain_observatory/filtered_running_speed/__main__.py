from allensdk.brain_observatory.\
    filtered_running_speed.filtered_running_speed import (
        FilteredRunningSpeed
    )


DEFAULT_ZSCORE_THRESHOLD = 10.0
USE_LOWPASS_FILTER = True

if __name__ == "__main__":
    filtered_running_speed = FilteredRunningSpeed(
        DEFAULT_ZSCORE_THRESHOLD,
        USE_LOWPASS_FILTER
    )
