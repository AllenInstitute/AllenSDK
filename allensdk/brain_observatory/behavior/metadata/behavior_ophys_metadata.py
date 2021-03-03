import numpy as np
from typing import Optional, Any

from allensdk.brain_observatory.behavior.metadata.behavior_metadata import \
    BehaviorMetadata


class BehaviorOphysMetadata(BehaviorMetadata):
    """Container class for behavior ophys metadata"""
    def __init__(self, extractor: Any,
                 stimulus_timestamps: np.ndarray,
                 ophys_timestamps: np.ndarray,
                 behavior_stimulus_file: dict):

        """Note: cannot properly type extractor due to circular dependency
        between extractor and transformer.
        TODO fix circular dependency and update type"""

        super().__init__(extractor=extractor,
                         stimulus_timestamps=stimulus_timestamps,
                         behavior_stimulus_file=behavior_stimulus_file)
        self._extractor = extractor
        self._ophys_timestamps = ophys_timestamps

        # project_code needs to be excluded from comparison
        # since it's only exposed internally
        self._exclude_from_equals = {'project_code'}

    @property
    def emission_lambda(self) -> float:
        return 520.0

    @property
    def excitation_lambda(self) -> float:
        return 910.0

    @property
    def experiment_container_id(self) -> int:
        return self._extractor.get_experiment_container_id()

    @property
    def field_of_view_height(self) -> int:
        return self._extractor.get_field_of_view_shape()['height']

    @property
    def field_of_view_width(self) -> int:
        return self._extractor.get_field_of_view_shape()['width']

    @property
    def imaging_depth(self) -> int:
        return self._extractor.get_imaging_depth()

    @property
    def imaging_plane_group(self) -> Optional[int]:
        return self._extractor.get_imaging_plane_group()

    @property
    def imaging_plane_group_count(self) -> int:
        return self._extractor.get_plane_group_count()

    @property
    def indicator(self) -> str:
        return 'GCAMP6f'

    @property
    def ophys_experiment_id(self) -> int:
        return self._extractor.get_ophys_experiment_id()

    @property
    def ophys_frame_rate(self) -> float:
        return self._get_frame_rate(timestamps=self._ophys_timestamps)

    @property
    def ophys_session_id(self) -> int:
        return self._extractor.get_ophys_session_id()

    @property
    def project_code(self) -> Optional[str]:
        try:
            project_code = self._extractor.get_project_code()
        except NotImplementedError:
            # Project code only returned by LIMS
            project_code = None
        return project_code

    @property
    def targeted_structure(self) -> str:
        return self._extractor.get_targeted_structure()

    def to_dict(self) -> dict:
        """Returns dict representation of all properties in class"""
        vars_ = vars(BehaviorOphysMetadata)
        d = self._get_properties(vars_=vars_)
        return {**super().to_dict(), **d}





