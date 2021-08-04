import numpy as np
from typing import Optional

from allensdk.brain_observatory.behavior.metadata.behavior_metadata import \
    BehaviorMetadata
from allensdk.brain_observatory.behavior.session_apis.abcs.\
    data_extractor_base.behavior_ophys_data_extractor_base import \
    BehaviorOphysDataExtractorBase


class BehaviorOphysMetadata(BehaviorMetadata):
    """Container class for behavior ophys metadata"""
    def __init__(self, extractor: BehaviorOphysDataExtractorBase,
                 stimulus_timestamps: np.ndarray,
                 ophys_timestamps: np.ndarray,
                 behavior_stimulus_file: dict):

        super().__init__(extractor=extractor,
                         stimulus_timestamps=stimulus_timestamps,
                         behavior_stimulus_file=behavior_stimulus_file)
        self._extractor = extractor
        self._ophys_timestamps = ophys_timestamps

        # project_code needs to be excluded from comparison
        # since it's only exposed internally
        self._exclude_from_equals = {'project_code'}

    @property
    def indicator(self) -> Optional[str]:
        """Parses indicator from reporter"""
        reporter_line = self.reporter_line
        return self.parse_indicator(reporter_line=reporter_line, warn=True)

    @property
    def emission_lambda(self) -> float:
        return 520.0

    @property
    def excitation_lambda(self) -> float:
        return 910.0

    # TODO rename to ophys_container_id
    @property
    def experiment_container_id(self) -> int:
        return self._extractor.get_ophys_container_id()

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
