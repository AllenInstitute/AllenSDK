import pandas as pd
from typing import Iterable

from allensdk.brain_observatory.behavior.behavior_project_cache.project_apis.abcs import (  # noqa: E501
    BehaviorProjectBase,
)
from allensdk.brain_observatory.behavior.behavior_session import (
    BehaviorSession,
)
from allensdk.brain_observatory.behavior.behavior_ophys_experiment import (
    BehaviorOphysExperiment,
)
from allensdk.core.utilities import literal_col_eval
from allensdk.brain_observatory.behavior.behavior_project_cache.project_apis.data_io.project_cloud_api_base import (  # noqa: E501
    ProjectCloudApiBase,
)

COL_EVAL_LIST = ["ophys_experiment_id", "ophys_container_id", "driver_line"]


class BehaviorProjectCloudApi(BehaviorProjectBase, ProjectCloudApiBase):

    MANIFEST_COMPATIBILITY = ["1.0.0", "2.0.0"]

    def _load_manifest_tables(self):

        expected_metadata = set(
            [
                "behavior_session_table",
                "ophys_session_table",
                "ophys_experiment_table",
                "ophys_cells_table",
            ]
        )

        cache_metadata = set(self.cache._manifest.metadata_file_names)

        if cache_metadata != expected_metadata:
            raise RuntimeError(
                "expected S3CloudCache object to have "
                f"metadata file names: {expected_metadata} "
                f"but it has {cache_metadata}"
            )

        self._get_ophys_session_table()
        self._get_behavior_session_table()
        self._get_ophys_experiment_table()
        self._get_ophys_cells_table()

    def get_behavior_session(
        self, behavior_session_id: int
    ) -> BehaviorSession:
        """get a BehaviorSession by specifying behavior_session_id

        Parameters
        ----------
        behavior_session_id: int
            the id of the behavior_session

        Returns
        -------
        BehaviorSession

        Notes
        -----
        entries in the _behavior_session_table represent
        (1) ophys_sessions which have a many-to-one mapping between nwb files
        and behavior sessions. (file_id is NaN)
        AND
        (2) behavior only sessions, which have a one-to-one mapping with
        nwb files. (file_id is not Nan)
        In the case of (1) this method returns an object which is just behavior
        data which is shared by all experiments in 1 session. This is extracted
        from the nwb file for the first-listed ophys_experiment.

        """
        row = self._behavior_session_table.query(
            f"behavior_session_id=={behavior_session_id}"
        )
        if row.shape[0] != 1:
            raise RuntimeError(
                "The behavior_session_table should have "
                "1 and only 1 entry for a given "
                "behavior_session_id. For "
                f"{behavior_session_id} "
                f" there are {row.shape[0]} entries."
            )
        row = row.squeeze()
        has_file_id = not pd.isna(row[self.cache.file_id_column])
        if not has_file_id:
            oeid = row.ophys_experiment_id[0]
            row = self._ophys_experiment_table.query(f"index=={oeid}")
        file_id = str(int(row[self.cache.file_id_column]))
        data_path = self._get_data_path(file_id=file_id)
        return BehaviorSession.from_nwb_path(nwb_path=str(data_path))

    def get_behavior_ophys_experiment(
        self, ophys_experiment_id: int
    ) -> BehaviorOphysExperiment:
        """get a BehaviorOphysExperiment by specifying ophys_experiment_id

        Parameters
        ----------
        ophys_experiment_id: int
            the id of the ophys_experiment

        Returns
        -------
        BehaviorOphysExperiment

        """
        row = self._ophys_experiment_table.query(
            f"index=={ophys_experiment_id}"
        )
        if row.shape[0] != 1:
            raise RuntimeError(
                "The behavior_ophys_experiment_table should "
                "have 1 and only 1 entry for a given "
                f"ophys_experiment_id. For "
                f"{ophys_experiment_id} "
                f" there are {row.shape[0]} entries."
            )
        file_id = str(int(row[self.cache.file_id_column]))
        data_path = self._get_data_path(file_id=file_id)
        return BehaviorOphysExperiment.from_nwb_path(str(data_path))

    def _get_ophys_session_table(self):
        session_table_path = self._get_metadata_path(
            fname="ophys_session_table"
        )
        df = literal_col_eval(
            pd.read_csv(session_table_path, dtype={"mouse_id": str}),
            columns=COL_EVAL_LIST,
        )
        df["date_of_acquisition"] = pd.to_datetime(df["date_of_acquisition"])
        self._ophys_session_table = df.set_index("ophys_session_id")

    def get_ophys_session_table(self) -> pd.DataFrame:
        """Return a pd.Dataframe table summarizing ophys_sessions
        and associated metadata.

        Notes
        -----
        - Each entry in this table represents the metadata of an ophys_session.
        Link to nwb-hosted files in the cache is had via the
        'ophys_experiment_id' column (can be a list)
        and experiment_table
        """
        return self._ophys_session_table

    def _get_behavior_session_table(self):
        session_table_path = self._get_metadata_path(
            fname="behavior_session_table"
        )
        df = literal_col_eval(
            pd.read_csv(session_table_path, dtype={"mouse_id": str}),
            columns=COL_EVAL_LIST,
        )
        df["date_of_acquisition"] = pd.to_datetime(df["date_of_acquisition"])

        self._behavior_session_table = df.set_index("behavior_session_id")

    def get_behavior_session_table(self) -> pd.DataFrame:
        """Return a pd.Dataframe table with both behavior-only
        (BehaviorSession) and with-ophys (BehaviorOphysExperiment)
        sessions as entries.

        Notes
        -----
        - In the first case, provides a critical mapping of
        behavior_session_id to file_id, which the cache uses to find the
        nwb path in cache.
        - In the second case, provides a critical mapping of
        behavior_session_id to a list of ophys_experiment_id(s)
        which can be used to find file_id mappings in ophys_experiment_table
        see method get_behavior_session()
        """
        return self._behavior_session_table

    def _get_ophys_experiment_table(self):
        experiment_table_path = self._get_metadata_path(
            fname="ophys_experiment_table"
        )
        df = literal_col_eval(
            pd.read_csv(experiment_table_path, dtype={"mouse_id": str}),
            columns=COL_EVAL_LIST,
        )
        df["date_of_acquisition"] = pd.to_datetime(df["date_of_acquisition"])

        self._ophys_experiment_table = df.set_index("ophys_experiment_id")

    def _get_ophys_cells_table(self):
        ophys_cells_table_path = self._get_metadata_path(
            fname="ophys_cells_table"
        )
        df = literal_col_eval(
            pd.read_csv(ophys_cells_table_path), columns=COL_EVAL_LIST
        )
        # NaN's for invalid cells force this to float, push to int
        df["cell_specimen_id"] = pd.array(
            df["cell_specimen_id"], dtype="Int64"
        )
        self._ophys_cells_table = df.set_index("cell_roi_id")

    def get_ophys_cells_table(self):
        return self._ophys_cells_table

    def get_ophys_experiment_table(self):
        """returns a pd.DataFrame where each entry has a 1-to-1
        relation with an ophys experiment (i.e. imaging plane)

        Notes
        -----
        - the file_id column allows the underlying cache to link
        this table to a cache-hosted NWB file. There is a 1-to-1
        relation between nwb files and ophy experiments. See method
        get_behavior_ophys_experiment()
        """
        return self._ophys_experiment_table

    def get_natural_movie_template(self, number: int) -> Iterable[bytes]:
        """Download a template for the natural movie stimulus. This is the
        actual movie that was shown during the recording session.
        :param number: identifier for this scene
        :type number: int
        :returns: An iterable yielding an npy file as bytes
        """
        raise NotImplementedError()

    def get_natural_scene_template(self, number: int) -> Iterable[bytes]:
        """Download a template for the natural scene stimulus. This is the
        actual image that was shown during the recording session.
        :param number: idenfifier for this movie (note that this is an int,
            so to get the template for natural_movie_three should pass 3)
        :type number: int
        :returns: iterable yielding a tiff file as bytes
        """
        raise NotImplementedError()
