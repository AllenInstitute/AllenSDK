import pandas as pd

from allensdk.brain_observatory.behavior.behavior_project_cache.\
    project_apis.data_io.project_cloud_api_base import ProjectCloudApiBase  # noqa: E501

from allensdk.brain_observatory.behavior.behavior_session import (
    BehaviorSession)

from allensdk.brain_observatory.ecephys.behavior_ecephys_session \
    import BehaviorEcephysSession


class VisualBehaviorNeuropixelsProjectCloudApi(ProjectCloudApiBase):

    MANIFEST_COMPATIBILITY = ["0.1.0", "10.0.0"]

    def _load_manifest_tables(self):

        self._get_ecephys_session_table()
        self._get_behavior_session_table()
        self._get_unit_table()
        self._get_probe_table()
        self._get_channel_table()

    def get_behavior_session(
            self, behavior_session_id: int) -> BehaviorSession:
        """
        Since we are not releasing behavior-only NWB files
        with the VBN June 2022 release, the behavior sesion data
        is obtained from the behavior_ecephys_session NWB file
        based on the associated behavior_ecephys_session_id

        Parameters
        ----------
        behavior_session_id: int
            the id of the behavior_session

        Returns
        -------
        BehaviorSession

        Notes
        -----
        behavior session does not include file_id.
        The file id is accessed via ecephys_session_id key
        from the ecephys_session_table
        """
        row = self._behavior_session_table.query(
                f"behavior_session_id=={behavior_session_id}")
        if row.shape[0] != 1:
            raise RuntimeError("The behavior_session_table should have "
                               "1 and only 1 entry for a given "
                               "behavior_session_id. For "
                               f"{behavior_session_id} "
                               f" there are {row.shape[0]} entries.")
        row = row.squeeze()
        ecephys_session_id = int(row.ecephys_session_id)

        row = self._ecephys_session_table.query(f"index=={ecephys_session_id}")

        if len(row) == 0:
            raise RuntimeError(f"ecephys_session: {ecephys_session_id} "
                               f"corresponding to "
                               f"behavior_session: {behavior_session_id} "
                               f"does not exist in the ecephys_session_table ")

        file_id = str(int(row[self.cache.file_id_column]))
        data_path = self._get_data_path(file_id=file_id)

        return BehaviorSession.from_nwb_path(str(data_path))

    def get_ecephys_session(
        self,
        ecephys_session_id: int
    ) -> BehaviorEcephysSession:

        """get a BehaviorEcephysSession by specifying ecephys_session_id

        Parameters
        ----------
        ecephys_session_id: int
            the id of the ecephys session

        Returns
        -------
        BehaviorEcephysSession

        """
        row = self._ecephys_session_table.query(
                f"index=={ecephys_session_id}")
        if row.shape[0] != 1:
            raise RuntimeError("The behavior_ecephys_session_table should "
                               "have 1 and only 1 entry for a given "
                               f"ecephys_session_id. For "
                               f"{ecephys_session_id} "
                               f" there are {row.shape[0]} entries.")
        file_id = str(int(row[self.cache.file_id_column]))
        data_path = self._get_data_path(file_id=file_id)
        return BehaviorEcephysSession.from_nwb_path(
            str(data_path))

    def _get_ecephys_session_table(self):
        session_table_path = self._get_metadata_path(
            fname="ecephys_sessions")
        df = pd.read_csv(session_table_path)
        self._ecephys_session_table = df.set_index("ecephys_session_id")

    def get_ecephys_session_table(self) -> pd.DataFrame:
        """Return a pd.Dataframe table summarizing ecephys_sessions
        and associated metadata.

        """
        return self._ecephys_session_table

    def _get_behavior_session_table(self):
        session_table_path = self._get_metadata_path(
            fname='behavior_sessions')
        df = pd.read_csv(session_table_path)
        self._behavior_session_table = df.set_index("behavior_session_id")

    def get_behavior_session_table(self) -> pd.DataFrame:
        return self._behavior_session_table

    def _get_probe_table(self):
        probe_table_path = self._get_metadata_path(
            fname="probes")
        df = pd.read_csv(probe_table_path)
        self._probe_table = df.set_index("ecephys_probe_id")

    def get_probe_table(self):
        return self._probe_table

    def _get_unit_table(self):
        unit_table_path = self._get_metadata_path(
            fname="units")
        df = pd.read_csv(unit_table_path)
        self._unit_table = df.set_index("unit_id")

    def get_unit_table(self):
        return self._unit_table

    def _get_channel_table(self):
        channel_table_path = self._get_metadata_path(
            fname="channels")
        df = pd.read_csv(channel_table_path)
        self._channel_table = df.set_index("ecephys_channel_id")

    def get_channel_table(self):
        return self._channel_table
