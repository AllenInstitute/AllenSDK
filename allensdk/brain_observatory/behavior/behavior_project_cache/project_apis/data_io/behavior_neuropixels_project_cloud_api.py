import numpy as np
import pandas as pd
from allensdk.brain_observatory.behavior.behavior_project_cache.project_apis.data_io.project_cloud_api_base import (  # noqa: E501
    ProjectCloudApiBase,
)
from allensdk.brain_observatory.behavior.behavior_session import (
    BehaviorSession,
)
from allensdk.brain_observatory.ecephys.behavior_ecephys_session import (
    BehaviorEcephysSession,
)
from allensdk.brain_observatory.ecephys._probe import ProbeWithLFPMeta


class VisualBehaviorNeuropixelsProjectCloudApi(ProjectCloudApiBase):

    MANIFEST_COMPATIBILITY = ["0.1.0", "10.0.0"]

    def _load_manifest_tables(self):

        self._get_ecephys_session_table()
        self._get_behavior_session_table()
        self._get_unit_table()
        self._get_probe_table()
        self._get_channel_table()

    def get_behavior_session(
        self, behavior_session_id: int
    ) -> BehaviorSession:
        """
        Retrieve behavior session data from either the released behavior
        only nwb or the behavior side of the released ecephys data.

        Checks first if the session is behavior only and if so returns the
        behavior sessions. Failing this, we check if the the behavior session
        has an associated ecephys session and return that if it exists.

        Parameters
        ----------
        behavior_session_id: int
            the id of the behavior_session

        Returns
        -------
        BehaviorSession
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
        ecephys_session_id = row.ecephys_session_id
        # If a file_id for the behavior session is not set, attempt to load
        # an associated ecephys session.
        if row[self.cache.file_id_column] < 0 or np.isnan(
            row[self.cache.file_id_column]
        ):
            row = self._ecephys_session_table.query(
                f"index=={ecephys_session_id}"
            )

        if len(row) == 0:
            raise RuntimeError(
                f"behavior_session: {behavior_session_id} "
                f"corresponding to "
                f"ecephys_session: {ecephys_session_id}"
                f"does not exist in the behavior_session  "
                "or ecephys_session tables."
            )

        file_id = str(int(row[self.cache.file_id_column]))
        data_path = self._get_data_path(file_id=file_id)

        return BehaviorSession.from_nwb_path(str(data_path))

    def get_ecephys_session(
        self, ecephys_session_id: int
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
        session_meta = self._ecephys_session_table.query(
            f"index=={ecephys_session_id}"
        )
        probes_meta = self._probe_table[
            (self._probe_table["ecephys_session_id"] == ecephys_session_id)
            & (self._probe_table["has_lfp_data"])
        ]
        if session_meta.shape[0] != 1:
            raise RuntimeError(
                "The behavior_ecephys_session_table should "
                "have 1 and only 1 entry for a given "
                f"ecephys_session_id. For "
                f"{ecephys_session_id} "
                f" there are {session_meta.shape[0]} entries."
            )
        session_file_id = str(int(session_meta[self.cache.file_id_column]))
        session_data_path = self._get_data_path(file_id=session_file_id)

        def make_lazy_load_filepath_function(file_id):
            """Due to late binding closure. See:
            https://docs.python-guide.org/writing/gotchas/
            #late-binding-closures"""

            def f():
                return self._get_data_path(file_id=file_id)

            return f

        # Backwards compatibility check for VBN data that doesn't contain
        # the LFP dataset.
        has_probe_file = self.cache.file_id_column in probes_meta.columns

        if not probes_meta.empty and has_probe_file:
            probe_meta = {
                p.name: ProbeWithLFPMeta(
                    lfp_csd_filepath=make_lazy_load_filepath_function(
                        file_id=str(int(getattr(
                            p, self.cache.file_id_column)))
                        ),
                    lfp_sampling_rate=p.lfp_sampling_rate
                )
                for p in probes_meta.itertuples(index=False)
            }
        else:
            probe_meta = None
        return BehaviorEcephysSession.from_nwb_path(
            str(session_data_path), probe_meta=probe_meta
        )

    def _get_ecephys_session_table(self):
        session_table_path = self._get_metadata_path(fname="ecephys_sessions")
        df = pd.read_csv(session_table_path)
        self._ecephys_session_table = df.set_index("ecephys_session_id")

    def get_ecephys_session_table(self) -> pd.DataFrame:
        """Return a pd.Dataframe table summarizing ecephys_sessions
        and associated metadata.

        """
        return self._ecephys_session_table

    def _get_behavior_session_table(self):
        session_table_path = self._get_metadata_path(fname="behavior_sessions")
        df = pd.read_csv(session_table_path)
        self._behavior_session_table = df.set_index("behavior_session_id")

    def get_behavior_session_table(self) -> pd.DataFrame:
        return self._behavior_session_table

    def _get_probe_table(self):
        probe_table_path = self._get_metadata_path(fname="probes")
        df = pd.read_csv(probe_table_path)
        self._probe_table = df.set_index("ecephys_probe_id")

    def get_probe_table(self):
        return self._probe_table

    def _get_unit_table(self):
        unit_table_path = self._get_metadata_path(fname="units")
        df = pd.read_csv(unit_table_path)
        self._unit_table = df.set_index("unit_id")

    def get_unit_table(self):
        return self._unit_table

    def _get_channel_table(self):
        channel_table_path = self._get_metadata_path(fname="channels")
        df = pd.read_csv(channel_table_path)
        self._channel_table = df.set_index("ecephys_channel_id")

    def get_channel_table(self):
        return self._channel_table
