from typing import Optional, List

import pandas as pd

from allensdk.brain_observatory.behavior.behavior_project_cache.tables\
    .ophys_mixin import \
    OphysMixin
from allensdk.brain_observatory.behavior.behavior_project_cache.tables\
    .project_table import \
    ProjectTable
from allensdk.brain_observatory.behavior.behavior_project_cache.tables\
    .util.experiments_table_utils import (
        add_experience_level_to_experiment_table,
        add_passive_flag_to_ophys_experiment_table,
        add_image_set_to_experiment_table)


class ExperimentsTable(ProjectTable, OphysMixin):
    """Class for storing and manipulating project-level data
    at the behavior-ophys experiment level"""
    def __init__(self, df: pd.DataFrame,
                 suppress: Optional[List[str]] = None):
        """
        Parameters
        ----------
        df
            The behavior-ophys experiment-level data
        suppress
            columns to drop from table
        """
        ProjectTable.__init__(self, df=df, suppress=suppress)
        OphysMixin.__init__(self)
        self.final_processing()

    def postprocess_additional(self):
        pass

    def final_processing(self):
        # This method is necessary because self.post_process_additional()
        # is called by the ProjectTable.__init__(), which is called
        # before OphysMixin.__init__(). OphysMixin.__init__() joins
        # some of the Behavior and Ophys columns into sigle, session-wide
        # columns, which the functions below must access (specifically,
        # OphysMixin.__init__() joins session_type_behavior and
        # session_type_ophys into session_type, which is the column
        # that add_image_set_to_experiment acts on. A future ticket
        # should revisit the workflow of these classes to make it
        # possible for the function calls below to be incorporated
        # into post_process_additional

        self._df = add_experience_level_to_experiment_table(self._df)
        self._df = add_passive_flag_to_ophys_experiment_table(self._df)
        self._df = add_image_set_to_experiment_table(self._df)
