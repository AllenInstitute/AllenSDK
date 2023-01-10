import warnings


class OphysMixin:
    """A mixin class for ophys project data"""
    def __init__(self):
        # If we're in the state of combining behavior and ophys data
        if 'date_of_acquisition_behavior' in self._df and \
                'date_of_acquisition_ophys' in self._df:

            # Prioritize ophys_date_of_acquisition
            self._df['date_of_acquisition'] = \
                self._df['date_of_acquisition_ophys']
            self._df = self._df.drop(
                ['date_of_acquisition_behavior',
                 'date_of_acquisition_ophys'], axis=1)
        self._clean_up_project_code()

    def _clean_up_project_code(self):
        """Remove duplicate project_code columns from the data frames. This is
        as depending on the table we get the project_code either through the
        behavior_sessions or ophys_sessions tables.

        Additionally test that the values in the columns are identical.
        """

        if 'project_code_behavior' in self._df and \
                'project_code_ophys' in self._df:

            if (self._df['project_code_ophys'].isna()).sum() == 0:
                # If there are no missing ophys_session_ids for the table then
                # we are loading of the ophys tables and should be able to
                # compare the ids directly to assure ourselves that the
                # project ids are the same between ophys and behavior sessions.
                if not self._df['project_code_ophys'].equals(
                        self._df['project_code_behavior']):
                    warnings.warn("BehaviorSession and OphysSession "
                                  "project_code's do not agree. This is "
                                  "likely due to issues with the data in "
                                  "LIMS. Using OphysSession project_code.")
                self._df['project_code'] = self._df['project_code_ophys']
            else:
                # If there are missing ophys_session_ids for the table then
                # we are loading of the behavior table first will need to mask
                # to only the sessions that have ophys_session_ids before
                # comparing project_codes.
                has_ophys_session_mask = ~self._df['project_code_ophys'].isna()
                if not self._df['project_code_behavior'][
                        has_ophys_session_mask].equals(
                            self._df['project_code_ophys'][
                                has_ophys_session_mask]):
                    warnings.warn("BehaviorSession and OphysSession "
                                  "project_codes do not agree. This is likely "
                                  "due to issues with the data in LIMS. Using "
                                  "BehaviorSession project_code.")
                self._df['project_code'] = self._df['project_code_behavior']
            self._df = self._df.drop(
                ['project_code_ophys',  'project_code_behavior'],
                axis=1)
