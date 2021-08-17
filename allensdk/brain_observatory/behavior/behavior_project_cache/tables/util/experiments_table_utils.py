import pandas as pd


def add_experience_level_to_experiment_table(
        experiments_table: pd.DataFrame) -> pd.DataFrame:
    """
    adds a column to ophys_experiment_table that contains a string
    indicating whether a session had exposure level of Familiar,
    Novel 1, or Novel >1, based on session number and
    prior_exposure_to_image_set

    Parameters
    ----------
    experiments_table: pd.DataFrame

    Returns
    -------
    experiments_table: pd.DataFrame

    Notes
    -----
    Does not change the input DataFrame in-place
    """
    # Ported from
    # https://github.com/AllenInstitute/visual_behavior_analysis/
    # blob/master/visual_behavior/data_access/utilities.py#L1307

    # do not modify in place
    experiments_table = experiments_table.copy(deep=True)

    # add experience_level column with strings indicating relevant conditions
    experiments_table['experience_level'] = 'None'

    session_123 = experiments_table.session_number.isin([1, 2, 3])
    familiar_indices = experiments_table[session_123].index.values

    experiments_table.at[familiar_indices, 'experience_level'] = 'Familiar'

    session_4 = (experiments_table.session_number == 4)
    zero_prior_exp = (experiments_table.prior_exposures_to_image_set == 0)

    novel_indices = experiments_table[
                          session_4
                          & zero_prior_exp].index.values

    experiments_table.at[novel_indices,
                         'experience_level'] = 'Novel 1'

    session_456 = experiments_table.session_number.isin([4, 5, 6])
    nonzero_prior_exp = (experiments_table.prior_exposures_to_image_set != 0)
    novel_gt_1_indices = experiments_table[
                             session_456
                             & nonzero_prior_exp].index.values

    experiments_table.at[novel_gt_1_indices,
                         'experience_level'] = 'Novel >1'

    return experiments_table


def add_passive_flag_to_ophys_experiment_table(
        experiments_table: pd.DataFrame) -> pd.DataFrame:
    """
    adds a column to ophys_experiment_table that contains a Boolean
    indicating whether a session was passive or not based on session
    number

    Parameters
    ----------
    experiments_table: pd.DataFrame

    Returns
    -------
    experiments_table: pd.DataFrame

    Note
    ----
    Does not change the input DataFrame in-place
    """

    # Ported from
    # https://github.com/AllenInstitute/visual_behavior_analysis/blob/master
    # /visual_behavior/data_access/utilities.py#L1344

    experiments_table = experiments_table.copy(deep=True)

    experiments_table['passive'] = False

    session_25 = experiments_table.session_number.isin([2, 5])
    passive_indices = experiments_table[session_25].index.values
    experiments_table.at[passive_indices, 'passive'] = True

    return experiments_table


def add_image_set_to_experiment_table(
        experiments_table: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a column 'image_set' to the experiment_table, determined based
    on the image set listed in the session_type column string

    Parameters
    ----------
    experiments_table: pd.DataFrame

    Returns
    --------
    experiments_table: pd.DataFrame

    Notes
    -----
    Does not alter the input DataFrame in-place
    """

    # Ported from
    # https://github.com/AllenInstitute/visual_behavior_analysis/blob/master/
    # visual_behavior/data_access/utilities.py#L1403

    experiments_table = experiments_table.copy(deep=True)

    experiments_table['image_set'] = [
            session_type[15]
            if len(session_type) > 15 else 'N/A'
            for session_type
            in experiments_table.session_type.values.astype(str)]
    return experiments_table
