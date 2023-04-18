import pandas as pd


def order_metadata_table_columns(input_df: pd.DataFrame) -> pd.DataFrame:
    """Return the data frame but with columns ordered.

    Parameters
    ----------
    input_df : pandas.DataFrame
        Data frame with columns to be ordered.

    Returns
    -------
    output_df : pandas.DataFrame
        DataFrame the same as the input but with columns reordered.
    """
    column_order = [
        'behavior_session_id', 'ophys_session_id', 'ophys_container_id',
        'mouse_id', 'indicator', 'full_genotype',  'driver_line', 'cre_line',
        'reporter_line', 'sex', 'age_in_days',
        'imaging_depth', 'targeted_structure', 'targeted_imaging_depth',
        'imaging_plane_group',
        'project_code', 'session_type', 'session_number', 'image_set',
        'behavior_type', 'passive',  'experience_level',
        'prior_exposures_to_session_type', 'prior_exposures_to_image_set',
        'prior_exposures_to_omissions',
        'date_of_acquisition', 'equipment_name', 'published_at',
        'isi_experiment_id',
    ]
    # Use only columns that are in the input dataframe's columns.
    pruned_order = []
    for col in column_order:
        if col in input_df.columns:
            pruned_order.append(col)
    # Get the full list of columns in the data frame with our ordered columns
    # first.
    pruned_order.extend(
        list(set(input_df.columns).difference(set(pruned_order))))
    return input_df[pruned_order]
