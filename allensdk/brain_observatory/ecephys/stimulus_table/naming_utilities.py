import re

import pandas as pd
import numpy as np


GABOR_DIAMETER_RE = re.compile(r'gabor_(\d*\.{0,1}\d*)_{0,1}deg(?:_\d+ms){0,1}')
GENERIC_MOVIE_RE = re.compile(r'natural_movie_(?P<number>\d+|one|two|three|four|five|six|seven|eight|nine)(_shuffled){0,1}(_more_repeats){0,1}')
DIGIT_NAMES = {'1': 'one', '2': 'two', '3': 'three', '4': 'four', '5': 'five', '6': 'six', '7': 'seven', '8': 'eight', '9': 'nine'}
SHUFFLED_MOVIE_RE = re.compile(r'natural_movie_shuffled')
NUMERAL_RE = re.compile(r'(?P<number>\d+)')


def add_number_to_shuffled_movie(
    table, 
    natural_movie_re=GENERIC_MOVIE_RE, 
    template_re=SHUFFLED_MOVIE_RE,
    stim_colname='stimulus_name', 
    template='natural_movie_{}_shuffled',
    tmp_colname='__movie_number__'
):
    '''
    '''

    if not table[stim_colname].str.contains(SHUFFLED_MOVIE_RE).any():
        return table
    table = table.copy()

    table[tmp_colname] = table[stim_colname].str.extract(natural_movie_re, expand=True)['number']

    unique_numbers = [item for item in table[tmp_colname].dropna(inplace=False).unique()]
    if len(unique_numbers) != 1:
        raise ValueError(f'unable to uniquely determine a movie number for this session. Candidates: {unique_numbers}')
    movie_number = unique_numbers[0]

    def renamer(row):
        if not isinstance(row[stim_colname], str):
            return row[stim_colname]
        if not template_re.match(row[stim_colname]):
            return row[stim_colname]
        else:
            return template.format(movie_number)

    table[stim_colname] = table.apply(renamer, axis=1)
    table.drop(columns=tmp_colname, inplace=True)
    return table


def standardize_movie_numbers(
    table, 
    movie_re=GENERIC_MOVIE_RE, 
    numeral_re=NUMERAL_RE,
    digit_names=DIGIT_NAMES, 
    stim_colname='stimulus_name'
):
    '''
    '''

    replace = lambda match_obj: digit_names[match_obj['number']]

    movie_rows = table[stim_colname].str.contains(movie_re, na=False)
    table.loc[movie_rows,stim_colname] = table.loc[movie_rows,stim_colname].str.replace(numeral_re, replace)

    return table


def extract_gabor_parameters(
    table, 
    gabor_name='gabor', 
    gabor_diameter_regex=GABOR_DIAMETER_RE, 
    stim_colname='stimulus_name', 
    diameter_colname='diameter'
):
    ''' the gabor_20_deg_250ms stimulus has diameter (a parameter we want) and duration (a parameter encoded already by start and stop times)
    baked into the name. This function splits them out.
    '''

    table[diameter_colname] = table[stim_colname].str.extract(gabor_diameter_regex)
    table[diameter_colname] = table[diameter_colname].astype(float)

    table[stim_colname][~table[diameter_colname].isna()] = gabor_name

    return table


def map_stimulus_names(table, name_map=None, stim_colname='stimulus_name'):
    '''
    '''

    if name_map is None:
        return table

    if None in name_map:
        name_map[np.nan] = name_map[None]

    table[stim_colname] = table[stim_colname].replace(to_replace=name_map, inplace=False)
    return table

