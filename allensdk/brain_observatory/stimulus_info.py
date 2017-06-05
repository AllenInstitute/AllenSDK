# Copyright 2016 Allen Institute for Brain Science
# This file is part of Allen SDK.
#
# Allen SDK is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3 of the License.
#
# Allen SDK is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Allen SDK.  If not, see <http://www.gnu.org/licenses/>.
import six
import matplotlib.colors as mcolors

# some handles for stimulus types
DRIFTING_GRATINGS = 'drifting_gratings'
DRIFTING_GRATINGS_SHORT = 'dg'
DRIFTING_GRATINGS_COLOR = '#a6cee3'

STATIC_GRATINGS = 'static_gratings'
STATIC_GRATINGS_SHORT = 'sg'
STATIC_GRATINGS_COLOR = '#1f78b4'

NATURAL_MOVIE_ONE = 'natural_movie_one'
NATURAL_MOVIE_ONE_SHORT = 'nm1'
NATURAL_MOVIE_ONE_COLOR = '#b2df8a'

NATURAL_MOVIE_TWO = 'natural_movie_two'
NATURAL_MOVIE_TWO_SHORT = 'nm2'
NATURAL_MOVIE_TWO_COLOR = '#33a02c'

NATURAL_MOVIE_THREE = 'natural_movie_three'
NATURAL_MOVIE_THREE_SHORT = 'nm3'
NATURAL_MOVIE_THREE_COLOR = '#fb9a99'

NATURAL_SCENES = 'natural_scenes'
NATURAL_SCENES_SHORT = 'ns'
NATURAL_SCENES_COLOR = '#e31a1c'

LOCALLY_SPARSE_NOISE = 'locally_sparse_noise'
LOCALLY_SPARSE_NOISE_SHORT = 'lsn'
LOCALLY_SPARSE_NOISE_COLOR = '#fdbf6f'

LOCALLY_SPARSE_NOISE_4DEG = 'locally_sparse_noise_4deg'
LOCALLY_SPARSE_NOISE_4DEG_SHORT = 'lsn4'
LOCALLY_SPARSE_NOISE_4DEG_COLOR = '#fdbf6f'

LOCALLY_SPARSE_NOISE_8DEG = 'locally_sparse_noise_8deg'
LOCALLY_SPARSE_NOISE_8DEG_SHORT = 'lsn8'
LOCALLY_SPARSE_NOISE_8DEG_COLOR = '#ff7f00'

SPONTANEOUS_ACTIVITY = 'spontaneous'
SPONTANEOUS_ACTIVITY_SHORT = 'sp'
SPONTANEOUS_ACTIVITY_COLOR = '#cab2d6'

# handles for stimulus names
THREE_SESSION_A = 'three_session_A'
THREE_SESSION_B = 'three_session_B'
THREE_SESSION_C = 'three_session_C'
THREE_SESSION_C2 = 'three_session_C2'

SESSION_STIMULUS_MAP = {
    THREE_SESSION_A: [DRIFTING_GRATINGS, NATURAL_MOVIE_ONE, NATURAL_MOVIE_THREE, SPONTANEOUS_ACTIVITY],
    THREE_SESSION_B: [STATIC_GRATINGS, NATURAL_SCENES, NATURAL_MOVIE_ONE, SPONTANEOUS_ACTIVITY],
    THREE_SESSION_C: [LOCALLY_SPARSE_NOISE, NATURAL_MOVIE_ONE, NATURAL_MOVIE_TWO, SPONTANEOUS_ACTIVITY],
    THREE_SESSION_C2: [LOCALLY_SPARSE_NOISE_4DEG, LOCALLY_SPARSE_NOISE_8DEG, NATURAL_MOVIE_ONE, NATURAL_MOVIE_TWO, SPONTANEOUS_ACTIVITY]
}

LOCALLY_SPARSE_NOISE_DIMENSIONS = {
    LOCALLY_SPARSE_NOISE: [ 16, 28 ],
    LOCALLY_SPARSE_NOISE_4DEG: [ 16, 28 ],    
    LOCALLY_SPARSE_NOISE_8DEG: [ 8, 14 ],
    }

# Note: the "8deg" stimulus is actually 9.3 visual degrees on a side
LOCALLY_SPARSE_NOISE_PIXEL_SIZE = {
    LOCALLY_SPARSE_NOISE: 4.65,
    LOCALLY_SPARSE_NOISE_4DEG: 4.65,
    LOCALLY_SPARSE_NOISE_8DEG: 9.3 
}

def sessions_with_stimulus(stimulus):
    """ Return the names of the sessions that contain a given stimulus. """
    
    sessions = set()
    for session, session_stimuli in six.iteritems(SESSION_STIMULUS_MAP):
        if stimulus in session_stimuli:
            sessions.add(session)

    return sorted(list(sessions))


def stimuli_in_session(session):
    """ Return a list what stimuli are available in a given session.

    Parameters
    ----------
    session: string
        Must be one of: [stimulus_info.THREE_SESSION_A, stimulus_info.THREE_SESSION_B, stimulus_info.THREE_SESSION_C, stimulus_info.THREE_SESSION_C2]
    """
    return SESSION_STIMULUS_MAP[session]


def all_stimuli():
    """ Return a list of all stimuli in the data set """
    return set([v for k, vl in six.iteritems(SESSION_STIMULUS_MAP) for v in vl])
