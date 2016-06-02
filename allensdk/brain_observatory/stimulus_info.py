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

# some handles for stimulus types
DRIFTING_GRATINGS = 'drifting_gratings'
STATIC_GRATINGS = 'static_gratings'
NATURAL_MOVIE_ONE = 'natural_movie_one'
NATURAL_MOVIE_TWO = 'natural_movie_two'
NATURAL_MOVIE_THREE = 'natural_movie_three'
NATURAL_SCENES = 'natural_scenes'
LOCALLY_SPARSE_NOISE = 'locally_sparse_noise'
SPONTANEOUS_ACTIVITY = 'spontaneous'

# handles for stimulus names
THREE_SESSION_A = 'three_session_A'
THREE_SESSION_B = 'three_session_B'
THREE_SESSION_C = 'three_session_C'

SESSION_STIMULUS_MAP = {
    THREE_SESSION_A: [ DRIFTING_GRATINGS, NATURAL_MOVIE_ONE, NATURAL_MOVIE_THREE, SPONTANEOUS_ACTIVITY ],
    THREE_SESSION_B: [ STATIC_GRATINGS, NATURAL_SCENES, NATURAL_MOVIE_ONE, SPONTANEOUS_ACTIVITY ],
    THREE_SESSION_C: [ LOCALLY_SPARSE_NOISE, NATURAL_MOVIE_ONE, NATURAL_MOVIE_TWO, SPONTANEOUS_ACTIVITY ]
    }

def stimuli_in_session(session):
    """ Return a list what stimuli are available in a given session. 
    
    Parameters
    ----------
    session: string
        Must be one of: [stimulus_info.THREE_SESSION_A, stimulus_info.THREE_SESSION_B, stimulus_info.THREE_SESSION_C]
    """
    return SESSION_STIMULUS_MAP[session]

def all_stimuli():
    """ Return a list of all stimuli in the data set """
    return set([ v for k,vl in SESSION_STIMULUS_MAP.iteritems() for v in vl ])
