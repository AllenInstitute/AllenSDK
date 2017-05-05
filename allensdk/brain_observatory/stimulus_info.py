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
        Must be one of: [stimulus_info.THREE_SESSION_A, stimulus_info.THREE_SESSION_B, stimulus_info.THREE_SESSION_C]
    """
    return SESSION_STIMULUS_MAP[session]


def all_stimuli():
    """ Return a list of all stimuli in the data set """
    return set([v for k, vl in six.iteritems(SESSION_STIMULUS_MAP) for v in vl])

class BinaryIntervalSearchTree(object):

    @staticmethod
    def from_df(input_df):
        search_list = input_df.to_dict('records')



        new_list = []
        for x in search_list:
            if x['start'] == x['end']:
               new_list.append((x['start'], x['end'], x))
            else:
               # -.01 prevents endpoint-overlapping intervals; assigns ties to intervals that start at requested index
               new_list.append((x['start'], x['end'] - .01, x))
        return BinaryIntervalSearchTree(new_list)


    def __init__(self, search_list):
        """Create a binary tree to search for a point within a list of intervals.  Assumes that the intervals are
        non-overlapping.  If two intervals share an endpoint, the left-side wins the tie.

        :param search_list: list of interval tuples; in the tuple, first element is interval start, then interval
        end (inclusive), then the return value for the lookup

        Example:
        bist = BinaryIntervalSearchTree([(0,.5,'A'), (1,2,'B')])
        print bist.search(1.5)
        """

        # Double-check that the list is sorted
        search_list = sorted(search_list, key=lambda x:x[0])

        # Check that the intervals are non-overlapping (except potentially at the end point)
        for x, y in zip(search_list[:-1], search_list[1:]):
            assert x[1] < y[0]


        self.data = {}
        self.add(search_list)

    def add(self, input_list, tmp=None):
        if tmp is None:
            tmp = []

        if len(input_list) == 1:
            self.data[tuple(tmp)] = input_list[0]
        else:
            self.add(input_list[:int(len(input_list)/2)], tmp=tmp+[0])
            self.add(input_list[int(len(input_list)/2):], tmp=tmp+[1])
            self.data[tuple(tmp)] = input_list[int(len(input_list)/2)-1]

    def search(self, fi, tmp=None):
        if tmp is None:
            tmp = []

        if (self.data[tuple(tmp)][0] <= fi) and (fi <= self.data[tuple(tmp)][1]):
            return_val = self.data[tuple(tmp)]
        elif fi < self.data[tuple(tmp)][1]:
            return_val = self.search(fi, tmp=tmp + [0])
        else:
            return_val = self.search(fi, tmp=tmp + [1])

        # print 'CHECKING:', return_val[0], fi, return_val[1], tmp
        assert (return_val[0] <= fi) and (fi <= return_val[1])
        return return_val

class StimulusSearch(object):

    def __init__(self, nwb_dataset):

        self.nwb_data = nwb_dataset
        self.epoch_df = nwb_dataset.get_stimulus_epoch_table()
        self.master_df = nwb_dataset.get_stimulus_table('master')
        self.epoch_bst = BinaryIntervalSearchTree.from_df(self.epoch_df)
        self.master_bst = BinaryIntervalSearchTree.from_df(self.master_df)

    def search(self, fi):

        try:

            # Look in fine-grain tree:
            search_result = self.master_bst.search(fi)
            return search_result
        except KeyError:

            # Current frame not found in a fine-grain interval;
            #   see if it is unregistered to a coarse-grain epoch:
            try:

                # THis will thow KeyError if not in coarse-grain epoch
                self.epoch_bst.search(fi)

                # Frame is in a coarse-grain  epoch, but not a fine grain interval;
                #   look backwards to find most recent find nearest matching interval
                if fi < self.epoch_df.iloc[0]['start']:

                    # Breakout if we go before the experiment:
                    return None
                else:
                    return self.search(fi-1)

            except KeyError:

                # Frame is unregistered at the coarse level; return None
                return None