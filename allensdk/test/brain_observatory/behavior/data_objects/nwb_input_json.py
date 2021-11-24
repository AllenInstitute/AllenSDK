import json
from pathlib import Path


class NwbInputJson:
    def __init__(self):
        dir = Path(__file__).parent.resolve()
        test_data_dir = dir / 'test_data'
        with open(test_data_dir / 'test_input.json') as f:
            dict_repr = json.load(f)
        dict_repr = dict_repr['session_data']
        dict_repr['sync_file'] = str(test_data_dir / 'sync.h5')
        dict_repr['behavior_stimulus_file'] = str(test_data_dir /
                                                  'behavior_stimulus_file.pkl')
        dict_repr['dff_file'] = str(test_data_dir / 'demix_file.h5')
        dict_repr['demix_file'] = str(test_data_dir / 'demix_file.h5')
        dict_repr['events_file'] = str(test_data_dir / 'events.h5')

        self._dict_repr = dict_repr

    @property
    def dict_repr(self):
        return self._dict_repr
