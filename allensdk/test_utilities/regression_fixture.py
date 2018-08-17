import os
import sys
import logging
import json
from pkg_resources import resource_filename  # @UnresolvedImport

if 'TEST_SESSION_ANALYSIS_REGRESSION_DATA' in os.environ:
    data_file = os.environ['TEST_SESSION_ANALYSIS_REGRESSION_DATA']
else:
    data_file = resource_filename(__name__, '../test/brain_observatory/test_session_analysis_regression_data_list.json')

def get_list_of_path_dict():
    pyversion = sys.version_info[0]
    logging.debug("loading " + data_file)
    with open(data_file,'r') as f:
        return [curr_fixture for curr_fixture in json.load(f) if curr_fixture['version'] == pyversion]
