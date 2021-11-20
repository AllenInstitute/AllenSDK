import os

# ignore test_optimize_run.py if don't have neuron installed
collect_ignore = []
if os.getenv("TEST_NEURON") != 'true':
    collect_ignore.append("test_optimize_run.py")
