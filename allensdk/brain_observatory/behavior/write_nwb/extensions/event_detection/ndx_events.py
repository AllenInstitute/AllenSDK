import os
from pynwb import load_namespaces, get_class

# Set path of the namespace.yaml file to the expected install location
ndx_events_specpath = os.path.join(
    os.path.dirname(__file__),
    'ndx-event-detection.namespace.yaml'
)

# Load the namespace
load_namespaces(ndx_events_specpath)


EventDetection = get_class('EventDetection', 'ndx-event-detection')
