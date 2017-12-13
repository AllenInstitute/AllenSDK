"""
Eye Tracking
=================================

This is an example of how to get eye tracking data for an
ophys_experiment, along with how to handle experiments that may not
have it.
"""
# sphinx_gallery_thumbnail_number = 3
from allensdk.brain_observatory.brain_observatory_exceptions import NoEyeTrackingException
from allensdk.core.brain_observatory_cache import BrainObservatoryCache
from allensdk.brain_observatory.observatory_plots import PUPIL_COLOR_MAP
from scipy.stats import gaussian_kde
import numpy as np
from matplotlib import pyplot as plt

#####################
# Open a cache for grabbing and cacheing experiments using the API.
boc = BrainObservatoryCache(manifest_file='boc/manifest.json')

#####################
# Get an experiment with eye tracking data. Wrap the call in a try-except,
# since many experiments do not have eye tracking and attempting to get pupil
# location from those experiments will throw a NoEyeTrackingException.
experiment_id = 569407590
data_set = boc.get_ophys_experiment_data(experiment_id)
try:
    timestamps, locations = data_set.get_pupil_location()
except NoEyeTrackingException:
    print("No eye tracking for experiment {}.".format(experiment_id))

#####################
# Plot some eye tracking data.

# looking at azimuth and altitude over time
# by default locations returned are (azimuth, altitude)
# passing as_spherical=False to get_pupil_location will return (x,y) in cm
timestamps, locations = data_set.get_pupil_location()
plt.figure(figsize=(14,4.5))
plt.plot(timestamps, locations.T[0])
plt.plot(timestamps, locations.T[1])
plt.title("Eye position over time")
plt.xlabel("time (s)")
plt.ylabel("angle (deg)")
plt.legend(['azimuth', 'altitude'])
plt.show()

#pupil size over time
timestamps, area = data_set.get_pupil_size()
plt.figure(figsize=(14,4.5))
plt.plot(timestamps, area)
plt.title("Pupil size over time")
plt.xlabel("time (s)")
plt.ylabel("area (px)")
plt.ylim(0, 20000)
plt.show()

######################
# Create a scatter plot of gaze positions, colored by a density estimage. In
# this example, the density estimate is calculated over every 100th data point
# for the sake of speed.
plt.figure()
xy_deg = locations[~np.isnan(locations).any(axis=1)]
c = gaussian_kde(xy_deg.T[:,::100])(xy_deg.T)
plt.scatter(xy_deg[:,0], xy_deg[:,1], s=5, c=c, cmap=PUPIL_COLOR_MAP,
            edgecolor='')
plt.title("Eye position scatter plot")
plt.xlabel("azimuth (deg)")
plt.ylabel("altitude (deg)")
plt.show()