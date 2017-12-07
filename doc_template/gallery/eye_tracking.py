"""
Eye Tracking
=================================

This is an example of how to get eye tracking data for an
ophys_experiment, along with how to handle experiments that may not
have it.
"""
from allensdk.brain_observatory.brain_observatory_exceptions import NoEyeTrackingException
from allensdk.core.brain_observatory_cache import BrainObservatoryCache
from matplotlib import pyplot as plt

#####################
# Open a cache for grabbing and cacheing experiments using the API.
boc = BrainObservatoryCache(manifest_file='boc/manifest.json')

#####################
# Get an experiment with eye tracking data. Wrap the call in a try-except,
# since many experiments do not have eye tracking and attempting to get pupil
# location from those experiments will throw a NoEyeTrackingException.
data_set = boc.get_ophys_experiment_data(569407590)
try:
    timestamps, locations = data_set.get_pupil_location()
except NoEyeTrackingException:
    print("No eye tracking for experiment %s." % data_set.get_metadata()["ophys_experiment_id"])

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

# scatter of gaze positions over approximate screen area
plt.figure()
plt.scatter(locations.T[0], locations.T[1], s=2, c="m", edgecolor="")
plt.title("Eye position scatter plot")
plt.xlim(-70, 70)
plt.ylim(-60, 60)
plt.xlabel("azimuth (deg)")
plt.ylabel("altitude (deg)")
plt.show()