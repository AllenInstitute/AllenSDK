# Optimization of single neuron biophysical models with DEAP

Start a single neuron optimization by passive a specimen ID to `start_specimen.py`.

	$ ./start_specimen.py 471789504
	
If [GNU parallel](http://www.gnu.org/software/parallel/) is installed, you can start multiple specimens at time by
	
	$ parallel --gnu ./start_specimen.py ::: 471789504 475549334 ...

The scripts `monitor_stage_1.py` and `monitor_stage_2.py` are run periodically with cron to check if all cluster jobs associated with a specimen have completed. If they have, they start up the next stage (or mark the specimen as complete).

## Requirements

* [DEAP](https://github.com/DEAP/deap)
* [NEURON 7.4](http://www.neuron.yale.edu/neuron/)
* [AllenSDK](http://alleninstitute.github.io/AllenSDK/) (with feature extractor)
