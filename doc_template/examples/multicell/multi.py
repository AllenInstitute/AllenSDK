from allensdk.model.biophys_sim.config import Config
from utils import Utils
import numpy

config = Config().load('config.json')

print('Saving full config to config_full.json')
with open('config_full.json', 'wb') as f:
    import json
    f.write(json.dumps(config.data, indent=2))

# access fit like so:
fit_ids = config.data['fit_ids'][0]
lid = fit_ids['larry']
fit_larry = config.data['fit'][lid]

# configure NEURON
utils = Utils(config)
h = utils.h

# configure model
manifest = config.manifest
morphology_path = config.manifest.get_path('MORPHOLOGY')
utils.generate_morphology()
utils.load_cell_parameters()

# configure stimulus
utils.setup_iclamp()

# configure recording
vec = utils.record_values()

# configure duration and time step of simulation
h.dt = 0.025
h.tstop = 20

# run the model
h.finitialize()
h.run()

# scaling
mV = 1.0e-3
ms = 1.0e-3
output_data = numpy.array(vec['v']) * mV
output_times = numpy.array(vec['t']) * ms
output = numpy.column_stack((output_times, output_data))

# write to a dat File
v_out_path = manifest.get_path("output_dat")
with open (v_out_path, "w") as f:
    numpy.savetxt(f, output)