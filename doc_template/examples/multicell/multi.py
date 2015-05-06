from allensdk.model.biophys_sim.config import Config
from utils import Utils

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

config = Config().load('config.json')

# configure NEURON
utils = Utils(config)
h = utils.h

# configure model
manifest = config.manifest

utils.generate_cells()
utils.connect_cells()

# configure stimulus
utils.setup_iclamp_step(utils.cells[0], 0.27, 1020.0, 750.0)
h.dt = 0.025
h.tstop = 3000

# configure recording
vec = utils.record_values()

# run the model
h.finitialize()
h.run()

fig, axes = plt.subplots(3, 1, sharex=True)
for i in range(len(utils.cells)):
    axes[i].plot(vec["t"], vec["v"][i])
    axes[i].set_title(utils.cells_data[i]["type"])
plt.tight_layout()
plt.show()
