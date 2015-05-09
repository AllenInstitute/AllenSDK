Generalized LIF Models
======================

The Allen Cell Types Database contains Generalized Leaky Integrate and Fire 
(GLIF) models that simulate the firing behavior of neurons at five levels of complexity.
Review the GLIF technical `white paper <http://help.brain-map.org/display/celltypes/Documentation>`_ 
for details on these models and how their parameters were optimized.

The Allen SDK GLIF simulation module is an explicit time-stepping simulator 
that evolves a neuron's simulated voltage over the course of an input
current stimulus.  The module also tracks the neuron's simulated spike
threshold and registers action potentials whenever voltage surpasses threshold.
Action potentials initiate reset rules that update voltage, threshold, and 
(optionally) trigger afterspike currents.  

The GLIF simulator in this package has a modular architecture
that enables users to choose from a number of dynamics and reset rules that
update the simulation's voltage, spike threshold, and afterspike currents
during the simulation. The GLIF package contains a built-in set of rules,
however developers can plug in custom rule implementations provided they
follow a simple argument specification scheme.

The Allen SDK GLIF simulator was developed and tested with Python 2.7.8, installed
as part of `Anaconda Python <https://store.continuum.io/cshop/anaconda/>`_ distribution 
version `2.1.0 <http://repo.continuum.io/archive/index.html>`_. 

The rest of this page provides examples demonstrating how to download models, 
examples of simulating these models, and general GLIF model documentation. 

**Note:** the GLIF simulator module is still under heavy development and
may change significantly in the future.


Downloading  GLIF Models
------------------------

There are two ways to download files necessary to run a GLIF model.  The
first way is to visit http://celltypes.brain-map.org and find cells that have 
GLIF models available for download.  The electrophysiology details page
for a cell has a neuronal model download link.  Specifically:

   1. Click 'More Options +' and filter for GLIF models.
   2. Click the electrophysiology thumbnail for a cell on the right hand panel.
   3. Choose a GLIF model from the 'Show model responses' dropdown.
   4. Scroll down to the model response click 'Download model'.

One such link (for a simple LIF neuronal model, ID 472423251), would look
like this::

    http://api.brain-map.org/neuronal_model/download/472423251

This link returns .zip archive containing the neuron configuration file 
and sweep metadata required to simulate the model with stimuli applied to 
the cell.  Specifically, the .zip archive will contain:

    * **472423251_neuron_config.json**: JSON config file for the GLIF model
    * **ephys_sweeps.json**: JSON with metadata for sweeps presented to the cell
    * **neuronal_model.json**: JSON with general metadata for the cell

If you would like to reproduce the model traces seen in the Cell Types Database, 
you can download an NWB file containing both the stimulus and cell response traces via a 
'Download data' link on the cell's electrophysiology page. See the :doc:`File Formats <file_formats>` 
page for more details on the NWB file format.

You can also download all of these files, including the cell's NWB file,
using the :py:class:`GlifApi <allensdk.api.queries.glif_api.GlifApi>` 
class::

    from allensdk.api.queries.glif_api import GlifApi
    import allensdk.core.json_utilities as json_utilities

    neuronal_model_id = 472423251
    
    glif_api = GlifApi()
    glif_api.get_neuronal_model(neuronal_model_id)
    glif_api.cache_stimulus_file('472423251.nwb')
    
    neuron_config = glif_api.get_neuron_config()
    json_utilities.write('472423251_neuron_config.json', neuron_config)
    
    ephys_sweeps = glif_api.get_ephys_sweeps()
    json_utilities.write('ephys_sweeps.json', ephys_sweeps)

Running a GLIF Simulation
-------------------------

To run a GLIF simulation, the most important file you you need is the ``neuron_config`` 
JSON file.  You can use this file to instantiate a simulator and feed in your own stimulus::

    import allensdk.core.json_utilities as json_utilities
    from allensdk.model.glif.glif_neuron import GlifNeuron

    # initialize the neuron
    neuron_config = json_utilities.read('472423251_neuron_config.json')
    neuron = GlifNeuron.from_dict(neuron_config)

    # make a short square pulse. stimulus units should be in Amps.
    stimulus = [ 0.0 ] * 100 + [ 10e-9 ] * 100 + [ 0.0 ] * 100

    # important! set the neuron's dt value for your stimulus in seconds
    neuron.dt = 5e-6

    # simulate the neuron
    output = neuron.run(stimulus)

    voltage = output['voltage']
    threshold = output['threshold']
    spike_times = output['interpolated_spike_times']

.. note:: 
    
    The GLIF simulator does not simulate during action potentials.  
    Instead it inserts ``NaN`` values for a fixed number of time steps when voltage 
    surpasses threshold.  The simulator skips ``neuron.spike_cut_length`` time steps 
    after voltage surpasses threshold.

To reproduce the model's traces displayed on the Allen Cell Types Database web page,
the Allen SDK provides the :py:mod:`allensdk.core.model.glif.simulate_neuron` 
module for simulating all sweeps presented to a cell and storing them in the NWB format::

    import allensdk.core.json_utilities as json_utilities

    from allensdk.model.glif.glif_neuron import GlifNeuron
    from allensdk.model.glif.simulate_neuron import simulate_neuron

    neuron_config = json_utilities.read('472423251_neuron_config.json')
    ephys_sweeps = json_utilities.read('ephys_sweeps.json')
    ephys_file_name = '472423251.nwb'

    neuron = GlifNeuron.from_dict(neuron_config)

    simulate_neuron(neuron, ephys_sweeps, ephys_file_name, ephys_file_name, 0.05)

.. warning::

    These stimuli are sampled at a very high resolution (200kHz), 
    and a given cell can have many sweeps.  This process can take over an hour.

The ``simulate_neuron`` function call simulates all sweeps in the NWB file.  
Because the same NWB file is being used for both input and output, 
the cell's response traces will be overwritten as stimuli are simulated. 
``simulate_neuron`` optionally accepts a value which will be used to overwrite
these ``NaN`` values generated during action potentials (in this case 0.05 Volts).

If you would like to run a single sweep instead of all sweeps, try the following::

    import allensdk.core.json_utilities as json_utilities
    from allensdk.model.glif.glif_neuron import GlifNeuron
    from allensdk.core.nwb_data_set import NwbDataSet

    neuron_config = json_utilities.read('472423251_neuron_config.json')
    ephys_sweeps = json_utilities.read('ephys_sweeps.json')
    ephys_file_name = '472423251.nwb'

    # pull out the stimulus for the first sweep
    ephys_sweep = ephys_sweeps[0]
    ds = NwbDataSet(ephys_file_name)
    data = ds.get_sweep(ephys_sweep['sweep_number']) 
    stimulus = data['stimulus']

    # initialize the neuron
    # important! update the neuron's dt for your stimulus
    neuron = GlifNeuron.from_dict(neuron_config)
    neuron.dt = 1.0 / data['sampling_rate']

    # simulate the neuron
    output = neuron.run(stimulus)

    voltage = output['voltage']
    threshold = output['threshold']
    spike_times = output['interpolated_spike_times']

.. note:: 
    
    The ``dt`` value provided in the downloadable GLIF neuron configuration
    files does not correspond to the sampling rate of the original stimulus.  Stimuli were
    subsampled and filtered for parameter optimization.  Be sure to overwrite the neuron's
    ``dt`` with the correct sampling rate.

If you would like to plot the outputs of this simulation using numpy and matplotlib, try::

    import numpy as np
    import matplotlib.pyplot as plt

    voltage = output['voltage']
    threshold = output['threshold']
    interpolated_spike_times = output['interpolated_spike_times']
    spike_times = output['interpolated_spike_times']
    interpolated_spike_voltages = output['interpolated_spike_voltage']
    interpolated_spike_thresholds = output['interpolated_spike_threshold']
    grid_spike_indices = output['spike_time_steps']
    grid_spike_times = output['grid_spike_times']
    after_spike_currents = output['AScurrents']

    # create a time array for plotting
    time = np.arange(len(stimulus))*neuron.dt

    plt.figure(figsize=(10, 10))

    # plot stimulus
    plt.subplot(3,1,1)
    plt.plot(time, stimulus)
    plt.xlabel('time (s)')
    plt.ylabel('current (A)')
    plt.title('Stimulus')

    # plot model output
    plt.subplot(3,1,2)
    plt.plot(time,  voltage, label='voltage')
    plt.plot(time,  threshold, label='threshold')
    plt.plot(interpolated_spike_times, interpolated_spike_voltages, 'x', 
             label='interpolated spike')
    plt.plot((grid_spike_indices-1)*neuron.dt, voltage[grid_spike_indices-1], '.', 
             label='last step before spike')
    plt.xlabel('time (s)')
    plt.ylabel('voltage (V)')
    plt.legend(loc=3)
    plt.title('Model Response')

    # plot after spike currents
    plt.subplot(3,1,3)
    for ii in range(np.shape(after_spike_currents)[1]):
        plt.plot(time, after_spike_currents[:,ii])
    plt.xlabel('time (s)')
    plt.ylabel('current (A)')
    plt.title('After Spike Currents')

    plt.tight_layout()
    plt.show()

.. note:: 

    There is both an interpolated and grid spike time.  The grid spike is the first time step 
    where the voltage is higher than the threshold.  Note that if you try to plot the voltage at the grid 
    spike indices the output will be ``NaN``. The interpolated spike is the calculated intersection of the 
    threshold and voltage between the time steps.

GLIF Configuration
------------------

Instances of the :py:class:`~allensdk.model.glif.glif_neuron.GlifNeuron` 
class require many parameters for initialization.  
Fixed neuron parameters are stored directly as properties on the class instance:

================ ===================================== ========== ========
Parameter        Description                           Units      Type
================ ===================================== ========== ========
El               resting potential                     Volts      float
dt               time duration of each simulation step seconds    float
R_input          input resistance                      Ohms       float
C                capacitance                           Farads     float
asc_vector       afterspike current coefficients       Amps       np.array 
spike_cut_length spike duration                        time steps int
th_inf           instantaneous threshold               Volts      float
th_adapt         adapted threshold                     Volts      float
================ ===================================== ========== ========

Some of these fixed parameters were optimized to fit Allen Cell Types Database 
electrophysiology data.  Optimized coefficients for these
parameters are stored by name in the ``neuron.coeffs`` dictionary. For more details
on which parameters were optimized, please see the technical 
`white paper <http://help.brain-map.org/display/celltypes/Documentation>`_.

The :py:class:`~allensdk.model.glif.glif_neuron.GlifNeuron` class has six 
methods that can be customized: three rules 
for updating voltage, threshold, and afterspike currents during the 
simulation; and three rules for updating those values when a spike is detected
(voltage surpasses threshold).

========================= ==============================================================
Method Type               Description
========================= ==============================================================
voltage_dynamics_method   Update simulation voltage for the next time step.
threshold_dynamics_method Update simulation threshold for the next time step.
AScurrent_dynamics_method Update afterspike current coefficients for the next time step.
voltage_reset_method      Reset simulation voltage after a spike occurs.
threshold_reset_method    Reset simulation threshold after a spike occurs.
AScurrent_reset_method    Reset afterspike current coefficients after a spike occurs.
========================= ==============================================================

The GLIF neuron configuration files available from the Allen Brain Atlas API use built-in
methods, however you can supply your own custom method if you like::

    # define your own custom voltage reset rule 
    # this one linearly scales the input voltage
    def custom_voltage_reset_rule(neuron, voltage_t0, custom_param_a, custom_param_b):
        return custom_param_a * voltage_t0 + custom_param_b

    # initialize a neuron from a neuron config file
    neuron_config = json_utilities.read('472423251_neuron_config.json')
    neuron = GlifNeuron.from_dict(neuron_config)

    # configure a new method and overwrite the neuron's old method
    method = neuron.configure_method('custom', custom_voltage_reset_rule, 
                                     { 'custom_param_a': 0.1, 'custom_param_b': 0.0 })
    neuron.voltage_reset_method = method

    output = neuron.run(stimulus)
    

Notice that the function is allowed to take custom parameters (here ``custom_param_a`` and 
``custom_param_b``), which are configured on method initialization from a dictionary. For more details, 
see the documentation for the :py:class:`GlifNeuron <allensdk.model.glif.glif_neuron.GlifNeuron>` and 
:py:class:`GlifNeuronMethod <allensdk.model.glif.glif_neuron_methods.GlifNeuronMethod>` classes.


Built-in Dynamics Rules
-----------------------

The job of a dynamics rule is to describe how the simulator should update
the voltage, spike threshold, and afterspike currents of the simulator at
a given simulation time step.  

**Voltage Dynamics Rules**

These methods update the output voltage of the simulation.  They all expect a voltage, 
afterspike current vector, and current injection value to be passed in by the GlifNeuron. All 
other function parameters must be fixed using the GlifNeuronMethod class.  They all return an 
updated voltage value.

    :py:meth:`allensdk.model.glif.glif_neuron_methods.dynamics_voltage_forward_euler`
    :py:meth:`allensdk.model.glif.glif_neuron_methods.dynamics_voltage_euler_exact`

**Threshold Dynamics Rules**

These methods update the spike threshold of the simulation.  They all expect the current
threshold and voltage values of the simulation to be passed in by the GlifNeuron. All 
other function parameters must be fixed using the GlifNeuronMethod class.  They all return an 
updated threshold value.

    :py:meth:`allensdk.model.glif.glif_neuron_methods.dynamics_threshold_three_components`
    :py:meth:`allensdk.model.glif.glif_neuron_methods.dynamics_threshold_inf`

**Afterspike Current Dynamics Rules**

These methods expect current afterspike current coefficients, current time step, 
and time steps of all previous spikes to be passed in by the GlifNeuron. All other function 
parameters must be fixed using the GlifNeuronMethod class.  They all return an updated
afterspike current array.

    :py:meth:`allensdk.model.glif.glif_neuron_methods.dynamics_AScurrent_exp`
    :py:meth:`allensdk.model.glif.glif_neuron_methods.dynamics_AScurrent_none`

Built-in Reset Rules
--------------------

The job of a reset rule is to describe how the simulator should update
the voltage, spike threshold, and afterspike currents of the simulator 
after the simulator has detected that the simulated voltage has surpassed
threshold.

**Voltage Reset Rules**

These methods update the output voltage of the simulation after voltage has surpassed threshold. 
They all expect a voltageto be passed in by the GlifNeuron. All other function parameters must be 
fixed using the GlifNeuronMethod class.  They all return an updated voltage value.

    :py:meth:`allensdk.model.glif.glif_neuron_methods.reset_voltage_zero`
    :py:meth:`allensdk.model.glif.glif_neuron_methods.reset_voltage_bio_rules`

**Threshold Reset Rules**

These methods update the spike threshold of the simulation after a spike has been detected.  
They all expect the current threshold and the reset voltage value of the simulation to be passed in by the GlifNeuron. All other function parameters must be fixed using the GlifNeuronMethod 
class.  They all return an updated threshold value.

    :py:meth:`allensdk.model.glif.glif_neuron_methods.reset_threshold_inf`
    :py:meth:`allensdk.model.glif.glif_neuron_methods.reset_threshold_three_components`

**Afterspike Reset Reset Rules**

These methods expect current afterspike current coefficients to be passed in by 
the GlifNeuron. All other function parameters must be fixed using the GlifNeuronMethod 
class.  They all return an updated afterspike current array.

    :py:meth:`allensdk.model.glif.glif_neuron_methods.reset_AScurrent_none`
    :py:meth:`allensdk.model.glif.glif_neuron_methods.reset_AScurrent_sum`


