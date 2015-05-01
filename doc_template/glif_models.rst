Generalized LIF Models
======================

The Allen Cell Types Database contains Generalized Leaky Integrate and Fire 
(GLIF) that model the firing behavior of neurons at five levels of complexity.
Review the GLIF technical white paper for details on these models and
how their parameters were optimized (TODO).

The Allen SDK GLIF simulation module is an explicit time-stepping simulator 
that evolves a neuron's simulated voltage over the course of an input
current stimulus.  The modules also tracks the neuron's simulated spike
threshold and registers action potentials whenever voltage surpasses threshold.
Action potentials initiate reset rules that update voltage, threshold, and 
(optionally) trigger afterspike currents.  

The GLIF simulator in this package has a modular architecture
that enables users to choose from a number of dynamics and reset rules that
update the simulation's voltage, spike threshold, and afterspike currents
during the simulation. The GLIF package contains a built-in set of rules,
however developers can plug in custom rule implementations provided they
follow a simple argument specification scheme.

**Note:** the GLIF simulator module is still under heavy development and
may change significantly in the future.


Downloading  GLIF Models
------------------------

Visit http://celltypes.brain-map.org to find cells that have GLIF models
available for download.  Specifically:

   1. Click 'More Options +' and filter for GLIF models.
   2. Click the electrophysiology thumbnail for a cell on the right hand panel.
   3. Choose a GLIF model from the 'Show model responses' dropdown.
   4. Scroll down to the model response click 'Download model'.

One such link (for a simple LIF neuronal model, ID 472423251), would look
like this::

    http://api.brain-map.org/neuronal_model/download/472423251

This link returns .zip archive containing the neuron configuration file 
and sweep metadata required to simulate the model using stimuli applied to 
the cell.  To download the stimulus itself, the following code demonstrates 
how to retrieve the corresponding cell's NWB file (as well as the neuron 
configuration file and sweep metadata)::

    from allensdk.api.queries.glif_api import GlifApi
    import allensdk.core.json_utilities as json_utilities

    neuronal_model_id = 472423251
    
    glif_api = GlifApi()
    glif_api.get_neuronal_model(neuronal_model_id)
    glif_api.cache_stimulus_file('experiment.nwb')
    
    neuron_config = glif_api.get_neuron_config()
    json_utilities.write('neuron_config.json', neuron_config)
    
    ephys_sweeps = glif_api.get_ephys_sweeps()
    json_utilities.write('ephys_sweeps.json', ephys_sweeps)

Running a GLIF Simulation
-------------------------

You can use the files downloaded above to simulate all of the sweeps presented 
to the original cell as follows::

    
    import allensdk.core.json_utilities as json_utilities

    from allensdk.model.glif.neuron import GlifNeuron
    from allensdk.model.glif.simulate_neuron import simulation_neuron

    neuron_config = read_json('neuron_config.json')
    ephys_sweeps = json_utilities.read('ephys_sweeps.json')
    neuron = GlifNeuron.from_dict(neuron_config)

    simulate_neuron(neuron, ephys_sweeps, 'experiment.nwb', 'experiment.nwb', 0.05)

Note: in this case, simulated sweep voltages will overwrite the responses in 
the downloaded NWB file.  

If you have a custom stimulus you would like to apply to a neuronal model, 
try the following::

    from allensdk.model.glif.neuron import GlifNeuron
    import allensdk.core.json_utilities as json_utilities

    neuron_config = read_json('neuron_config.json')
    neuron = GlifNeuron.from_dict(neuron_config)

    # provide your own stimulus as an array of voltages (in volts)
    stimulus = ... 
    
    # important! provide the dt of your stimulus
    neuron.dt = 5e-6
    output = neuron.run(stimulus)

    voltage = output['voltage']
    threshold = output['threshold']
    spike_times = output['interpolated_spike_times']

GLIF Configuration
------------------

Instances of the GlifNeuron class require many parameters for initialization.  
Fixed neuron parameters are stored directly as parameters on the class instance:

================ ===================================== ========== ========
Parameter        Description                           Units      Type
================ ===================================== ========== ========
El               resting potential                     Volts      float
dt               time duration of each simulation step seconds    float
R_input          input resistance                      Ohms       float
C                capacitance                           Farads     float
asc_vector       afterspike current coefficients                  np.array 
spike_cut_length spike duration                        time steps int
th_inf           instantaneous threshold               Volts      float
th_adapt         adapted threshold                     Volts      float
================ ===================================== ========== ========

Some of these fixed parameters were optimized to fit Allen Cell Types Database 
electrophysiology data.  Optimized coefficients for these
parameters are stored by name in the instance.coeffs dictionary. For more details
on which parameters where optimized, please see the technical white paper (TODO link).

**Note about dt**: the `dt` value provided in the downloadable GLIF neuron configuration
files does not correspond to the sampling rate of the original stimulus.  Stimuli were
subsampled and filtered for parameter optimization.  Be sure to overwrite the neuron's
`dt` with the correct sampling rate::

    from allensdk.model.glif.neuron import GlifNeuron
    import allensdk.core.json_utilities as json_utilities
    from allensdk.core.nwb_data_set import NwbDataSet

    nwb_file_name = ...
    neuron_config_file_name = ...
    sweep_number = ...

    # load an NWB file
    ds = NwbDataSet(nwb_file_name)
    sweep_data = ds.get_sweep(sweep_number)

    # initialize the neuron
    neuron_config = read_json(neuron_config_file_name)
    neuron = GlifNeuron.from_dict(neuron_config)

    # overwrite dt and simulate the neuron
    neuron.dt = 1.0 / sweep_data['sampling_rate']
    neuron.run(sweep_data['stimulus'])

**Note about spike_cut_length**: the GLIF simulator can optionally skip ahead for 
a fixed amount of time when a spike is detected.  If you set `spike_cut_length` to
a positive value, `spike_cut_length` time steps will not be simulated after a spike
and instead be replaced with NaN values in the simulated outputs.

The GlifNeuron class has six methods that can be customized: three rules 
for updating voltage, spike threshold, and afterspike currents during the 
simulation; and three rules for updating those values when a spike is detected
(voltage surpasses spike threshold).

========================= ==============================================================
Method Type               Description
========================= ==============================================================
voltage_dynamics_method   Update simulation voltage for the next time step.
threshold_dynamics_method Update simulation spike threshold for the next time step.
AScurrent_dynamics_method Update afterspike current coefficients for the next time step.
voltage_reset_method      Reset simulation voltage after a spike occurs.
threshold_reset_method    Reset simulation spike threshold after a spike occurs.
AScurrent_reset_method    Reset afterspike current coefficients after a spike occurs.
========================= ==============================================================

The GLIF neuron configuration files available from the Allen Brain Atlas API use built-in
methods, however you can supply your own custom method if you like::

    # define your own custom voltage reset rule 
    # this one just returns the previous voltage value
    def custom_voltage_reset_rule(neuron, voltage_t0, custom_param_a, custom_param_b):
        return voltage_t0  

    # initialize a neuron from a neuron config file
    neuron_config = json_utilities.read('neuron_config.json')
    neuron = GlifNeuron.from_dict(neuron_config)

    # configure a new method and overwrite the neuron's old method
    method = neuron.configure_method('custom', custom_voltage_reset_rule, 
                                     { 'custom_param_a':1, 'custom_param_b': 2 })
    neuron.voltage_reset_method = method

Notice that the function is allowed to take custom parameters (here 'a' and 'b'), which are
configured on method initialization from a dictionary. For more details, see the documentation 
for the :py:class:`GlifNeuron` and :py:class:`GlifNeuronMethod` classes.

Built-in Dynamics Rules
-----------------------

The job of a dynamics rule is to describe how the simulator should update
the voltage, spike threshold, and afterspike currents of the simulator at
a given simulation time step.  

Voltage Dynamics Rules
++++++++++++++++++++++

These methods update the output voltage of the simulation.  They all expect a voltage, 
afterspike current vector, and current injection value to be passed in by the GlifNeuron. All 
other function parameters must be fixed using the GlifNeuronMethod class.  They all return an 
updated voltage value.

.. autofunction:: allensdk.model.glif.glif_neuron_methods.dynamics_voltage_forward_euler
    :noindex:

.. autofunction:: allensdk.model.glif.glif_neuron_methods.dynamics_voltage_euler_exact
    :noindex:

Threshold Dynamics Rules
++++++++++++++++++++++++

These methods update the spike threshold of the simulation.  They all expect the current
threshold and voltage values of the simulation to be passed in by the GlifNeuron. All 
other function parameters must be fixed using the GlifNeuronMethod class.  They all return an 
updated threshold value.

.. autofunction:: allensdk.model.glif.glif_neuron_methods.dynamics_threshold_three_components
    :noindex:

.. autofunction:: allensdk.model.glif.glif_neuron_methods.dynamics_threshold_inf
    :noindex:

Afterspike Current Dynamics Rules
+++++++++++++++++++++++++++++++++

These methods expect current afterspike current coefficients, current time step, 
and time steps of all previous spikes to be passed in by the GlifNeuron. All other function 
parameters must be fixed using the GlifNeuronMethod class.  They all return an updated
afterspike current array.

.. autofunction:: allensdk.model.glif.glif_neuron_methods.dynamics_AScurrent_exp
    :noindex:

.. autofunction:: allensdk.model.glif.glif_neuron_methods.dynamics_AScurrent_none
    :noindex:

Built-in Reset Rules
--------------------

The job of a reset rule is to describe how the simulator should update
the voltage, spike threshold, and afterspike currents of the simulator 
after the simulator has detected that the simulated voltage has surpassed
threshold.

Voltage Reset Rules
+++++++++++++++++++

These methods update the output voltage of the simulation after voltage has surpassed threshold. 
They all expect a voltageto be passed in by the GlifNeuron. All other function parameters must be 
fixed using the GlifNeuronMethod class.  They all return an updated voltage value.

.. autofunction:: allensdk.model.glif.glif_neuron_methods.reset_voltage_zero
    :noindex:

.. autofunction:: allensdk.model.glif.glif_neuron_methods.reset_voltage_bio_rules
    :noindex:


Threshold Reset Rules
+++++++++++++++++++++

These methods update the spike threshold of the simulation after a spike has been detected.  
They all expect the current threshold and the reset voltage value of the simulation to be passed in by the GlifNeuron. All other function parameters must be fixed using the GlifNeuronMethod 
class.  They all return an updated threshold value.

.. autofunction:: allensdk.model.glif.glif_neuron_methods.reset_threshold_inf
    :noindex:

.. autofunction:: allensdk.model.glif.glif_neuron_methods.reset_threshold_three_components
    :noindex:

Afterspike Reset Reset Rules
++++++++++++++++++++++++++++

These methods expect current afterspike current coefficients to be passed in by 
the GlifNeuron. All other function parameters must be fixed using the GlifNeuronMethod 
class.  They all return an updated afterspike current array.

.. autofunction:: allensdk.model.glif.glif_neuron_methods.reset_AScurrent_none
    :noindex:

.. autofunction:: allensdk.model.glif.glif_neuron_methods.reset_AScurrent_sum
    :noindex:



