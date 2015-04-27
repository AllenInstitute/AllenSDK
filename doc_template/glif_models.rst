Generalized LIF Models
======================

The Allen Cell Types Database contains Generalized Leaky Integrate and Fire 
(GLIF) that model the firing behavior of neurons at five levels of complexity:

    1. Leaky Integrate and Fire (LIF)
    2. LIF + Reset Rules (LIF-R)
    3. LIF + Afterspike Currents (LIF-ASC)
    4. LIF-R + Afterspike Currents (LIF-R-ASC)
    5. LIF-R-ASC + Threshold Adaptation (LIF-R-ASC-A)

Please review the GLIF technical white paper for details on these models and
how their parameters were optimized (TODO).

The GLIF simulator in this package has a modular architecture
that enables users to choose from a number of dynamics and reset rules that
update the simulation's voltage, spike threshold, and afterspike currents
during the simulation. The GLIF package contains a built-in set of rules,
however developers can plug in custom rule implementations provided they
follow a simple argument specification scheme.

Running a GLIF Simulation
-------------------------

The Allen Cell Types Atlas contains a large number of GLIF models that
have been optimized under various configurations against electrophysiology
traces from cells in the Atlas.  To run a model, first download that cell's
model configuration file, sweep information, and electrophysiology file
from the Allen Brain Atlas API::

    http://api.brain-map.org/neuronal_model/download/472423251

The result will be a .zip archive containing the model configuration file 
and sweep metadata required to simulate the model.  If you would like to 
download the stimulus used to train the model, the following code 
demonstrates how to download the corresponding NWB file (as well as the
model configuration file and sweep metadata)::

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

You can use these files to simulate all of the sweeps for that cell as 
follows::

    from allensdk.model.glif.neuron import GlifNeuron
    from allensdk.model.glif.simulate_neuron import simulation_neuron
    import allensdk.core.json_utilities as json_utilities

    neuron_config = read_json('neuron_config.json')
    ephys_sweeps = json_utilities.read('ephys_sweeps.json)
    neuron = GlifNeuron.from_dict(neuron_config)

    simulate_neuron(neuron, ephys_sweeps, 'experiment.nwb', 'experiment.nwb', .05)

Note that in this case, simulated sweep voltages will overwrite the responses in 
the downloaded NWB file.
    
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

.. autofunction:: allensdk.model.glif.glif_neuron_methods.dynamics_voltage_linear
    :noindex:

.. autofunction:: allensdk.model.glif.glif_neuron_methods.dynamics_voltage_piecewise_linear
    :noindex:

Threshold Dynamics Rules
++++++++++++++++++++++++

These methods update the spike threshold of the simulation.  They all expect the current
threshold and voltage values of the simulation to be passed in by the GlifNeuron. All 
other function parameters must be fixed using the GlifNeuronMethod class.  They all return an 
updated threshold value.

.. autofunction:: allensdk.model.glif.glif_neuron_methods.dynamics_threshold_adapt_standard
    :noindex:

.. autofunction:: allensdk.model.glif.glif_neuron_methods.dynamics_threshold_inf
    :noindex:

.. autofunction:: allensdk.model.glif.glif_neuron_methods.dynamics_threshold_fixed
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

.. autofunction:: allensdk.model.glif.glif_neuron_methods.reset_voltage_v_before
    :noindex:

.. autofunction:: allensdk.model.glif.glif_neuron_methods.reset_voltage_fixed
    :noindex:

Threshold Reset Rules
+++++++++++++++++++++

These methods update the spike threshold of the simulation after a spike has been detected.  
They all expect the current threshold and the reset voltage value of the simulation to be passed in by the GlifNeuron. All other function parameters must be fixed using the GlifNeuronMethod 
class.  They all return an updated threshold value.

.. autofunction:: allensdk.model.glif.glif_neuron_methods.reset_threshold_max_v_th
    :noindex:

.. autofunction:: allensdk.model.glif.glif_neuron_methods.reset_threshold_inf
    :noindex:

.. autofunction:: allensdk.model.glif.glif_neuron_methods.reset_threshold_fixed
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



