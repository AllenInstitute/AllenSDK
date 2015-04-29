Generalized LIF Models
======================

The Allen Cell Types Database contains Generalized Leaky Integrate and Fire 
(GLIF) that model the firing behavior of neurons at five levels of complexity.
Review the GLIF technical white paper for details on these models and
how their parameters were optimized (TODO).

The GLIF simulator in this package has a modular architecture
that enables users to choose from a number of dynamics and reset rules that
update the simulation's voltage, spike threshold, and afterspike currents
during the simulation. The GLIF package contains a built-in set of rules,
however developers can plug in custom rule implementations provided they
follow a simple argument specification scheme.

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

    from allensdk.model.glif.neuron import GlifNeuron
    from allensdk.model.glif.simulate_neuron import simulation_neuron
    import allensdk.core.json_utilities as json_utilities

    neuron_config = read_json('neuron_config.json')
    ephys_sweeps = json_utilities.read('ephys_sweeps.json)
    neuron = GlifNeuron.from_dict(neuron_config)

    simulate_neuron(neuron, ephys_sweeps, 'experiment.nwb', 'experiment.nwb', .05)

Note that in this case, simulated sweep voltages will overwrite the responses in 
the downloaded NWB file.

If you have a custom stimulus you would like to apply to a neuronal model, 
you can instead do the following::

    from allensdk.model.glif.neuron import GlifNeuron
    import allensdk.core.json_utilities as json_utilities

    neuron_config = read_json('neuron_config.json')
    neuron = GlifNeuron.from_dict(neuron_config)

    # provide your own stimulus as an array of voltages (in volts)
    stimulus = ... 

    output = neuron.run(stimulus)

    voltage = output['voltage']
    threshold = output['threshold']
    spike_times = output['interpolated_spike_times']

GLIF Configuration
------------------

The 'neuron_config.json' files contain

El, dt, tau, R_input, C, asc_vector, spike_cut_length, th_inf, th_adapt, coeffs,
                 AScurrent_dynamics_method, voltage_dynamics_method, threshold_dynamics_method,
                 AScurrent_reset_method, voltage_reset_method, threshold_reset_method,
                 init_method_data, init_voltage, init_threshold, init_AScurrents, 
    

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



