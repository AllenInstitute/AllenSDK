Generalized Linear Integrate and Fire Models
============================================

**TODO: High description level of GLIF models goes here.**

The GLIF model implementation in this package has a modular architecture
that enables users to choose from a number of dynamics and reset rules that
will be applied to simulation voltage, spike threshold, and afterspike currents
during the simulation. The GLIF package contains a built-in set of rules,
however developers can plug in custom rule implementations provided they
follow a simple argument specification scheme.

Running a GLIF Simulation
-------------------------

The Allen Cell Types Atlas contains a large number of GLIF models that
have been optimized under various configurations against electrophysiology
traces from cells in the Atlas.  To run a model, first download that cell's
model configuration file, sweep information, and electrophysiology file::

    TODO

You can use these files to simulate all of the sweeps for that cell as 
follows::

    from allensdk.model.GLIF.neuron import GLIFNeuron
    from allensdk.model.GLIF.simulate_neuron import simulation_neuron
    from allensdk.model.GLIF.utilities import read_json

    neuron_config_file_name = 'neuron_config.json'
    sweep_file_name = 'sweeps.json'
    input_file_name = 'some_input_file.ext'
    output_file_name = 'some_file_name.ext'

    neuron_config = read_json(neuron_config_file_name)
    sweeps = read_json(sweep_file_name)
    neuron = GLIFNeuron.from_dict(neuron_config)
    simulate_neuron(neuron, data_config['sweeps'], output_file_name)
    
Built-in Dynamics Rules
-----------------------

The job of a dynamics rule is to describe how the simulator should update
the voltage, spike threshold, and afterspike currents of the simulator at
a given simulation time step.  

Voltage Dynamics Rules
++++++++++++++++++++++

These methods update the output voltage of the simulation.  They all expect a voltage, 
afterspike current vector, and current injection value to be passed in by the GLIFNeuron. All 
other function parameters must be fixed using the GLIFNeuronMethod class.  They all return an 
updated voltage value.

.. autofunction:: allensdk.model.GLIF.neuron_methods.dynamics_voltage_linear
    :noindex:

.. autofunction:: allensdk.model.GLIF.neuron_methods.dynamics_voltage_piecewise_linear
    :noindex:

Threshold Dynamics Rules
++++++++++++++++++++++++

These methods update the spike threshold of the simulation.  They all expect the current
threshold and voltage values of the simulation to be passed in by the GLIFNeuron. All 
other function parameters must be fixed using the GLIFNeuronMethod class.  They all return an 
updated threshold value.

.. autofunction:: allensdk.model.GLIF.neuron_methods.dynamics_threshold_adapt_standard
    :noindex:

.. autofunction:: allensdk.model.GLIF.neuron_methods.dynamics_threshold_inf
    :noindex:

.. autofunction:: allensdk.model.GLIF.neuron_methods.dynamics_threshold_fixed
    :noindex:

Afterspike Current Dynamics Rules
+++++++++++++++++++++++++++++++++

These methods expect current afterspike current coefficients, current time step, 
and time steps of all previous spikes to be passed in by the GLIFNeuron. All other function 
parameters must be fixed using the GLIFNeuronMethod class.  They all return an updated
afterspike current array.

.. autofunction:: allensdk.model.GLIF.neuron_methods.dynamics_AScurrent_exp
    :noindex:

.. autofunction:: allensdk.model.GLIF.neuron_methods.dynamics_AScurrent_none
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
They all expect a voltageto be passed in by the GLIFNeuron. All other function parameters must be 
fixed using the GLIFNeuronMethod class.  They all return an updated voltage value.

.. autofunction:: allensdk.model.GLIF.neuron_methods.reset_voltage_zero
    :noindex:

.. autofunction:: allensdk.model.GLIF.neuron_methods.reset_voltage_v_before
    :noindex:

.. autofunction:: allensdk.model.GLIF.neuron_methods.reset_voltage_fixed
    :noindex:

Threshold Reset Rules
+++++++++++++++++++++

These methods update the spike threshold of the simulation after a spike has been detected.  
They all expect the current threshold and the reset voltage value of the simulation to be passed in by the GLIFNeuron. All other function parameters must be fixed using the GLIFNeuronMethod 
class.  They all return an updated threshold value.

.. autofunction:: allensdk.model.GLIF.neuron_methods.reset_threshold_max_v_th
    :noindex:

.. autofunction:: allensdk.model.GLIF.neuron_methods.reset_threshold_inf
    :noindex:

.. autofunction:: allensdk.model.GLIF.neuron_methods.reset_threshold_fixed
    :noindex:

Afterspike Reset Reset Rules
++++++++++++++++++++++++++++

These methods expect current afterspike current coefficients to be passed in by 
the GLIFNeuron. All other function parameters must be fixed using the GLIFNeuronMethod 
class.  They all return an updated afterspike current array.

.. autofunction:: allensdk.model.GLIF.neuron_methods.reset_AScurrent_none
    :noindex:

.. autofunction:: allensdk.model.GLIF.neuron_methods.reset_AScurrent_sum
    :noindex:



