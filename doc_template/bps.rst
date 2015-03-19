BPS Utility
===========



Conf File
---------

The .conf file is a file that can be parsed by
the Allen Wrench application_config class.
It is in "INI" format, and the comment character is #.

::

    [biophys]
    workdir: 10_cells/work
    log_config_path: common/logs_on.conf
    debug: off
    model_file: 10_cells/model.json, 10_cells/run_config.json


It tells bps what wordir to run in, the location of the log configuration,
whether to connect to a debugger, the locations of the model file(s) relative to the workdir,
and the location of the run configuration.

Several model files can be specified on the model_file line, separated by commas.
The model files and run file are combined
into a single :doc:`model description </model_description>`.
By convention, the model structure should be specified in the model_file line.
The run_file is used for other experimental conditions including the stimulus.

The logging config file is also in "INI" format.
It is a standard Python logging configuration file.


Embedding Application Configuration in a Model Description File
---------------------------------------------------------------

Rather than using a .conf file, it is possible to embed application configuration
in a :doc:`model description </model_description>` file in a "biophys" section.
Note that the model_file value is an array rather than a comma-separated string.
An application configuration section embedded in the same file as model description sections
would reference itself in the model_file entry.


model.json:
::

    {
        "biophys": [{
            "workdir": "10_cells/work",
            "log_config_path": "common/logs_on.conf",
            "debug": "off",
            "model_file": [ "model.json", "run_config.json" ]
        }]
    }



BPS Commands
------------

::

    bps help
    
Show a brief help message listing the other bps commands.


::

    bps nrnivmodl bps.conf
    
This will call the NEURON package to compile the NMODL .mod files.
It must be called before using the run_simple or run_model commands.


::

    bps run_simple bps.conf
    
This will run a single threaded simulation locally.


::

    bps run_model bps.conf
    
This will run a multi-threaded simulation locally.


::

    bps cluster_run_model bps.conf
    
This is the remote equivalent of run_model (server_side).


::

    bps run_model_cluster bps.conf
    
This will submit a simulation to run on a PBS cluster.


::

    bps qsub_script bps.conf
    
This will generate a script that can be used to submit a simulation to run on a PBS cluster.
It does not actually run the job.


::

    bps nrnivmodl_cluster bps.conf
    
This submits a job to a cluster to run the NEURON nrnivmodl command to compile the NMODL .mod files.


