Customization
=============

Customizing Your Simulation
---------------------------

The Allen Toolkit is designed as a component library, and in most cases the components can
be used independently of each other.  Some changes you can make are:

  #. Change the directory structure.  The manifest.json file contains the locations of
     various files and directories used by the simulation.  They can be changed in the
     manifest file, or you may choose to not use the manifest.json file in your template
     scripts.
     
  #. Manage multiple configurations.  If you have many different simulations to run,
     you can use different .conf files to combine the model, the stimulus and application
     parameters.  description JSON files can also be split into separate files so they can be
     mixed and matched at run-time.
     
  #. Modify the model description.  The model description format is very general.
     The structure is a JSON hash table with section names as keys.  Feel free to add new
     sections.  Within each section is an array of JSON objects.  The objects in each section
     are arbitrary.  It is best to limit the complexity of the objects in a section
     (shallow or no nesting, built-in literal values, etc.) to maintain compatibility with
     general-purpose viewing and processing tools.
     
  #. Add a specialized data format.  If there is a need to access a specialized data format,
     Allen SDK has the concept of a LOB (large object) parser.  This can be used to read or
     write the format using a third-party library, and then access the data through the manifest.
     LOBParsers are not intended to be an all-purpose solution, but they can be helpful in
     accessing data that is not easy to encode in JSON format.

  #. Use a different programming language.  While the examples use the Python programming
     language, it is not a requirement.  The formats in Allen SDK are designed to be
     readable from other programming languages.
     
