A script for writing extracellular electrophysiology data from LIMS -> NWB 2.0
==============================================================================


Notes
=====

LIMS
----
- The format of uploaded data has changed in many minor ways over time - for instance metrics.npy -> metrics.csv. These changes  ended up meaning that I couldn't find 2 sessions whose relevant data could be accessed in the same way.
- acquisition dates in LIMS are all null. this field is required for NWB creation
- channels are not modeled in LIMS. This is a problem because there is no global source for truth about which one is which - each data getter on the API needs to seperately invent the same mapping of ids, for instance.
- units have the same problem
- some probes have metrics file on the filesystem but not in LIMS
- how to tell if a file has gone through all the stages of upload?
- probe name needs to be part of a more sensible model (right now it is a weird code for targeted region)

pynwb
-----
- I don't know how to assign waveforms to units. There are two ways of storing spike time information (UnitTimes and Clustering). Clustering has ClusterWaveforms, but no association to the Units table I can see, while UnitTimes does not seem to have any place for waveforms.
- Roundtrip testing is hard, as data tends to come back as h5py objects. This makes it difficult to have confidence that the data we expect to write has actually been written.
- every electrode must have an x, y, z. This is facially sensible, as electrode ought to be locatable within the 3D space of the brain. However, there is no way to say "we should have this but don't" (as is the case for us until we get fmost online) and no way to specify alternate coordinate systems, such as the 2D neuropixels probe coordinates. Moreover, there is no way to identify the reference space that x, y, z live in. For us it will be the CCF, but for someone else ... who knows? I used placeholders here.
- the problem of odd required fields is all over pynwb. "source" is a good example. Generally it is hard to say in a well-known way: 1) I didn't collect this or 2) I did collect this but in a slightly different way with these additional data. If the right answer to those is "use extensions", then these fields should not be required by default.
- devices probably ought to be a table - you often want metadata on them. 