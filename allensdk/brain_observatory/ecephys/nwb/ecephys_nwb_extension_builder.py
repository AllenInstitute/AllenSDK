from pynwb.spec import NWBAttributeSpec, NWBGroupSpec, NWBNamespaceBuilder

# This is the script used to generate the AIBS ecephys NWB extension .yaml
# files. It can be run by installing pynwb and executing
# `nwb_extension_builder.py`. It will generate the .yaml extension files in
# the same directory which the script is run in. For more details see:
# https://pynwb.readthedocs.io/en/stable/extensions.html

ns_builder = NWBNamespaceBuilder(doc="Allen Institute Ecephys Extension",
                                 version="0.2.0",
                                 name="ndx-aibs-ecephys",
                                 author="Allen Institute for Brain Science",
                                 contact="waynew@alleninstitute.org")

probe_id_attr = NWBAttributeSpec(name="probe_id",
                                 doc="Unique ID of the neuropixels probe",
                                 dtype="int")

# Ecephys probe device extension (inherits from NWB `Device`)
sampling_rate_attr = NWBAttributeSpec(name="sampling_rate",
                                      doc="The sampling rate for the device",
                                      dtype="float64")

ecephys_probe_attributes = [sampling_rate_attr, probe_id_attr]

ecephys_probe_ext = NWBGroupSpec(doc="A neuropixels probe device",
                                 attributes=ecephys_probe_attributes,
                                 neurodata_type_def="EcephysProbe",
                                 neurodata_type_inc="Device")

# Ecephys electrode group extension (inherits from NWB `ElectrodeGroup`)
has_lfp_data_attr = NWBAttributeSpec(name="has_lfp_data",
                                     doc="Indicates availability of LFP data",
                                     dtype="bool")

lfp_sampling_rate = NWBAttributeSpec(name="lfp_sampling_rate",
                                     doc=("The sampling rate at which data "
                                          "were acquired on this electrode "
                                          "group's channels"),
                                     dtype="float64")

ecephys_egroup_attributes = [has_lfp_data_attr, probe_id_attr,
                             lfp_sampling_rate]

ecephys_egroup_ext = NWBGroupSpec(doc=("A group consisting of the channels "
                                       "on a single neuropixels probe"),
                                  attributes=ecephys_egroup_attributes,
                                  neurodata_type_def="EcephysElectrodeGroup",
                                  neurodata_type_inc="ElectrodeGroup")

# Ecephys specimen metadata extension (inherits from NWB `Subject`)
specimen_name_attr = NWBAttributeSpec(name="specimen_name",
                                      doc="Full name of specimen",
                                      dtype="text")

age_in_days_attr = NWBAttributeSpec(name="age_in_days",
                                    doc="Age of specimen in days",
                                    dtype="float")

strain_attr = NWBAttributeSpec(name="strain",
                               doc="Specimen strain",
                               dtype="text")

ecephys_specimen_attributes = [specimen_name_attr, age_in_days_attr,
                               strain_attr]

ecephys_specimen_ext = NWBGroupSpec(doc="Metadata for ecephys specimen",
                                    attributes=ecephys_specimen_attributes,
                                    neurodata_type_def="EcephysSpecimen",
                                    neurodata_type_inc="Subject")

ext_source = "ndx-aibs-ecephys.extension.yaml"
ns_builder.add_spec(ext_source, ecephys_probe_ext)
ns_builder.add_spec(ext_source, ecephys_egroup_ext)
ns_builder.add_spec(ext_source, ecephys_specimen_ext)

namespace_path = "ndx-aibs-ecephys.namespace.yaml"
ns_builder.export(namespace_path)
