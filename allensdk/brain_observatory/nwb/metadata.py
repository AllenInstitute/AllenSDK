import os
from pynwb import get_type_map, NWBFile, register_class, docval, load_namespaces, popargs, get_class
from pynwb.spec import NWBNamespaceBuilder, NWBGroupSpec, NWBAttributeSpec, NWBDatasetSpec
from pynwb.file import LabMetaData
from marshmallow import fields

from allensdk.brain_observatory.behavior.schemas import OphysBehaviorMetaDataSchema, OphysBehaviorTaskParametersSchema, STYPE_DICT, TYPE_DICT


def extract_from_schema(schema):

    # Extract fields from Schema:
    docval_list = [{'name': 'name', 'type': str, 'doc': 'name'}]
    attributes = []
    nwbfields_list = []
    for name, val in schema().fields.items():
        if type(val) == fields.List:
            attributes.append(NWBAttributeSpec(name=name, dtype=STYPE_DICT[type(val)], doc=val.metadata['doc'], shape=val.metadata['shape']))
        else:
            attributes.append(NWBAttributeSpec(name=name, dtype=STYPE_DICT[type(val)], doc=val.metadata['doc']))
        docval_list.append({'name': name, 'type': TYPE_DICT[type(val)], 'doc': val.metadata['doc']})
        nwbfields_list.append(name)

    return docval_list, attributes, nwbfields_list


def load_LabMetaData_extension(schema, prefix):

    docval_list, attributes, nwbfields_list = extract_from_schema(schema)
    neurodata_type = schema.neurodata_type
    outdir = os.path.abspath(os.path.dirname(__file__))
    ns_path = '%s_namespace.yaml' % prefix

    # Read spec and load namespace:
    ns_abs_path = os.path.join(outdir, ns_path)
    load_namespaces(ns_abs_path)

    @register_class(neurodata_type, prefix)
    class ExtensionClass(LabMetaData):
        __nwbfields__ = tuple(nwbfields_list)

        @docval(*docval_list)
        def __init__(self, **kwargs):
            name = kwargs.pop('name')
            super(ExtensionClass, self).__init__(name=name)
            for attr, val in kwargs.items():
                setattr(self, attr, val)

    return ExtensionClass


def create_LabMetaData_extension_from_schemas(schema_list, prefix):

    # Initializations:
    outdir = os.path.abspath(os.path.dirname(__file__))
    ext_source = '%s_extension.yaml' % prefix
    ns_path = '%s_namespace.yaml' % prefix
    neurodata_type_list_as_str = str([schema.neurodata_type for schema in schema_list])
    extension_doc = 'LabMetaData extensions: {neurodata_type_list_as_str} ({prefix})'.format(neurodata_type_list_as_str=neurodata_type_list_as_str, prefix=prefix)
    ns_builder = NWBNamespaceBuilder(extension_doc, prefix)

    for schema in schema_list:
        docval_list, attributes, nwbfields_list = extract_from_schema(schema)
        neurodata_type = schema.neurodata_type

        # Build the spec:
        ext_group_spec = NWBGroupSpec(
            neurodata_type_def=neurodata_type,
            neurodata_type_inc='LabMetaData',
            doc=extension_doc,
            attributes=attributes)

        # Add spec to builder:

        ns_builder.add_spec(ext_source, ext_group_spec)
    
    # Export spec
    ns_builder.export(ns_path, outdir=outdir)


if __name__ == "__main__":

    # Run this module to regenerate the extension yaml files into this dir:
    prefix = 'AIBS_ophys_behavior'
    create_LabMetaData_extension_from_schemas([OphysBehaviorMetaDataSchema, OphysBehaviorTaskParametersSchema], prefix)
