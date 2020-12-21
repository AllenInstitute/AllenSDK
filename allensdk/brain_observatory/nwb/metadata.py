import os

from marshmallow import fields
import pynwb
from pynwb.spec import NWBNamespaceBuilder, NWBGroupSpec, NWBAttributeSpec

from allensdk.brain_observatory.behavior.schemas import STYPE_DICT, TYPE_DICT


def extract_from_schema(schema):

    if hasattr(schema, 'neurodata_skip'):
        fields_to_skip = schema.neurodata_skip
    else:
        fields_to_skip = set()

    # Extract fields from Schema:
    docval_list = [{'name': 'name', 'type': str, 'doc': 'name'}]
    attributes = []
    nwbfields_list = []
    for name, val in schema().fields.items():

        if name in fields_to_skip:
            continue

        if type(val) == fields.List:
            attributes.append(NWBAttributeSpec(name=name,
                                               dtype=STYPE_DICT[type(val)],
                                               doc=val.metadata['doc'],
                                               shape=val.metadata['shape']))
        else:
            attributes.append(NWBAttributeSpec(name=name,
                                               dtype=STYPE_DICT[type(val)],
                                               doc=val.metadata['doc']))
        docval_list.append({'name': name,
                            'type': TYPE_DICT[type(val)],
                            'doc': val.metadata['doc']})
        nwbfields_list.append(name)

    return docval_list, attributes, nwbfields_list


def load_pynwb_extension(schema, prefix: str):

    docval_list, attributes, nwbfields_list = extract_from_schema(schema)
    neurodata_type = schema.neurodata_type
    outdir = os.path.abspath(os.path.dirname(__file__))
    ns_path = f'{prefix}.namespace.yaml'

    # Read spec and load namespace:
    ns_abs_path = os.path.join(outdir, ns_path)
    pynwb.load_namespaces(ns_abs_path)

    return pynwb.get_class(neurodata_type, prefix)


def create_pynwb_extension_from_schemas(schema_list, prefix: str):

    # Initializations:
    outdir = os.path.abspath(os.path.dirname(__file__))
    ext_source = f'{prefix}.extension.yaml'
    ns_path = f'{prefix}.namespace.yaml'

    extension_doc = ("Allen Institute behavior and optical "
                     "physiology extensions")

    ns_builder = NWBNamespaceBuilder(
        doc=extension_doc,
        name=prefix,
        version="0.2.0",
        author="Allen Institute for Brain Science",
        contact="waynew@alleninstitute.org")

    # Loops through and create NWB custom group specs for schemas found in:
    # allensdk.brain_observatory.behavior.schemas
    for schema in schema_list:
        docval_list, attributes, nwbfields_list = extract_from_schema(schema)

        # Build the spec:
        ext_group_spec = NWBGroupSpec(
            neurodata_type_def=schema.neurodata_type,
            neurodata_type_inc=schema.neurodata_type_inc,
            doc=schema.neurodata_doc,
            attributes=attributes)

        # Add spec to builder:
        ns_builder.add_spec(ext_source, ext_group_spec)

    # Export spec
    ns_builder.export(ns_path, outdir=outdir)
