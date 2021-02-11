import os.path

from pynwb.spec import NWBNamespaceBuilder, export_spec, NWBGroupSpec, NWBDatasetSpec

NAMESPACE = 'ndx-event-detection'


def main():
    # these arguments were auto-generated from your cookiecutter inputs
    ns_builder = NWBNamespaceBuilder(
        doc="""Store the event detection output""",
        name=f"""{NAMESPACE}""",
        version="""0.1.0""",
        author=list(map(str.strip, """Adam Amster""".split(','))),
        contact=list(map(str.strip, """aamster@alleninstitute.org""".split(',')))
    )

    ns_builder.include_type('RoiResponseSeries', namespace='core')
    ns_builder.include_type('DynamicTableRegion', namespace='core')
    ns_builder.include_type('TimeSeries', namespace='core')
    ns_builder.include_type('NWBDataInterface', namespace='core')

    events_spec = NWBGroupSpec(
        neurodata_type_def='EventDetection',
        neurodata_type_inc='RoiResponseSeries',
        name='event_detection',
        doc='Stores event detection output',
        datasets=[
            NWBDatasetSpec(
                name='lambdas',
                dtype='float',
                doc='calculated regularization weights',
                shape=(None,)
            ),
            NWBDatasetSpec(
                name='noise_stds',
                dtype='float',
                doc='calculated noise std deviations',
                shape=(None,)
            )
        ]
    )

    new_data_types = [events_spec]

    # export the spec to yaml files in the spec folder
    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__)))
    export_spec(ns_builder, new_data_types, output_dir)


if __name__ == "__main__":
    # usage: python create_extension_spec.py
    main()