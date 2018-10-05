import datetime
from argparse import ArgumentParser
from allensdk.experimental.nwb.stimulus import VisualCodingStimulusAdapter
from allensdk.experimental.nwb import ophys
from allensdk.experimental.nwb.api.ophys_lims_api import OphysLimsApi
from pynwb import NWBFile, NWBHDF5IO, TimeSeries
from pynwb.ophys import OpticalChannel
from pynwb.form.backends.hdf5.h5_utils import H5DataIO


def write_nwb(output_file, stimulus_adapter, ophys_adapter, **compression_opts):
    """Write Visual Coding NWB 2.0 file

    Parameters
    ----------
    output_file : string
        Name of the NWB file.
    stimulus_adapter
        Adapter providing pynwb representations of data derived from stimulus
        files.
    ophys_adapter
        Adapter providing session metadata, roi data, and dff from data
        sources.
    """
    session_metadata = ophys_adapter.session_metadata

    nwbfile = ophys.create_base_nwb(session_metadata)

    nwbfile.add_acquisition(stimulus_adapter.running_speed)

    processing = ophys.add_ophys_module(
        nwbfile, 'visual_coding_pipeline',
        ("Processing module for Allen Brain Observatory: "
         "Visual Coding 2-Photon Experiments"),
        ophys_adapter.dff_source)

    dff_interface = processing.get_data_interface('dff_interface')

    optical_channel = ophys_adapter.get_optical_channel()

    description = "{} field of view in {} at depth {} um".format(
        session_metadata['fov'],
        session_metadata['targeted_structure'],
        session_metadata['imaging_depth_um'])

    imaging_plane = ophys.add_imaging_plane(
        nwbfile, optical_channel, session_metadata, description)

    image_seg = ophys.get_image_segmentation(processing, "segmentation")
    plane_seg = ophys.get_plane_segmentation(image_seg, imaging_plane,
                                             ophys_adapter.roi_mask_dict,
                                             source="segmentation",
                                             description="Segmented cells")

    rt_region = plane_seg.create_roi_table_region(
        description="segmented cells labeled by cell_specimen_id",
        names=[roi for roi in ophys_adapter.roi_mask_dict.keys()])

    dff, t = ophys_adapter.dff_traces
    ophys.get_dff_series(dff_interface, rt_region, dff, t,
                         source=ophys_adapter.dff_source,
                         **compression_opts)

    with NWBHDF5IO(output_file, mode='w') as io:
        io.write(nwbfile)


def main():
    parser = ArgumentParser()
    parser.add_argument("experiment_id", type=int)
    parser.add_argument("nwb_file", type=str)
    parser.add_argument("--compress", action="store_true")

    args = parser.parse_args()

    opts = {}
    if args.compress:
        opts = {"compression": True, "compression_opts": 9}

    api = OphysLimsApi()
    ophys_adapter = ophys.OphysAdapter(args.experiment_id, api)
    pkl = api.get_pickle_file(args.experiment_id)
    sync = api.get_sync_file(args.experiment_id)
    stimulus_adapter = VisualCodingStimulusAdapter(pkl, sync)

    write_nwb(args.nwb_file, stimulus_adapter, ophys_adapter, **opts)

if __name__ == "__main__":
    main()
