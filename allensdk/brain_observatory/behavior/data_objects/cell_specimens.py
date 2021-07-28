import numpy as np
import pandas as pd
from pynwb import NWBFile, ProcessingModule
from pynwb.ophys import OpticalChannel, ImageSegmentation

import allensdk.brain_observatory.roi_masks as roi
from allensdk.brain_observatory.behavior.data_files.dff_file import DFFFile
from allensdk.brain_observatory.behavior.data_objects import DataObject
from allensdk.brain_observatory.behavior.data_objects.base \
    .readable_interfaces import \
    JsonReadableInterface, LimsReadableInterface, NwbReadableInterface, \
    InternalReadableInterface
from allensdk.brain_observatory.behavior.data_objects.base \
    .writable_interfaces import \
    NwbWritableInterface
from allensdk.brain_observatory.behavior.data_objects.dff_traces import \
    DFF_traces
from allensdk.brain_observatory.behavior.data_objects.metadata \
    .ophys_experiment_metadata.field_of_view_shape import \
    FieldOfViewShape
from allensdk.brain_observatory.behavior.data_objects.metadata \
    .ophys_experiment_metadata.imaging_plane import \
    ImagingPlane
from allensdk.brain_observatory.behavior.data_objects.timestamps \
    .ophys_timestamps import \
    OphysTimestamps
from allensdk.brain_observatory.nwb import CELL_SPECIMEN_COL_DESCRIPTIONS
from allensdk.internal.api import PostgresQueryMixin


class CellSpecimenMeta(DataObject, InternalReadableInterface,
                       JsonReadableInterface, NwbReadableInterface):
    def __init__(self, imaging_plane: ImagingPlane, emission_lambda=520.0):
        super().__init__(name='cell_spcimen_meta', value=self)
        self._emission_lambda = emission_lambda
        self._imaging_plane = imaging_plane

    @property
    def emission_lambda(self):
        return self._emission_lambda

    @property
    def imaging_plane(self):
        return self._imaging_plane

    @classmethod
    def from_internal(cls, ophys_experiment_id: int,
                      lims_db: PostgresQueryMixin,
                      ophys_timestamps: OphysTimestamps) -> "CellSpecimenMeta":
        imaging_plane_meta = ImagingPlane.from_internal(
            ophys_experiment_id=ophys_experiment_id, lims_db=lims_db,
            ophys_timestamps=ophys_timestamps)
        return cls(imaging_plane=imaging_plane_meta)

    @classmethod
    def from_json(cls, dict_repr: dict,
                  ophys_timestamps: OphysTimestamps) -> "CellSpecimenMeta":
        imaging_plane_meta = ImagingPlane.from_json(
            dict_repr=dict_repr, ophys_timestamps=ophys_timestamps)
        return cls(imaging_plane=imaging_plane_meta)

    @classmethod
    def from_nwb(cls, nwbfile: NWBFile) -> "CellSpecimenMeta":
        ophys_module = nwbfile.processing['ophys']
        image_seg = ophys_module.data_interfaces['image_segmentation']
        plane_segmentations = image_seg.plane_segmentations
        cell_specimen_table = plane_segmentations['cell_specimen_table']

        imaging_plane = cell_specimen_table.imaging_plane
        optical_channel = imaging_plane.optical_channel[0]
        emission_lambda = optical_channel.emission_lambda

        imaging_plane = ImagingPlane.from_nwb(nwbfile=nwbfile)
        return CellSpecimenMeta(emission_lambda=emission_lambda,
                                imaging_plane=imaging_plane)


class CellSpecimens(DataObject, LimsReadableInterface,
                    JsonReadableInterface, NwbReadableInterface,
                    NwbWritableInterface):
    def __init__(self, cell_specimen_table: pd.DataFrame,
                 meta: CellSpecimenMeta,
                 dff_traces: DFF_traces):
        super().__init__(name='cell_specimen_table', value=self)
        self._meta = meta
        self._cell_specimen_table = cell_specimen_table
        self._dff_traces = dff_traces

    @property
    def table(self) -> pd.DataFrame:
        return self._cell_specimen_table

    @property
    def roi_masks(self) -> pd.DataFrame:
        return self._cell_specimen_table[['cell_roi_id', 'roi_mask']]

    @property
    def meta(self) -> CellSpecimenMeta:
        return self._meta

    @property
    def dff_traces(self) -> pd.DataFrame:
        df = self.table[['cell_roi_id']].join(self._dff_traces.value,
                                              on='cell_roi_id')
        return df

    @classmethod
    def from_lims(cls, ophys_experiment_id: int,
                  lims_db: PostgresQueryMixin,
                  ophys_timestamps: OphysTimestamps) -> "CellSpecimens":
        def _get_ophys_cell_segmentation_run_id() -> int:
            """Get the ophys cell segmentation run id associated with an
            ophys experiment id"""
            query = """
                    SELECT oseg.id
                    FROM ophys_experiments oe
                    JOIN ophys_cell_segmentation_runs oseg
                    ON oe.id = oseg.ophys_experiment_id
                    WHERE oseg.current = 't'
                    AND oe.id = {};
                    """.format(ophys_experiment_id)
            return lims_db.fetchone(query, strict=True)

        def _get_cell_specimen_table():
            ophys_cell_seg_run_id = _get_ophys_cell_segmentation_run_id()
            query = """
                    SELECT *
                    FROM cell_rois cr
                    WHERE cr.ophys_cell_segmentation_run_id = {};
                    """.format(ophys_cell_seg_run_id)
            initial_cs_table = pd.read_sql(query, lims_db.get_connection())
            cst = initial_cs_table.rename(
                columns={'id': 'cell_roi_id', 'mask_matrix': 'roi_mask'})
            cst.drop(['ophys_experiment_id',
                      'ophys_cell_segmentation_run_id'],
                     inplace=True, axis=1)
            cst = cst.to_dict()
            fov_shape = FieldOfViewShape.from_lims(
                ophys_experiment_id=ophys_experiment_id, lims_db=lims_db)
            cst = cls._postprocess(
                cell_specimen_table=cst, fov_shape=fov_shape)
            return cst

        def _get_dff_traces():
            dff_file = DFFFile.from_lims(
                ophys_experiment_id=ophys_experiment_id,
                db=lims_db)
            return DFF_traces.from_data_file(
                dff_file=dff_file,
                cell_roi_id_list=cell_specimen_table['cell_roi_id'].values)

        cell_specimen_table = _get_cell_specimen_table()
        meta = CellSpecimenMeta.from_internal(
            ophys_experiment_id=ophys_experiment_id, lims_db=lims_db,
            ophys_timestamps=ophys_timestamps)
        dff_traces = _get_dff_traces()

        return cls(cell_specimen_table=cell_specimen_table, meta=meta,
                   dff_traces=dff_traces)

    @classmethod
    def from_json(cls, dict_repr: dict,
                  ophys_timestamps: OphysTimestamps) -> "CellSpecimens":
        cell_specimen_table = dict_repr['cell_specimen_table_dict']
        fov_shape = FieldOfViewShape.from_json(dict_repr=dict_repr)
        cell_specimen_table = cls._postprocess(
            cell_specimen_table=cell_specimen_table, fov_shape=fov_shape)

        def _get_dff_traces():
            dff_file = DFFFile.from_json(dict_repr=dict_repr)
            return DFF_traces.from_data_file(
                dff_file=dff_file,
                cell_roi_id_list=cell_specimen_table['cell_roi_id'].values)

        meta = CellSpecimenMeta.from_json(dict_repr=dict_repr,
                                          ophys_timestamps=ophys_timestamps)
        dff_traces = _get_dff_traces()
        return cls(cell_specimen_table=cell_specimen_table, meta=meta,
                   dff_traces=dff_traces)

    @classmethod
    def from_nwb(cls, nwbfile: NWBFile,
                 filter_invalid_rois=False) -> "CellSpecimens":
        # NOTE: ROI masks are stored in full frame width and height arrays
        ophys_module = nwbfile.processing['ophys']
        image_seg = ophys_module.data_interfaces['image_segmentation']
        plane_segmentations = image_seg.plane_segmentations
        cell_specimen_table = plane_segmentations['cell_specimen_table']

        def _read_table(cell_specimen_table):
            df = cell_specimen_table.to_dataframe()

            # Because pynwb stores this field as "image_mask", it is renamed
            # here
            df = df.rename(columns={'image_mask': 'roi_mask'})

            df.index.rename('cell_roi_id', inplace=True)
            df['cell_specimen_id'] = [None if id_ == -1 else id_
                                      for id_ in df['cell_specimen_id'].values]

            df.reset_index(inplace=True)
            df.set_index('cell_specimen_id', inplace=True)

            if filter_invalid_rois:
                df = df[df["valid_roi"]]
            return df

        df = _read_table(cell_specimen_table=cell_specimen_table)
        meta = CellSpecimenMeta.from_nwb(nwbfile=nwbfile)
        dff_traces = DFF_traces.from_nwb(nwbfile=nwbfile)

        return cls(cell_specimen_table=df, meta=meta, dff_traces=dff_traces)

    def to_nwb(self, nwbfile: NWBFile,
               ophys_timestamps: OphysTimestamps) -> NWBFile:
        """
        :param nwbfile
            In-memory nwb file object
        :param ophys_timestamps
            ophys timestamps
        """
        # 1. Add cell specimen table
        cell_roi_table = self.table.reset_index().set_index(
            'cell_roi_id')
        metadata = nwbfile.lab_meta_data['metadata']

        device = nwbfile.get_device()

        # FOV:
        fov_width = metadata.field_of_view_width
        fov_height = metadata.field_of_view_height
        imaging_plane_description = "{} field of view in {} at depth {} " \
                                    "um".format(
            (fov_width, fov_height),
            self._meta.imaging_plane.targeted_structure,  # noqa E501
            metadata.imaging_depth)

        # Optical Channel:
        optical_channel = OpticalChannel(
            name='channel_1',
            description='2P Optical Channel',
            emission_lambda=self._meta.emission_lambda)

        # Imaging Plane:
        imaging_plane = nwbfile.create_imaging_plane(
            name='imaging_plane_1',
            optical_channel=optical_channel,
            description=imaging_plane_description,
            device=device,
            excitation_lambda=self._meta.imaging_plane.excitation_lambda,
            imaging_rate=self._meta.imaging_plane.ophys_frame_rate,
            indicator=self._meta.imaging_plane.indicator,
            location=self._meta.imaging_plane.targeted_structure)

        # Image Segmentation:
        image_segmentation = ImageSegmentation(name="image_segmentation")

        if 'ophys' not in nwbfile.processing:
            ophys_module = ProcessingModule('ophys', 'Ophys processing module')
            nwbfile.add_processing_module(ophys_module)
        else:
            ophys_module = nwbfile.processing['ophys']

        ophys_module.add_data_interface(image_segmentation)

        # Plane Segmentation:
        plane_segmentation = image_segmentation.create_plane_segmentation(
            name='cell_specimen_table',
            description="Segmented rois",
            imaging_plane=imaging_plane)

        for col_name in cell_roi_table.columns:
            # the columns 'roi_mask', 'pixel_mask', and 'voxel_mask' are
            # already defined in the nwb.ophys::PlaneSegmentation Object
            if col_name not in ['id', 'mask_matrix', 'roi_mask',
                                'pixel_mask', 'voxel_mask']:
                # This builds the columns with name of column and description
                # of column both equal to the column name in the cell_roi_table
                plane_segmentation.add_column(
                    col_name,
                    CELL_SPECIMEN_COL_DESCRIPTIONS.get(
                        col_name,
                        "No Description Available"))

        # go through each roi and add it to the plan segmentation object
        for cell_roi_id, table_row in cell_roi_table.iterrows():
            # NOTE: The 'roi_mask' in this cell_roi_table has already been
            # processing by the function from
            # allensdk.brain_observatory.behavior.session_apis.data_io
            # .ophys_lims_api
            # get_cell_specimen_table() method. As a result, the ROI is
            # stored in
            # an array that is the same shape as the FULL field of view of the
            # experiment (e.g. 512 x 512).
            mask = table_row.pop('roi_mask')

            csid = table_row.pop('cell_specimen_id')
            table_row['cell_specimen_id'] = -1 if csid is None else csid
            table_row['id'] = cell_roi_id
            plane_segmentation.add_roi(image_mask=mask, **table_row.to_dict())

        # 2. Add DFF traces
        self._dff_traces.to_nwb(nwbfile=nwbfile,
                                ophys_timestamps=ophys_timestamps)

        return nwbfile

    @staticmethod
    def _postprocess(cell_specimen_table: dict,
                     fov_shape: FieldOfViewShape) -> pd.DataFrame:
        """Converts raw cell_specimen_table dict to dataframe"""
        cell_specimen_table = pd.DataFrame.from_dict(
            cell_specimen_table).set_index(
            'cell_roi_id').sort_index()
        fov_width = fov_shape.width
        fov_height = fov_shape.height

        # Convert cropped ROI masks to uncropped versions
        roi_mask_list = []
        for cell_roi_id, table_row in cell_specimen_table.iterrows():
            # Deserialize roi data into AllenSDK RoiMask object
            curr_roi = roi.RoiMask(image_w=fov_width, image_h=fov_height,
                                   label=None, mask_group=-1)
            curr_roi.x = table_row['x']
            curr_roi.y = table_row['y']
            curr_roi.width = table_row['width']
            curr_roi.height = table_row['height']
            curr_roi.mask = np.array(table_row['roi_mask'])
            roi_mask_list.append(curr_roi.get_mask_plane().astype(np.bool))

        cell_specimen_table['roi_mask'] = roi_mask_list
        cell_specimen_table = cell_specimen_table[
            sorted(cell_specimen_table.columns)]

        cell_specimen_table.index.rename('cell_roi_id', inplace=True)
        cell_specimen_table.reset_index(inplace=True)
        cell_specimen_table.set_index('cell_specimen_id', inplace=True)
        return cell_specimen_table
