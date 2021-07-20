import warnings

import numpy as np
import pandas as pd
from typing import List, Optional

import pynwb
from pynwb import NWBFile

from allensdk.brain_observatory.behavior.data_objects import DataObject
from allensdk.brain_observatory.behavior.data_objects.base \
    .readable_interfaces import \
    LimsReadableInterface, JsonReadableInterface, NwbReadableInterface
from allensdk.brain_observatory.behavior.data_objects.base\
    .writable_interfaces import \
    NwbWritableInterface
from allensdk.brain_observatory.behavior.schemas import \
    OphysEyeTrackingRigMetadataSchema
from allensdk.brain_observatory.nwb import load_pynwb_extension
from allensdk.internal.api import PostgresQueryMixin


class RigGeometry(DataObject, LimsReadableInterface, JsonReadableInterface,
                  NwbReadableInterface, NwbWritableInterface):
    def __init__(self, equipment: str,
                 monitor_position_mm: List,
                 monitor_rotation_deg: List,
                 camera_position_mm: List,
                 camera_rotation_deg: List,
                 led_position: List):
        super().__init__(name='rig_geometry', value=self)
        self._monitor_position_mm = monitor_position_mm
        self._monitor_rotation_deg = monitor_rotation_deg
        self._camera_position_mm = camera_position_mm
        self._camera_rotation_deg = camera_rotation_deg
        self._led_position = led_position
        self._equipment = equipment

    @property
    def monitor_position_mm(self):
        return self._monitor_position_mm

    @property
    def monitor_rotation_deg(self):
        return self._monitor_rotation_deg

    @property
    def camera_position_mm(self):
        return self._camera_position_mm

    @property
    def camera_rotation_deg(self):
        return self._camera_rotation_deg

    @property
    def led_position(self):
        return self._led_position

    @property
    def equipment(self):
        return self._equipment

    def to_nwb(self, nwbfile: NWBFile) -> NWBFile:
        eye_tracking_rig_mod = pynwb.ProcessingModule(
            name='eye_tracking_rig_metadata',
            description='Eye tracking rig metadata module')

        nwb_extension = load_pynwb_extension(
            OphysEyeTrackingRigMetadataSchema, 'ndx-aibs-behavior-ophys')

        rig_metadata = nwb_extension(
            name="eye_tracking_rig_metadata",
            equipment=self._equipment,
            monitor_position=self._monitor_position_mm,
            monitor_position__unit_of_measurement="mm",
            camera_position=self._camera_position_mm,
            camera_position__unit_of_measurement="mm",
            led_position=self._led_position,
            led_position__unit_of_measurement="mm",
            monitor_rotation=self._monitor_rotation_deg,
            monitor_rotation__unit_of_measurement="deg",
            camera_rotation=self._camera_rotation_deg,
            camera_rotation__unit_of_measurement="deg"
        )

        eye_tracking_rig_mod.add_data_interface(rig_metadata)
        nwbfile.add_processing_module(eye_tracking_rig_mod)
        return nwbfile

    @classmethod
    def from_nwb(cls, nwbfile: NWBFile) -> "RigGeometry":
        try:
            et_mod = \
                nwbfile.get_processing_module("eye_tracking_rig_metadata")
        except KeyError as e:
            warnings.warn("This ophys session "
                          f"'{int(nwbfile.identifier)}' has no eye "
                          f"tracking rig metadata. (NWB error: {e})")
            raise

        meta = et_mod.get_data_interface("eye_tracking_rig_metadata")

        monitor_position = meta.monitor_position[:]
        monitor_position = (monitor_position.tolist()
                            if isinstance(monitor_position, np.ndarray)
                            else monitor_position)

        monitor_rotation = meta.monitor_rotation[:]
        monitor_rotation = (monitor_rotation.tolist()
                            if isinstance(monitor_rotation, np.ndarray)
                            else monitor_rotation)

        camera_position = meta.camera_position[:]
        camera_position = (camera_position.tolist()
                           if isinstance(camera_position, np.ndarray)
                           else camera_position)

        camera_rotation = meta.camera_rotation[:]
        camera_rotation = (camera_rotation.tolist()
                           if isinstance(camera_rotation, np.ndarray)
                           else camera_rotation)

        led_position = meta.led_position[:]
        led_position = (led_position.tolist()
                        if isinstance(led_position, np.ndarray)
                        else led_position)

        rig_geometry = {
            f"monitor_position_{meta.monitor_position__unit_of_measurement}":
                monitor_position,
            f"camera_position_{meta.camera_position__unit_of_measurement}":
                camera_position,
            "led_position": led_position,
            f"monitor_rotation_{meta.monitor_rotation__unit_of_measurement}":
                monitor_rotation,
            f"camera_rotation_{meta.camera_rotation__unit_of_measurement}":
                camera_rotation,
            "equipment": meta.equipment
        }
        return RigGeometry(**rig_geometry)

    @classmethod
    def from_json(cls, dict_repr: dict) -> "RigGeometry":
        return RigGeometry(**dict_repr['eye_tracking_rig_geometry'])

    @classmethod
    def from_lims(cls, ophys_experiment_id: int,
                  lims_db: PostgresQueryMixin) -> Optional["RigGeometry"]:
        query = f'''
            SELECT oec.*, oect.name as config_type, equipment.name as 
            equipment_name
            FROM ophys_sessions os
            JOIN observatory_experiment_configs oec ON oec.equipment_id = 
            os.equipment_id
            JOIN observatory_experiment_config_types oect ON oect.id = 
            oec.observatory_experiment_config_type_id
            JOIN ophys_experiments oe ON oe.ophys_session_id = os.id
            JOIN equipment ON equipment.id = oec.equipment_id
            WHERE oe.id = {ophys_experiment_id} AND 
                oec.active_date <= os.date_of_acquisition AND
                oect.name IN ('eye camera position', 'led position', 'screen position')
        '''  # noqa E501
        # Get the raw data
        rig_geometry = pd.read_sql(query, lims_db.get_connection())

        if rig_geometry.empty:
            # There is no rig geometry for this experiment
            return None

        # Map the config types to new names
        rig_geometry_config_type_map = {
            'eye camera position': 'camera',
            'screen position': 'monitor',
            'led position': 'led'
        }
        rig_geometry['config_type'] = rig_geometry['config_type'] \
            .map(rig_geometry_config_type_map)

        # Select the most recent config
        # that precedes the date_of_acquisition for this experiment
        rig_geometry = rig_geometry.sort_values('active_date', ascending=False)
        rig_geometry = rig_geometry.groupby('config_type') \
            .apply(lambda x: x.iloc[0])

        # Construct dictionary for positions
        position = rig_geometry[['center_x_mm', 'center_y_mm', 'center_z_mm']]
        position.index = [
            f'{v}_position_mm' if v != 'led'
            else f'{v}_position' for v in position.index]
        position = position.to_dict(orient='index')
        position = {
            config_type: [
                values['center_x_mm'],
                values['center_y_mm'],
                values['center_z_mm']
            ]
            for config_type, values in position.items()
        }

        # Construct dictionary for rotations
        rotation = rig_geometry[['rotation_x_deg', 'rotation_y_deg',
                                 'rotation_z_deg']]
        rotation = rotation[rotation.index != 'led']
        rotation.index = [f'{v}_rotation_deg' for v in rotation.index]
        rotation = rotation.to_dict(orient='index')
        rotation = {
            config_type: [
                values['rotation_x_deg'],
                values['rotation_y_deg'],
                values['rotation_z_deg']
            ] for config_type, values in rotation.items()
        }

        # Combine the dictionaries
        rig_geometry = {
            **position,
            **rotation,
            'equipment': rig_geometry['equipment_name'].iloc[0]
        }
        return RigGeometry(**rig_geometry)
