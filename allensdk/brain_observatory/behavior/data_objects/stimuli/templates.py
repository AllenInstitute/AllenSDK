import os
import numpy as np
from typing import Optional, List

import imageio
from pynwb import NWBFile

from allensdk.brain_observatory import nwb
from allensdk.brain_observatory.behavior.data_files import StimulusFile
from allensdk.brain_observatory.behavior.data_objects import DataObject
from allensdk.brain_observatory.behavior.data_objects.base \
    .readable_interfaces import \
    StimulusFileReadableInterface, NwbReadableInterface
from allensdk.brain_observatory.behavior.data_objects.base\
    .writable_interfaces import \
    NwbWritableInterface
from allensdk.brain_observatory.behavior.data_objects.stimuli.presentations \
    import \
    Presentations
from allensdk.brain_observatory.behavior.stimulus_processing import \
    get_stimulus_templates
from allensdk.brain_observatory.behavior.data_objects.stimuli \
    .stimulus_templates import \
    StimulusTemplate, StimulusTemplateFactory
from allensdk.brain_observatory.behavior.write_nwb.extensions\
    .stimulus_template.ndx_stimulus_template import \
    StimulusTemplateExtension
from allensdk.internal.core.lims_utilities import safe_system_path


class Templates(DataObject, StimulusFileReadableInterface,
                NwbReadableInterface, NwbWritableInterface):
    def __init__(self, templates: StimulusTemplate):
        super().__init__(name='stimulus_templates', value=templates)

    @classmethod
    def from_stimulus_file(
            cls, stimulus_file: StimulusFile,
            limit_to_images: Optional[List] = None) -> "Templates":
        """Get stimulus templates (movies, scenes) for behavior session."""

        # TODO: Eventually the `grating_images_dict` should be provided by the
        #       BehaviorLimsExtractor/BehaviorJsonExtractor classes.
        #       - NJM 2021/2/23

        gratings_dir = "/allen/programs/braintv/production/visualbehavior"
        gratings_dir = os.path.join(gratings_dir,
                                    "prod5/project_VisualBehavior")
        grating_images_dict = {
            "gratings_0.0": {
                "warped": np.asarray(imageio.imread(
                    safe_system_path(os.path.join(gratings_dir,
                                                  "warped_grating_0.png")))),
                "unwarped": np.asarray(imageio.imread(
                    safe_system_path(os.path.join(
                        gratings_dir, "masked_unwarped_grating_0.png"))))
            },
            "gratings_90.0": {
                "warped": np.asarray(imageio.imread(
                    safe_system_path(os.path.join(gratings_dir,
                                                  "warped_grating_90.png")))),
                "unwarped": np.asarray(imageio.imread(
                    safe_system_path(os.path.join(
                        gratings_dir, "masked_unwarped_grating_90.png"))))
            },
            "gratings_180.0": {
                "warped": np.asarray(imageio.imread(
                    safe_system_path(os.path.join(gratings_dir,
                                                  "warped_grating_180.png")))),
                "unwarped": np.asarray(imageio.imread(
                    safe_system_path(os.path.join(
                        gratings_dir, "masked_unwarped_grating_180.png"))))
            },
            "gratings_270.0": {
                "warped": np.asarray(imageio.imread(
                    safe_system_path(os.path.join(gratings_dir,
                                                  "warped_grating_270.png")))),
                "unwarped": np.asarray(imageio.imread(
                    safe_system_path(os.path.join(
                        gratings_dir, "masked_unwarped_grating_270.png"))))
            }
        }

        pkl = stimulus_file.data
        t = get_stimulus_templates(pkl=pkl,
                                   grating_images_dict=grating_images_dict,
                                   limit_to_images=limit_to_images)
        return Templates(templates=t)

    @classmethod
    def from_nwb(cls, nwbfile: NWBFile) -> "Templates":
        image_set_name = list(nwbfile.stimulus_template.keys())[0]
        image_data = list(nwbfile.stimulus_template.values())[0]

        image_attributes = [{'image_name': image_name}
                            for image_name in image_data.control_description]
        t = StimulusTemplateFactory.from_processed(
            image_set_name=image_set_name, image_attributes=image_attributes,
            warped=image_data.data[:], unwarped=image_data.unwarped[:]
        )
        return Templates(templates=t)

    def to_nwb(self, nwbfile: NWBFile,
               stimulus_presentations: Presentations) -> NWBFile:
        stimulus_templates = self.value

        unwarped_images = []
        warped_images = []
        image_names = []
        for image_name, image_data in stimulus_templates.items():
            image_names.append(image_name)
            unwarped_images.append(image_data.unwarped)
            warped_images.append(image_data.warped)

        image_index = np.zeros(len(image_names))
        image_index[:] = np.nan

        visual_stimulus_image_series = \
            StimulusTemplateExtension(
                name=stimulus_templates.image_set_name,
                data=warped_images,
                unwarped=unwarped_images,
                control=list(range(len(image_names))),
                control_description=image_names,
                unit='NA',
                format='raw',
                timestamps=image_index)

        nwbfile.add_stimulus_template(visual_stimulus_image_series)

        # Add index for this template to NWB in-memory object:
        nwb_template = nwbfile.stimulus_template[
            stimulus_templates.image_set_name]
        stimulus_index = stimulus_presentations.value[
            stimulus_presentations.value[
                'image_set'] == nwb_template.name]
        nwb.add_stimulus_index(nwbfile, stimulus_index, nwb_template)

        return nwbfile
