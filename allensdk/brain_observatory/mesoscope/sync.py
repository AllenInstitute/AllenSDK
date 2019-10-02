from allensdk.brain_observatory.sync_dataset import Dataset as SyncDataset  # NOQA: E402
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

def get_sync_data(self):
        # let's fix line labels where they are off #
        sync_file = self.get_sync_file()
        sync_dataset = SyncDataset(sync_file)
        line_labels = sync_dataset.line_labels

        wrong_labels = ['vsync_2p', 'stim_vsync', 'photodiode', 'acq_trigger', 'cam1', 'cam2',]
        correct_labels = ['2p_vsync', '', 'stim_vsync', '', 'stim_photodiode', 'acq_trigger', '', '', 'cam1_exposure', 'cam2_exposure', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', 'lick_sensor']

        if line_labels != correct_labels :
            if all([line_labels[i] == '' for i in range(len(line_labels))]) == True :
                logger.warning('Line labels are empty, replacing with defaults')
                line_labels = correct_labels
            else:
                logger.warning('Some line labels are incorrect, renaming using defaults')
                i = 0
                while i < len(line_labels):
                    line = line_labels[i]
                    if line in wrong_labels :
                        line_labels.pop(i)
                        line_labels.insert(i, correct_labels[i])
                    i +=1

        sync_dataset.line_labels = line_labels
        meta_data = sync_dataset.meta_data
        sample_freq = meta_data['ni_daq']['counter_output_freq']

        # 2P vsyncs
        vs2p_r = sync_dataset.get_rising_edges('2p_vsync')
        vs2p_f = sync_dataset.get_falling_edges('2p_vsync')  # new sync may be able to do units = 'sec', so conversion can be skipped
        frames_2p = vs2p_r / sample_freq
        vs2p_fsec = vs2p_f / sample_freq

        # use rising edge for Scientifica and Mesoscope falling edge for Nikon http://confluence.corp.alleninstitute.org/display/IT/Ophys+Time+Sync
        stimulus_times_no_monitor_delay = sync_dataset.get_rising_edges('stim_vsync') / sample_freq

        if 'lick_times' in meta_data['line_labels']:
            lick_times = sync_dataset.get_rising_edges('lick_1') / sample_freq
        elif 'lick_sensor' in meta_data['line_labels']:
            lick_times = sync_dataset.get_rising_edges('lick_sensor') / sample_freq
        else:
            lick_times = None
        if '2p_trigger' in meta_data['line_labels']:
            trigger = sync_dataset.get_rising_edges('2p_trigger') / sample_freq
        elif 'acq_trigger' in meta_data['line_labels']:
            trigger = sync_dataset.get_rising_edges('acq_trigger') / sample_freq
        if 'stim_photodiode' in meta_data['line_labels']:
            a = sync_dataset.get_rising_edges('stim_photodiode') / sample_freq
            b = sync_dataset.get_falling_edges('stim_photodiode') / sample_freq
            stim_photodiode = sorted(list(a)+list(b))
        elif 'photodiode' in meta_data['line_labels']:
            a = sync_dataset.get_rising_edges('photodiode') / sample_freq
            b = sync_dataset.get_falling_edges('photodiode') / sample_freq
            stim_photodiode = sorted(list(a)+list(b))
        if 'cam1_exposure' in meta_data['line_labels']:
            eye_tracking = sync_dataset.get_rising_edges('cam1_exposure') / sample_freq
        elif 'eye_tracking' in meta_data['line_labels']:
            eye_tracking = sync_dataset.get_rising_edges('eye_tracking') / sample_freq
        if 'cam2_exposure' in meta_data['line_labels']:
            behavior_monitoring = sync_dataset.get_rising_edges('cam2_exposure') / sample_freq
        elif 'behavior_monitoring' in meta_data['line_labels']:
            behavior_monitoring = sync_dataset.get_rising_edges('behavior_monitoring') / sample_freq

        sync_data = {'ophys_frames': frames_2p,
                     'lick_times': lick_times,
                     'ophys_trigger': trigger,
                     'eye_tracking': eye_tracking,
                     'behavior_monitoring': behavior_monitoring,
                     'stim_photodiode': stim_photodiode,
                     'stimulus_times_no_delay': stimulus_times_no_monitor_delay,
                     }

        return sync_data
