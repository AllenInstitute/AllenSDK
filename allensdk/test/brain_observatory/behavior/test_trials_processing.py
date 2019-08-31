import pytest
import pandas as pd
import numpy as np

from allensdk.internal.api.behavior_lims_api import BehaviorLimsApi
from allensdk.brain_observatory.behavior import trials_processing


@pytest.mark.requires_bamboo
@pytest.mark.parametrize(
    'behavior_experiment_id, ti, expected, exception', [
        (880293569, 5, (90, 90, None), None, ),
        (881236761, 0, None, IndexError, )
    ]
)
def test_get_ori_info_from_trial(behavior_experiment_id, ti, expected, exception, ):
    """was feeling worried that the values would be wrong,
    this helps reaffirm that maybe they are not...

    Notes
    -----
    - i may be rewriting code here but its more a sanity check really...
    """
    stim_output = pd.read_pickle(
        BehaviorLimsApi(behavior_experiment_id).get_behavior_stimulus_file()
    )
    trial_log = stim_output['items']['behavior']['trial_log']

    if exception:
        with pytest.raises(exception):
            trials_processing.get_ori_info_from_trial(trial_log, ti, )
    else:
        assert trials_processing.get_ori_info_from_trial(trial_log, ti, ) == expected


_test_response_latency_0 = np.array([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 0.3669842 , np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 0.41701037, np.nan, np.nan, 0.31692564, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 0.28356898, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 0.33363652, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 0.21683128, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 0.38365788, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])
_test_starttime_0 = np.array([ 19.99986754,  22.9857173 ,  25.25430697,  27.50627203,  30.50876019,  33.54466563,  37.314543  ,  38.06517225,  41.0677066 ,  42.58570486,  44.83770039,  52.3774269 ,  57.63187756,  62.91968583,  66.67286547,  68.17414034,  69.67541489,  74.19590836,  77.19846323,  78.69971244,  80.9516403 ,  83.95418202,  86.22276367,  87.72406035,  89.97592663,  92.97849208,  95.23039781,  97.49898909, 100.50154941, 102.75344423, 104.25473193, 107.25726563, 109.50917136, 111.7610819 , 114.01299179, 116.26490714, 119.26744951, 121.51938378, 124.53862772, 127.55782612, 129.80972512, 132.81226619, 137.31609721, 138.81734546, 141.08594764, 143.33787839, 146.35708545, 149.35962973, 156.86599991, 159.88526245, 163.65508305, 173.44672267, 175.69862167, 178.70117974, 180.95312267, 183.22166712, 186.22421269, 189.24343194, 191.49533542, 195.28190828, 198.28443076, 201.30364616, 203.55556985, 205.82415567, 209.577343  , 213.3472146 , 216.34976979, 218.61835433, 222.38820892, 226.14139112, 233.64779273, 234.39842519, 235.16570215, 237.4176191 , 239.66953991, 241.93811899, 244.94066135, 247.94320564, 249.44448821, 252.44703217, 253.96497334, 254.71559618, 256.2168919 , 263.73995532, 266.74247811, 268.99441912, 271.2629764 , 275.0161833 , 276.51745143, 278.03538907, 280.2873025 , 282.53921593, 284.80780047, 287.81034956, 290.06226106, 291.56355774, 294.56608823, 297.58530492, 299.83719974, 302.08911637, 304.34102787, 308.1109572 , 315.6172597 , 317.11854964, 319.37043805, 321.65571199, 323.90761451, 325.40892787, 327.66080216, 329.91271591, 332.9319358 , 336.68513692, 339.70434687, 341.20561115, 341.95625997, 345.72611233, 347.22738976, 349.4626279 , 357.01902553, 359.27093992, 362.29016334, 365.29272109, 368.31197049, 372.06513441, 373.58308296, 375.83498259, 377.3362504 , 379.58815677, 382.60747224, 387.11118876, 390.11375903, 392.3823596 , 394.63425956, 396.88616336, 399.92207297, 402.92464964, 405.94383939, 408.19575538, 410.46437038, 413.46690377, 416.4694612 , 418.72134384, 420.95662335, 423.95913236, 425.46040562, 427.71231745, 429.98091481, 432.98349662, 434.48471376, 435.98598863, 438.98854735, 442.0077961 , 444.25968163, 446.54494947, 448.02954296, 451.79940525, 454.81863412, 457.07053664, 459.33912952, 461.59103621, 463.85962107, 465.360903  , 467.61282348, 469.8814    , 471.3826691 , 474.38523327, 477.42112684, 479.67303705, 481.94162769, 484.94417229, 487.96340758, 490.21530914, 492.46722064, 494.73581031, 498.48898834, 501.50822074, 503.00949272, 505.2780933 , 507.53000255, 510.54922212, 514.31907961, 516.57098918, 518.83957885, 520.34085308, 523.3600781 , 525.61201879, 529.39853328, 531.6504467 , 533.91904696, 536.17094563, 539.94081562, 542.19272616, 545.19526724, 546.69653986, 548.19781249, 550.44972206, 553.46895029, 556.4714981 , 559.47403694, 562.47658507])

expected_result_0 = np.array([
    np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, 0.85743611, 0.82215855, 0.79754855, 0.77420232, 0.74532604, 0.72504419, 0.73828876, 0.73168092,
    0.73168145, 0.73844044, 0.745635, 0.75997116, 0.73889553, 0.7457893, 0.73212764, 0.72533739, 0., 0., 0., 0., 0., 0., 0., 0.83147215, 0.76759434, 0.76013235, 1.50562915, 1.37576878, 
    1.36403067, 1.3524898, 1.3640296, 1.36377167, 1.35249025, 1.35223636, 1.35223623, 1.31828762, 1.31828779, 1.30726822,
    1.30726804, 1.30703059, 1.28600222, 1.27551339, 1.26541718, 1.27551391, 1.26541723, 1.86854445, 1.78508514, 1.85409645, 1.86822076, 1.86854435,
    1.86854454, 1.88321881, 1.88321885, 1.31756348, 1.33989546, 1.35147388, 0.74517267, 0.75933052, 1.54807324, 1.44950587, 1.43676766, 1.44979704,
    1.46306592, 1.43676702, 1.47718591, 1.50468411, 1.51930166, 1.51930185, 1.51930188, 1.53387944, 1.5642303, 1.59545215, 1.58003398, 1.59580631,
    1.62831513, 0.87666335, 0.85784626, 1.6450693, 1.53453391, 1.54940651, 1.54974049, 1.56423021, 1.57968714, 1.5796865, 1.59545254, 1.5800338,
    1.53420629, 1.49127149, 0.78984375, 0.80576787, 0.82234767, 0.80576784, 0.83089596, 1.64507108, 1.51930204, 1.51930202, 1.50468432, 1.49096252,
    1.49065321, 1.46336336, 1.46306626, 1.4765797, 1.50468436, 1.50468414, 1.4903434, 1.44979783, 1.46336463, 0.78160518, 0.77403664, 0.77403649,
    0.76661287, 0.75932993, 0.74501851, 0.74501813, 0.74486366, 0.74501799, 0.75202743, 0.7593303, 0.75234155, 0.73168169, 0.7524993, 0.74548119,
    0.74517234, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
    0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
])

expected_result_1 = np.array([np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, 1.811146, 1.944290, 1.898119, 1.811146, 1.733465, 1.771897, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 2.417308, 2.047210, 1.994977, 3.890695, 3.321282, 3.253684, 3.190197, 3.190196, 3.255157, 3.255157, 1.853143, 1.898129, 1.897124, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 2.153860, 1.855050, 1.945345, 2.044882, 2.155151, 2.279434, 2.344817, 2.279435, 2.347877, 2.574765, 0.000000, 0.000000, 0.000000, 3.191613, 2.492687, 2.418930, 2.494413, 2.572924, 2.346344, 2.492686, 2.492686, 2.346343, 2.279434, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 2.574758, 2.157737, 2.217599, 2.157739, 2.214870, 2.279435, 2.346341,
 2.346345, 2.346345, 2.417310, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 2.752063, 2.213507, 2.277990, 2.343290, 2.344815, 2.213503, 1.992768, 2.153859, 2.097345, 2.152573, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,])

@pytest.mark.parametrize('kwargs, expected', [
    (
        {
            'response_latency': _test_response_latency_0,
            'starttime': _test_starttime_0,
            'trial_window': 15,
            'initial_trials': 10,
        }, 
        expected_result_0,
    ),
    (
        {
            'response_latency': _test_response_latency_0,
            'starttime': _test_starttime_0,
            'trial_window': 5,
            'initial_trials': 10,
        }, 
        expected_result_1,
    ),
])
def test_calculate_reward_rate(kwargs, expected):
    assert np.allclose(
        trials_processing.calculate_reward_rate(**kwargs),
        expected, 
    ), "calculated reward rate should match expected reward rate :("

def trial_data_and_expectation_0():
    test_trial = {
        'index': 3,
        'cumulative_rewards': 1,
        'licks': [(318.2737866026219, 18736),
        (318.4235244484611, 18745),
        (318.55351991075554, 18753),
        (318.6735239364698, 18760),
        (318.8235420609609, 18769),
        (318.9733899117824, 18778),
        (319.153503175955, 18789),
        (319.35351008052305, 18801),
        (321.24372627834714, 18914),
        (321.3438153063156, 18920),
        (321.49348118080985, 18929),
        (321.6237259097134, 18937)],
        'stimulus_changes': [(('im065', 'im065'),
        ('im062', 'im062'),
        317.76644976765834,
        18706)],
        'success': False,
        'cumulative_volume': 0.005,
        'trial_params': {'catch': False, 'auto_reward': True, 'change_time': 5},
        'rewards': [(0.005, 317.92325660388286, 18715)],
        'events': [['trial_start', '', 314.0120642698258, 18481],
        ['initial_blank', 'enter', 314.01216666808074, 18481],
        ['initial_blank', 'exit', 314.0122573636779, 18481],
        ['pre_change', 'enter', 314.01233489378524, 18481],
        ['pre_change', 'exit', 314.0124103759274, 18481],
        ['stimulus_window', 'enter', 314.01248819860115, 18481],
        ['stimulus_changed', '', 317.7666744586863, 18706],
        ['auto_reward', '', 317.76681547571155, 18706],
        ['response_window', 'enter', 317.9231027139341, 18715],
        ['response_window', 'exit', 318.532233361527, 18752],
        ['miss', '', 318.5324346472395, 18752],
        ['stimulus_window', 'exit', 322.0351179203675, 18962],
        ['no_lick', 'exit', 322.0352864386384, 18962],
        ['trial_end', '', 322.0353750862705, 18962]]
    }
    
    expected_result = {
        'reward_volume': 0.005,
        'hit': False,
        'false_alarm': False,
        'miss': False,
        'sham_change': False,
        'stimulus_change': True,
        'aborted': False,
        'go': False,
        'catch': False,
        'auto_rewarded': True,
        'correct_reject': False
    }

    return test_trial, expected_result

def trial_data_and_expectation_1():
    test_trial = {
        'index': 4,
        'cumulative_rewards': 1,
        'licks': [(324.1935569847751, 19091),
        (324.34329131981696, 19100),
        (324.49368158882305, 19109)],
        'stimulus_changes': [],
        'success': False,
        'cumulative_volume': 0.005,
        'trial_params': {'catch': False, 'auto_reward': True, 'change_time': 6},
        'rewards': [],
        'events': [['trial_start', '', 322.2688823113451, 18976],
        ['initial_blank', 'enter', 322.2689858798658, 18976],
        ['initial_blank', 'exit', 322.26907599033007, 18976],
        ['pre_change', 'enter', 322.2691523501716, 18976],
        ['pre_change', 'exit', 322.26922900257955, 18976],
        ['stimulus_window', 'enter', 322.26930536242105, 18976],
        ['early_response', '', 324.1937059010944, 19091],
        ['abort', '', 324.1937848940339, 19091],
        ['timeout', 'enter', 324.19388963282034, 19091],
        ['timeout', 'exit', 324.8042502297378, 19128],
        ['trial_end', '', 324.80448691598986, 19128]]
    }

    expected_result = {
        'reward_volume': 0,
        'hit': False,
        'false_alarm': False,
        'miss': False,
        'sham_change': False,
        'stimulus_change': False,
        'aborted': True,
        'go': False,
        'catch': False,
        'auto_rewarded': False,
        'correct_reject': False
    }

    return test_trial, expected_result

def trial_data_and_expectation_2():
    test_trial = {
        'index': 51,
        'cumulative_rewards': 11,
        'licks': [(542.6200214334176, 32186),
        (542.7097825733969, 32191),
        (542.8597161461861, 32200),
        (542.9599280520605, 32206),
        (543.059708422432, 32212),
        (543.15998088956, 32218),
        (543.2899491431752, 32226),
        (543.4098750536493, 32233),
        (543.5197477960238, 32240),
        (543.6596846660369, 32248),
        (543.7699336488565, 32255),
        (543.8897463361172, 32262),
        (544.0196821148575, 32270),
        (544.13974055793, 32277),
        (544.2596729048659, 32284),
        (544.3896745110557, 32292),
        (544.5397306691843, 32301)],
        'stimulus_changes': [(('im069', 'im069'),
        ('im085', 'im085'),
        542.2007438794369,
        32161)],
        'success': True,
        'cumulative_volume': 0.067,
        'trial_params': {'catch': False, 'auto_reward': False, 'change_time': 4},
        'rewards': [(0.007, 542.620156599114, 32186)],
        'events': [['trial_start', '', 539.1971251251088, 31981],
        ['initial_blank', 'enter', 539.197228401063, 31981],
        ['initial_blank', 'exit', 539.1973220223246, 31981],
        ['pre_change', 'enter', 539.1974007226976, 31981],
        ['pre_change', 'exit', 539.197477667672, 31981],
        ['stimulus_window', 'enter', 539.1975575383109, 31981],
        ['stimulus_changed', '', 542.2009428246179, 32161],
        ['response_window', 'enter', 542.3661398812824, 32171],
        ['hit', '', 542.6201402153932, 32186],
        ['response_window', 'exit', 542.9666720011281, 32207],
        ['stimulus_window', 'exit', 546.4695340323526, 32417],
        ['no_lick', 'exit', 546.4696966992947, 32417],
        ['trial_end', '', 546.4697827138287, 32417]]
    }

    expected_result = {
        'reward_volume': 0.007,
        'hit': True,
        'false_alarm': False,
        'miss': False,
        'sham_change': False,
        'stimulus_change': True,
        'aborted': False,
        'go': True,
        'catch': False,
        'auto_rewarded': False,
        'correct_reject': False
    }

    return test_trial, expected_result


@pytest.mark.parametrize("data_exp_getter", [
    trial_data_and_expectation_0, 
    trial_data_and_expectation_1,
    trial_data_and_expectation_2
])
def test_trial_data_from_log(data_exp_getter):
    data, expectation = data_exp_getter()
    assert trials_processing.trial_data_from_log(data) == expectation


def test_get_trial_timing():
    event_dict = {
        ('trial_start', ''): {'rebased_time': 306.4785879253758, 'frame': 18075},
        ('initial_blank', 'enter'): {'rebased_time': 306.47868008512637,
        'frame': 18075},
        ('initial_blank', 'exit'): {'rebased_time': 306.4787637603285,
        'frame': 18075},
        ('pre_change', 'enter'): {'rebased_time': 306.47883573270514, 'frame': 18075},
        ('pre_change', 'exit'): {'rebased_time': 306.4789062422286, 'frame': 18075},
        ('stimulus_window', 'enter'): {'rebased_time': 306.478977629464,
        'frame': 18075},
        ('stimulus_changed', ''): {'rebased_time': 310.9827406729944, 'frame': 18345},
        ('auto_reward', ''): {'rebased_time': 310.98279450599154, 'frame': 18345},
        ('response_window', 'enter'): {'rebased_time': 311.13223900212347,
        'frame': 18354},
        ('response_window', 'exit'): {'rebased_time': 311.73284526699706,
        'frame': 18390},
        ('miss', ''): {'rebased_time': 311.7330193465259, 'frame': 18390},
        ('stimulus_window', 'exit'): {'rebased_time': 315.2356723770604,
        'frame': 18600},
        ('no_lick', 'exit'): {'rebased_time': 315.23582480636213, 'frame': 18600},
        ('trial_end', ''): {'rebased_time': 315.23590438557534, 'frame': 18600}
    }

    licks = [
        312.24876,
        312.58027,
        312.73126,
        312.86627,
        313.02635,
        313.16292,
        313.54016,
        314.04408,
        314.47449,
        314.61011,
        314.75495,
    ]

    nan = np.nan
    stimulus_presentations_df = pd.DataFrame({
        'duration': {
            2: 0.2499599999999873,
            3: nan,
            4: 0.24966999999998052,
            5: 0.24990999999999985,
            6: 0.2499800000000505,
            7: 0.2499300000000062,
            8: 0.24996999999996206,
            9: 0.24968999999998687,
            10: 0.25983999999999696,
            11: nan,
            12: 0.24995000000001255,
            13: 0.2500800000000254,
            14: 0.2500800000000254
        },
        'end_frame': {
            2: 18091.0,
            3: nan,
            4: 18181.0,
            5: 18226.0,
            6: 18271.0,
            7: 18316.0,
            8: 18361.0,
            9: 18406.0,
            10: 18451.0,
            11: nan,
            12: 18541.0,
            13: 18586.0,
            14: 18631.0
        },
        'image_index': {
            2: 0,
            3: 8,
            4: 0,
            5: 0,
            6: 0,
            7: 0,
            8: 5,
            9: 5,
            10: 5,
            11: 8,
            12: 5,
            13: 5,
            14: 5
        },
        'image_name': {
            2: 'im065',
            3: 'omitted',
            4: 'im065',
            5: 'im065',
            6: 'im065',
            7: 'im065',
            8: 'im062',
            9: 'im062',
            10: 'im062',
            11: 'omitted',
            12: 'im062',
            13: 'im062',
            14: 'im062'
        },
        'image_set': {
            2: 'Natural_Images_Lum_Matched_set_training_2017.07.14_2',
            3: 'omitted',
            4: 'Natural_Images_Lum_Matched_set_training_2017.07.14_2',
            5: 'Natural_Images_Lum_Matched_set_training_2017.07.14_2',
            6: 'Natural_Images_Lum_Matched_set_training_2017.07.14_2',
            7: 'Natural_Images_Lum_Matched_set_training_2017.07.14_2',
            8: 'Natural_Images_Lum_Matched_set_training_2017.07.14_2',
            9: 'Natural_Images_Lum_Matched_set_training_2017.07.14_2',
            10: 'Natural_Images_Lum_Matched_set_training_2017.07.14_2',
            11: 'omitted',
            12: 'Natural_Images_Lum_Matched_set_training_2017.07.14_2',
            13: 'Natural_Images_Lum_Matched_set_training_2017.07.14_2',
            14: 'Natural_Images_Lum_Matched_set_training_2017.07.14_2'
        },
        'index': {
            2: 2,
            3: 0,
            4: 3,
            5: 4,
            6: 5,
            7: 6,
            8: 7,
            9: 8,
            10: 9,
            11: 1,
            12: 10,
            13: 11,
            14: 12
        },
        'omitted': {
            2: False,
            3: True,
            4: False,
            5: False,
            6: False,
            7: False,
            8: False,
            9: False,
            10: False,
            11: True,
            12: False,
            13: False,
            14: False
        },
        'orientation': {
            2: nan,
            3: nan,
            4: nan,
            5: nan,
            6: nan,
            7: nan,
            8: nan,
            9: nan,
            10: nan,
            11: nan,
            12: nan,
            13: nan,
            14: nan
        },
        'start_frame': {
            2: 18076,
            3: 18120,
            4: 18166,
            5: 18211,
            6: 18256,
            7: 18301,
            8: 18346,
            9: 18391,
            10: 18436,
            11: 18480,
            12: 18526,
            13: 18571,
            14: 18616
        },
        'start_time': {
            2: 307.2712,
            3: 308.00111,
            4: 308.77121,
            5: 309.52094,
            6: 310.27088,
            7: 311.02089,
            8: 311.77086,
            9: 312.52112,
            10: 313.27098,
            11: 314.01115,
            12: 314.78083,
            13: 315.53094,
            14: 316.2808
        },
        'stop_time': {
            2: 307.52116,
            3: nan,
            4: 309.02088,
            5: 309.77085,
            6: 310.52086,
            7: 311.27082,
            8: 312.02083,
            9: 312.77081,
            10: 313.53082,
            11: nan,
            12: 315.03078,
            13: 315.78102,
            14: 316.53088
        }
    })

    result = trials_processing.get_trial_timing(
        event_dict,
        stimulus_presentations_df,
        licks,
        go=False,
        catch=False,
        auto_rewarded=True,
        hit=False,
        false_alarm=False,
    )

    expected_result = {
        'start_time': 306.4785879253758,
        'stop_time': 315.23590438557534,
        'trial_length': 8.757316460199547,
        'response_time': nan,
        'change_frame': 18345,
        'change_time': 311.77086,
        'response_latency': 0.4778999999999769
        }

    # use assert_frame_equal to take advantage of the nice way it deals with NaNs
    pd.testing.assert_frame_equal(pd.DataFrame(result,index=[0]), pd.DataFrame(expected_result,index=[0]), check_names=False)
