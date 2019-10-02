import pytest
from mock import patch, mock_open, Mock, MagicMock
try:
    import __builtin__ as builtins
except:
    import builtins
from allensdk.model.biophysical.utils import Utils
from allensdk.model.biophys_sim.config import Config
from allensdk.internal.model.biophysical.run_simulate_lims \
    import RunSimulateLims
from allensdk.model.biophys_sim.neuron.hoc_utils import HocUtils

MANIFEST_JSON = '''
{
  "biophys": [
    {
      "model_type": "Biophysical - perisomatic", 
      "model_file": [
        "manifest_sdk.json", 
        "/projects/mousecelltypes/vol1/prod520/neuronal_model_488462965/487667205_fit.json"
      ]
    }
  ], 
  "runs": [
    {
      "sweeps_by_type": {
        "Noise 1": [
          39, 
          41, 
          43, 
          45
        ], 
        "Noise 2": [
          38, 
          40, 
          42, 
          44
        ], 
        "Ramp": [
          89, 
          90, 
          91
        ], 
        "Unknown": [
          5
        ], 
        "Short Square": [
          75, 
          76, 
          77, 
          78, 
          79, 
          80, 
          81, 
          82, 
          83, 
          84, 
          85, 
          86, 
          87, 
          88
        ], 
        "Ramp to Rheobase": [
          21, 
          22, 
          23
        ], 
        "Square - 2s Suprathreshold": [
          24, 
          25, 
          26, 
          27, 
          28, 
          29, 
          30, 
          31, 
          32, 
          33, 
          34, 
          35
        ], 
        "Long Square": [
          8, 
          9, 
          10, 
          11, 
          12, 
          13, 
          14, 
          15, 
          16, 
          17, 
          18, 
          19, 
          46, 
          47, 
          48, 
          49, 
          50, 
          51, 
          52, 
          53, 
          54, 
          55, 
          56, 
          57, 
          58, 
          59, 
          60, 
          61, 
          62, 
          63, 
          64, 
          65, 
          66, 
          67, 
          68, 
          69, 
          70, 
          71, 
          72, 
          73, 
          74
        ], 
        "Square - 0.5ms Subthreshold": [
          36, 
          37
        ], 
        "Test": [
          0, 
          1, 
          2, 
          3, 
          4, 
          6, 
          7, 
          20
        ]
      }, 
      "sweeps": [
        5, 
        6, 
        7, 
        8, 
        9, 
        10, 
        11, 
        12, 
        13, 
        14, 
        15, 
        16, 
        17, 
        18, 
        19, 
        20, 
        21, 
        22, 
        23, 
        24, 
        25, 
        26, 
        27, 
        28, 
        29, 
        30, 
        31, 
        32, 
        33, 
        34, 
        35, 
        36, 
        37, 
        38, 
        39, 
        40, 
        41, 
        42, 
        43, 
        44, 
        45, 
        46, 
        47, 
        48, 
        49, 
        50, 
        51, 
        52, 
        53, 
        54, 
        55, 
        56, 
        57, 
        58, 
        59, 
        60, 
        61, 
        62, 
        63, 
        64, 
        65, 
        66, 
        67, 
        68, 
        69, 
        70, 
        72, 
        73, 
        74, 
        75, 
        76, 
        77, 
        78, 
        79, 
        80, 
        81, 
        82, 
        83, 
        84, 
        85, 
        87, 
        91
      ], 
      "neuronal_model_run_id": 496537307
    }
  ], 
  "neuron": [
    {
      "hoc": [
        "stdgui.hoc", 
        "import3d.hoc", 
        "cell.hoc"
      ]
    }
  ], 
  "manifest": [
    {
      "type": "dir", 
      "spec": "/local1/tmp", 
      "key": "BASEDIR"
    }, 
    {
      "type": "dir", 
      "spec": "/local1/tmp", 
      "key": "WORKDIR"
    }, 
    {
      "type": "file", 
      "spec": "/projects/mousecelltypes/vol1/prod534/specimen_487667205/Pvalb-IRES-Cre_Ai14-212813.03.01.01_496163999_m.swc", 
      "key": "MORPHOLOGY"
    }, 
    {
      "type": "dir", 
      "spec": "templates", 
      "key": "CODE_DIR"
    }, 
    {
      "type": "dir", 
      "spec": "modfiles", 
      "key": "MODFILE_DIR"
    }, 
    {
      "type": "file", 
      "spec": "/projects/mousecelltypes/vol1/prod555/project_in vitro Single Cell Characterization_T301/Kv2like.mod", 
      "key": "MOD_FILE_Kv2like", 
      "format": "MODFILE"
    }, 
    {
      "type": "file", 
      "spec": "/projects/mousecelltypes/vol1/prod251/project_in vitro Single Cell Characterization_T301/NaV.mod", 
      "key": "MOD_FILE_NaV", 
      "format": "MODFILE"
    }, 
    {
      "type": "file", 
      "spec": "/projects/mousecelltypes/vol1/prod210/project_in vitro Single Cell Characterization_T301/Ca_HVA.mod", 
      "key": "MOD_FILE_Ca_HVA", 
      "format": "MODFILE"
    }, 
    {
      "type": "file", 
      "spec": "/projects/mousecelltypes/vol1/prod210/project_in vitro Single Cell Characterization_T301/Ca_LVA.mod", 
      "key": "MOD_FILE_Ca_LVA", 
      "format": "MODFILE"
    }, 
    {
      "type": "file", 
      "spec": "/projects/mousecelltypes/vol1/prod210/project_in vitro Single Cell Characterization_T301/CaDynamics.mod", 
      "key": "MOD_FILE_CaDynamics", 
      "format": "MODFILE"
    }, 
    {
      "type": "file", 
      "spec": "/projects/mousecelltypes/vol1/prod210/project_in vitro Single Cell Characterization_T301/Ih.mod", 
      "key": "MOD_FILE_Ih", 
      "format": "MODFILE"
    }, 
    {
      "type": "file", 
      "spec": "/projects/mousecelltypes/vol1/prod210/project_in vitro Single Cell Characterization_T301/Im.mod", 
      "key": "MOD_FILE_Im", 
      "format": "MODFILE"
    }, 
    {
      "type": "file", 
      "spec": "/projects/mousecelltypes/vol1/prod210/project_in vitro Single Cell Characterization_T301/Im_v2.mod", 
      "key": "MOD_FILE_Im_v2", 
      "format": "MODFILE"
    }, 
    {
      "type": "file", 
      "spec": "/projects/mousecelltypes/vol1/prod210/project_in vitro Single Cell Characterization_T301/K_P.mod", 
      "key": "MOD_FILE_K_P", 
      "format": "MODFILE"
    }, 
    {
      "type": "file", 
      "spec": "/projects/mousecelltypes/vol1/prod210/project_in vitro Single Cell Characterization_T301/K_T.mod", 
      "key": "MOD_FILE_K_T", 
      "format": "MODFILE"
    }, 
    {
      "type": "file", 
      "spec": "/projects/mousecelltypes/vol1/prod210/project_in vitro Single Cell Characterization_T301/Kd.mod", 
      "key": "MOD_FILE_Kd", 
      "format": "MODFILE"
    }, 
    {
      "type": "file", 
      "spec": "/projects/mousecelltypes/vol1/prod210/project_in vitro Single Cell Characterization_T301/Kv3_1.mod", 
      "key": "MOD_FILE_Kv3_1", 
      "format": "MODFILE"
    }, 
    {
      "type": "file", 
      "spec": "/projects/mousecelltypes/vol1/prod210/project_in vitro Single Cell Characterization_T301/Nap.mod", 
      "key": "MOD_FILE_Nap", 
      "format": "MODFILE"
    }, 
    {
      "type": "file", 
      "spec": "/projects/mousecelltypes/vol1/prod210/project_in vitro Single Cell Characterization_T301/NaTa.mod", 
      "key": "MOD_FILE_NaTa", 
      "format": "MODFILE"
    }, 
    {
      "type": "file", 
      "spec": "/projects/mousecelltypes/vol1/prod210/project_in vitro Single Cell Characterization_T301/NaTs.mod", 
      "key": "MOD_FILE_NaTs", 
      "format": "MODFILE"
    }, 
    {
      "type": "file", 
      "spec": "/projects/mousecelltypes/vol1/prod210/project_in vitro Single Cell Characterization_T301/SK.mod", 
      "key": "MOD_FILE_SK", 
      "format": "MODFILE"
    }, 
    {
      "type": "file", 
      "spec": "/projects/mousecelltypes/vol1/prod555/project_in vitro Single Cell Characterization_T301/Kv2like.mod", 
      "key": "MOD_FILE_Kv2like", 
      "format": "MODFILE"
    }, 
    {
      "type": "file", 
      "spec": "/projects/mousecelltypes/vol1/prod251/project_in vitro Single Cell Characterization_T301/NaV.mod", 
      "key": "MOD_FILE_NaV", 
      "format": "MODFILE"
    }, 
    {
      "type": "file", 
      "spec": "/projects/mousecelltypes/vol1/prod210/project_in vitro Single Cell Characterization_T301/Ca_HVA.mod", 
      "key": "MOD_FILE_Ca_HVA", 
      "format": "MODFILE"
    }, 
    {
      "type": "file", 
      "spec": "/projects/mousecelltypes/vol1/prod210/project_in vitro Single Cell Characterization_T301/Ca_LVA.mod", 
      "key": "MOD_FILE_Ca_LVA", 
      "format": "MODFILE"
    }, 
    {
      "type": "file", 
      "spec": "/projects/mousecelltypes/vol1/prod210/project_in vitro Single Cell Characterization_T301/CaDynamics.mod", 
      "key": "MOD_FILE_CaDynamics", 
      "format": "MODFILE"
    }, 
    {
      "type": "file", 
      "spec": "/projects/mousecelltypes/vol1/prod210/project_in vitro Single Cell Characterization_T301/Ih.mod", 
      "key": "MOD_FILE_Ih", 
      "format": "MODFILE"
    }, 
    {
      "type": "file", 
      "spec": "/projects/mousecelltypes/vol1/prod210/project_in vitro Single Cell Characterization_T301/Im.mod", 
      "key": "MOD_FILE_Im", 
      "format": "MODFILE"
    }, 
    {
      "type": "file", 
      "spec": "/projects/mousecelltypes/vol1/prod210/project_in vitro Single Cell Characterization_T301/Im_v2.mod", 
      "key": "MOD_FILE_Im_v2", 
      "format": "MODFILE"
    }, 
    {
      "type": "file", 
      "spec": "/projects/mousecelltypes/vol1/prod210/project_in vitro Single Cell Characterization_T301/K_P.mod", 
      "key": "MOD_FILE_K_P", 
      "format": "MODFILE"
    }, 
    {
      "type": "file", 
      "spec": "/projects/mousecelltypes/vol1/prod210/project_in vitro Single Cell Characterization_T301/K_T.mod", 
      "key": "MOD_FILE_K_T", 
      "format": "MODFILE"
    }, 
    {
      "type": "file", 
      "spec": "/projects/mousecelltypes/vol1/prod210/project_in vitro Single Cell Characterization_T301/Kd.mod", 
      "key": "MOD_FILE_Kd", 
      "format": "MODFILE"
    }, 
    {
      "type": "file", 
      "spec": "/projects/mousecelltypes/vol1/prod210/project_in vitro Single Cell Characterization_T301/Kv3_1.mod", 
      "key": "MOD_FILE_Kv3_1", 
      "format": "MODFILE"
    }, 
    {
      "type": "file", 
      "spec": "/projects/mousecelltypes/vol1/prod210/project_in vitro Single Cell Characterization_T301/Nap.mod", 
      "key": "MOD_FILE_Nap", 
      "format": "MODFILE"
    }, 
    {
      "type": "file", 
      "spec": "/projects/mousecelltypes/vol1/prod210/project_in vitro Single Cell Characterization_T301/NaTa.mod", 
      "key": "MOD_FILE_NaTa", 
      "format": "MODFILE"
    }, 
    {
      "type": "file", 
      "spec": "/projects/mousecelltypes/vol1/prod210/project_in vitro Single Cell Characterization_T301/NaTs.mod", 
      "key": "MOD_FILE_NaTs", 
      "format": "MODFILE"
    }, 
    {
      "type": "file", 
      "spec": "/projects/mousecelltypes/vol1/prod210/project_in vitro Single Cell Characterization_T301/SK.mod", 
      "key": "MOD_FILE_SK", 
      "format": "MODFILE"
    }, 
    {
      "type": "file", 
      "spec": "/projects/mousecelltypes/vol1/prod555/project_in vitro Single Cell Characterization_T301/Kv2like.mod", 
      "key": "MOD_FILE_Kv2like", 
      "format": "MODFILE"
    }, 
    {
      "type": "file", 
      "spec": "/projects/mousecelltypes/vol1/prod251/project_in vitro Single Cell Characterization_T301/NaV.mod", 
      "key": "MOD_FILE_NaV", 
      "format": "MODFILE"
    }, 
    {
      "type": "file", 
      "spec": "/projects/mousecelltypes/vol1/prod210/project_in vitro Single Cell Characterization_T301/Ca_HVA.mod", 
      "key": "MOD_FILE_Ca_HVA", 
      "format": "MODFILE"
    }, 
    {
      "type": "file", 
      "spec": "/projects/mousecelltypes/vol1/prod210/project_in vitro Single Cell Characterization_T301/Ca_LVA.mod", 
      "key": "MOD_FILE_Ca_LVA", 
      "format": "MODFILE"
    }, 
    {
      "type": "file", 
      "spec": "/projects/mousecelltypes/vol1/prod210/project_in vitro Single Cell Characterization_T301/CaDynamics.mod", 
      "key": "MOD_FILE_CaDynamics", 
      "format": "MODFILE"
    }, 
    {
      "type": "file", 
      "spec": "/projects/mousecelltypes/vol1/prod210/project_in vitro Single Cell Characterization_T301/Ih.mod", 
      "key": "MOD_FILE_Ih", 
      "format": "MODFILE"
    }, 
    {
      "type": "file", 
      "spec": "/projects/mousecelltypes/vol1/prod210/project_in vitro Single Cell Characterization_T301/Im.mod", 
      "key": "MOD_FILE_Im", 
      "format": "MODFILE"
    }, 
    {
      "type": "file", 
      "spec": "/projects/mousecelltypes/vol1/prod210/project_in vitro Single Cell Characterization_T301/Im_v2.mod", 
      "key": "MOD_FILE_Im_v2", 
      "format": "MODFILE"
    }, 
    {
      "type": "file", 
      "spec": "/projects/mousecelltypes/vol1/prod210/project_in vitro Single Cell Characterization_T301/K_P.mod", 
      "key": "MOD_FILE_K_P", 
      "format": "MODFILE"
    }, 
    {
      "type": "file", 
      "spec": "/projects/mousecelltypes/vol1/prod210/project_in vitro Single Cell Characterization_T301/K_T.mod", 
      "key": "MOD_FILE_K_T", 
      "format": "MODFILE"
    }, 
    {
      "type": "file", 
      "spec": "/projects/mousecelltypes/vol1/prod210/project_in vitro Single Cell Characterization_T301/Kd.mod", 
      "key": "MOD_FILE_Kd", 
      "format": "MODFILE"
    }, 
    {
      "type": "file", 
      "spec": "/projects/mousecelltypes/vol1/prod210/project_in vitro Single Cell Characterization_T301/Kv3_1.mod", 
      "key": "MOD_FILE_Kv3_1", 
      "format": "MODFILE"
    }, 
    {
      "type": "file", 
      "spec": "/projects/mousecelltypes/vol1/prod210/project_in vitro Single Cell Characterization_T301/Nap.mod", 
      "key": "MOD_FILE_Nap", 
      "format": "MODFILE"
    }, 
    {
      "type": "file", 
      "spec": "/projects/mousecelltypes/vol1/prod210/project_in vitro Single Cell Characterization_T301/NaTa.mod", 
      "key": "MOD_FILE_NaTa", 
      "format": "MODFILE"
    }, 
    {
      "type": "file", 
      "spec": "/projects/mousecelltypes/vol1/prod210/project_in vitro Single Cell Characterization_T301/NaTs.mod", 
      "key": "MOD_FILE_NaTs", 
      "format": "MODFILE"
    }, 
    {
      "type": "file", 
      "spec": "/projects/mousecelltypes/vol1/prod210/project_in vitro Single Cell Characterization_T301/SK.mod", 
      "key": "MOD_FILE_SK", 
      "format": "MODFILE"
    }, 
    {
      "type": "file", 
      "spec": "lims_message_simulate.json", 
      "key": "neuronal_model_run_data"
    }, 
    {
      "type": "file", 
      "spec": "/projects/mousecelltypes/vol1/prod514/Ephys_Roi_Result_487667203/487667203.nwb", 
      "key": "stimulus_path", 
      "format": "NWB"
    }, 
    {
      "type": "file", 
      "spec": "/local1/tmp/manifest_sdk.json", 
      "key": "manifest"
    }, 
    {
      "parent_key": "WORKDIR", 
      "type": "file", 
      "spec": "496537307_virtual_experiment.nwb", 
      "key": "output_path", 
      "format": "NWB"
    }, 
    {
      "type": "dir", 
      "spec": "/projects/mousecelltypes/vol1/prod520/neuronal_model_488462965/487667205_fit.json", 
      "key": "fit_parameters"
    }
  ],
  "passive": [
    {
      "ra": 29.0745151982, 
      "cm": [
        {
          "section": "soma", 
          "cm": 3.31732779736
        }, 
        {
          "section": "axon", 
          "cm": 3.31732779736
        }, 
        {
          "section": "dend", 
          "cm": 3.31732779736
        }
      ], 
      "e_pas": -85.65570068359375
    }
  ], 
  "fitting": [
    {
      "junction_potential": -14.0, 
      "sweeps": [
        56
      ]
    }
  ], 
  "conditions": [
    {
      "celsius": 34.0, 
      "erev": [
        {
          "ena": 53.0, 
          "section": "soma", 
          "ek": -107.0
        }
      ], 
      "v_init": -85.65570068359375
    }
  ], 
  "genome": [
    {
      "section": "soma", 
      "name": "gbar_Ih", 
      "value": 0.00090995210221476205, 
      "mechanism": "Ih"
    }, 
    {
      "section": "soma", 
      "name": "gbar_NaV", 
      "value": 0.081906946478702058, 
      "mechanism": "NaV"
    }, 
    {
      "section": "soma", 
      "name": "gbar_Kd", 
      "value": 2.4740872017758875e-08, 
      "mechanism": "Kd"
    }, 
    {
      "section": "soma", 
      "name": "gbar_Kv2like", 
      "value": 0.00160565599090932, 
      "mechanism": "Kv2like"
    }, 
    {
      "section": "soma", 
      "name": "gbar_Kv3_1", 
      "value": 2.296430043469444, 
      "mechanism": "Kv3_1"
    }, 
    {
      "section": "soma", 
      "name": "gbar_K_T", 
      "value": 0.059053715441415355, 
      "mechanism": "K_T"
    }, 
    {
      "section": "soma", 
      "name": "gbar_Im_v2", 
      "value": 3.9951061122506237e-14, 
      "mechanism": "Im_v2"
    }, 
    {
      "section": "soma", 
      "name": "gbar_SK", 
      "value": 6.8067829150919579e-10, 
      "mechanism": "SK"
    }, 
    {
      "section": "soma", 
      "name": "gbar_Ca_HVA", 
      "value": 0.0001014254260681964, 
      "mechanism": "Ca_HVA"
    }, 
    {
      "section": "soma", 
      "name": "gbar_Ca_LVA", 
      "value": 0.0097584217102168799, 
      "mechanism": "Ca_LVA"
    }, 
    {
      "section": "soma", 
      "name": "gamma_CaDynamics", 
      "value": 0.0007405691124034076, 
      "mechanism": "CaDynamics"
    }, 
    {
      "section": "soma", 
      "name": "decay_CaDynamics", 
      "value": 410.14174188957332, 
      "mechanism": "CaDynamics"
    }, 
    {
      "section": "soma", 
      "name": "g_pas", 
      "value": 5.2241645387452482e-05, 
      "mechanism": ""
    }, 
    {
      "section": "axon", 
      "name": "g_pas", 
      "value": 2.2719709304318236e-05, 
      "mechanism": ""
    }, 
    {
      "section": "dend", 
      "name": "g_pas", 
      "value": 1.0099597920543875e-07, 
      "mechanism": ""
    }
  ]
}
'''

@pytest.fixture
def run_simulate():
    rs = RunSimulateLims('manifest.json', 'out.json')
    
    return rs


def test_init(run_simulate):
    assert run_simulate.input_json == 'manifest.json'
    assert run_simulate.output_json == 'out.json'
    assert run_simulate.app_config == None
    assert run_simulate.manifest == None


@pytest.mark.xfail
@patch.object(Utils, "h")
@patch.object(HocUtils, "__init__")
def test_simulate(hoc_init, mock_h, run_simulate):
    # import allensdk.eclipse_debug
    
    mock_utils = Mock(name='mock_utils',
                      h=mock_h)

    with patch('allensdk.internal.api.queries.biophysical_module_reader.BiophysicalModuleReader',
               MagicMock(name="bio_mod_reader")) as bio_mod_reader:
        with patch('allensdk.model.biophysical.runner.save_nwb',
                   MagicMock(name="save_nwb")) as save_nwb:
            with patch('allensdk.model.biophysical.runner.NwbDataSet',
                       MagicMock(name='nwb_data_set')) as nwb_data_set:
                with patch('allensdk.model.biophysical.runner.copy',
                           MagicMock(name='shutil_copy')) as cp:
                    with patch('allensdk.model.biophysical.utils.create_utils',
                               return_value=mock_utils) as cu:
                        with patch(builtins.__name__ + ".open",
                                   mock_open(
                                       read_data=MANIFEST_JSON)):
                            fit_description = Config().load('manifest.json')
                            Utils.description = fit_description
                            run_simulate.simulate()
