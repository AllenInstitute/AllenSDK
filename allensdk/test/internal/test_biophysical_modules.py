from allensdk.internal.api.queries.biophysical_module_api \
    import BiophysicalModuleApi
import pytest
from mock import patch


@pytest.fixture
def biophysical_api():
    bma = BiophysicalModuleApi('http://axon:3000')

    return bma


def test_get_neuronal_model_runs(biophysical_api):
    neuronal_model_run_id = 464137111
    with patch.object(biophysical_api, "json_msg_query") as mock_query:
        biophysical_api.get_neuronal_model_runs(neuronal_model_run_id)
    expected = ("http://axon:3000/api/v2/data/query.json?q=model::Neuronal"
                "ModelRun,rma::criteria,[id$in464137111],rma::include,well_"
                "known_files(well_known_file_type),neuronal_model(well_known_"
                "files(well_known_file_type),specimen(project,specimen_tags,"
                "ephys_roi_result(ephys_qc_criteria,well_known_files(well_"
                "known_file_type)),neuron_reconstructions(well_known_files"
                "(well_known_file_type)),ephys_sweeps(ephys_sweep_tags,ephys_"
                "stimulus(ephys_stimulus_type))),neuronal_model_template"
                "(neuronal_model_template_type,well_known_files(well_known_"
                "file_type))),rma::options[num_rows$eq'all'][count$eqfalse]")
    mock_query.assert_called_once_with(expected)


def test_get_neuronal_models(biophysical_api):
    neuronal_model_id = 329322394
    with patch.object(biophysical_api, "json_msg_query") as mock_query:
        biophysical_api.get_neuronal_models(neuronal_model_id)
    expected = ("http://axon:3000/api/v2/data/query.json?q=model::Neuronal"
                "Model,rma::criteria,[id$in329322394],rma::include,"
                "well_known_files(well_known_file_type),specimen(project,"
                "specimen_tags,ephys_roi_result(ephys_qc_criteria,well_known_"
                "files(well_known_file_type)),neuron_reconstructions(well_"
                "known_files(well_known_file_type)),ephys_sweeps(ephys_sweep_"
                "tags,ephys_stimulus(ephys_stimulus_type))),neuronal_model_"
                "template(neuronal_model_template_type,well_known_files(well_"
                "known_file_type)),rma::options[num_rows$eq'all'][count$"
                "eqfalse]")
    mock_query.assert_called_once_with(expected)
