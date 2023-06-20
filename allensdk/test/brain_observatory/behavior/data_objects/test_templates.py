import pytest

from allensdk.brain_observatory.behavior.data_objects.stimuli.templates import (  # noqa: E501
    Templates,
)


@pytest.mark.parametrize(
    "input_templates, message",
    ([{"images1": None, "Other Input": None},
      "Found multiple image StimulusTemplates"],
     [{"movie1": None, "movie2": None},
      "Found multiple fingerprint movie StimulusTemplate"],
     [{"images1": None, "Other Input": None,
       "movie1": None, "movie2": None},
      "Found multiple image StimulusTemplates"])
)
def test_template_to_many_inputs_exception(input_templates, message):
    """Test that we catch exceptions for too many input StimulusTemplates."""

    with pytest.raises(NotImplementedError, match=message):
        Templates(templates=input_templates)
