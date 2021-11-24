import subprocess as sp
import os

import pytest


EXAMPLE_DIR = os.path.join(
    os.path.dirname(__file__),
    '..',
    '..',
    'doc_template',
    'examples_root',
    'examples'
)
EXAMPLES = [filename for filename in os.listdir(EXAMPLE_DIR) if filename.split('.')[-1] == 'py']


@pytest.mark.nightly
@pytest.mark.parametrize('script_name', EXAMPLES)
def test_inline_examples(script_name, tmpdir_factory):

    data_dir = tmpdir_factory.mktemp('inline_examples_data')
    sp.check_call(['python', os.path.join(EXAMPLE_DIR, script_name)], cwd=str(data_dir))