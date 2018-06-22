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


@pytest.mark.skipif(os.getenv('TEST_COMPLETE') != 'true', reason="partial testing")
def test_inline_examples():
    for filename in os.listdir(EXAMPLE_DIR):
        if filename.split('.')[-1] == 'py':
            sp.check_call(['python', os.path.join(EXAMPLE_DIR, filename)])