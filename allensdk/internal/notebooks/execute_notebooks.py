import logging
import os
import shutil
import tempfile
from argparse import ArgumentParser
from pathlib import Path

import papermill
from nbconvert import NotebookExporter
from nbconvert.preprocessors import TagRemovePreprocessor
from papermill import PapermillExecutionError
from traitlets.config import Config

parser = ArgumentParser()
parser.add_argument(
    '--notebooks_dir',
    required=True,
    help='Path to notebooks to execute'
)
args = parser.parse_args()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(name='Notebook runner')


class NotebookRunner:
    """Notebook runner"""
    def __init__(self, notebooks_dir: str):
        """

        Parameters
        ----------
        notebooks_dir
            Path to notebooks
        """
        self._notebook_paths = [
            Path(args.notebooks_dir) / x for x in os.listdir(notebooks_dir)
            if Path(x).suffix == '.ipynb']

    def run(self):
        """Runs each notebook, overwriting it with updated output,
        if it succeeds. Logs any errors

        Raises
        ------
        RuntimeError
            If there are any errors
        """
        errors = []

        for notebook_path in self._notebook_paths:
            with tempfile.TemporaryDirectory() as tmp_dir:
                tmp_nb_path = Path(tmp_dir) / 'scratch_nb.ipynb'
                try:
                    papermill.execute_notebook(
                        input_path=notebook_path,
                        output_path=tmp_nb_path,
                        # Note: notebook must have a variable with this name
                        # and the cell must have tag 'parameters'
                        parameters={
                            'output_dir': tmp_dir,
                            'resources_dir': str(Path(__file__).parent /
                                                 'resources')
                        },
                        kernel_name='python3'
                    )
                    self._remove_injected_parameters_cell(
                        notebook_path=tmp_nb_path)

                    logging.info('Executing notebook succeeded. '
                                 f'Overwriting with new notebook output. '
                                 f'Moving {tmp_nb_path} to {notebook_path}')
                    shutil.move(tmp_nb_path, notebook_path)
                except PapermillExecutionError as e:
                    logging.error(e)
                    errors.append(notebook_path.name)
        if len(errors) > 0:
            msg = f'{len(errors)} notebooks failed. Errors in: {errors}'
            logging.error(msg)

            # TODO uncomment once notebooks run successfully
            #  https://github.com/AllenInstitute/AllenSDK/issues/2611
            # raise RuntimeError(msg)
        return errors

    @staticmethod
    def _remove_injected_parameters_cell(notebook_path):
        """Removes cells with tag "injected-parameters" and outputs notebook"""
        c = Config()
        c.TagRemovePreprocessor.remove_cell_tags = ("injected-parameters",)
        c.NotebookExporter.preprocessors = [
            "nbconvert.preprocessors.TagRemovePreprocessor"]
        exporter = NotebookExporter(config=c)
        exporter.register_preprocessor(TagRemovePreprocessor(config=c),
                                       enabled=True)
        output = NotebookExporter(config=c).from_filename(
            notebook_path)
        with open(notebook_path, "w") as f:
            f.write(output[0])


def main():
    notebook_runner = NotebookRunner(notebooks_dir=args.notebooks_dir)
    notebook_runner.run()


if __name__ == '__main__':
    main()
