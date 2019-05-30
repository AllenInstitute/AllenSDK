import os
import pathlib

import argparse
from marshmallow import RAISE, ValidationError
from argschema import ArgSchemaParser
from argschema.schemas import DefaultSchema


def write_or_print_outputs(data, parser):
    data.update({'input_parameters': parser.args})
    if 'output_json' in parser.args:
        parser.output(data, indent=2)
    else:
        print(parser.get_output_json(data))  


def check_write_access_dir(dirpath):

    if os.path.exists(dirpath):
        test_filepath = pathlib.Path(dirpath, 'test_file.txt')
        try:
            with test_filepath.open() as _:
                pass
            os.remove(test_filepath)
            return True
        except PermissionError:
            raise ValidationError(f'don\'t have permissions to write in directory {dirpath}')
    else:
        try:
            pathlib.Path(dirpath).mkdir(parents=True)
            pathlib.Path(dirpath).rmdir()
            return True
        except PermissionError:
            raise ValidationError(f'Can\'t build path to requested location {dirpath}')

    raise RuntimeError('Unhandled case; this should not happen')


def check_write_access(filepath):
    try:
        fd = os.open(filepath, os.O_CREAT | os.O_EXCL)
        os.close(fd)
        os.remove(filepath)
        return True
    except FileExistsError:
        raise ValidationError(f'file at {filepath} already exists')
    except (FileNotFoundError, PermissionError):
        base_dir = os.path.dirname(filepath)
        return check_write_access_dir(base_dir)
    except Exception as e:
        raise e

    raise RuntimeError('Unhandled case; this should not happen')


def check_read_access(path):
    try:
        f = open(path, mode='r')
        f.close()
        return True
    except Exception as err:
        raise ValidationError(f'file at #{path} not readable (#{type(err)}: {err}')


class RaisingSchema(DefaultSchema):
    class Meta:
        unknown=RAISE


class ArgSchemaParserPlus(ArgSchemaParser):  # pragma: no cover

    def __init__(self, *args, **kwargs):
        parser = argparse.ArgumentParser()
        [known_args, extra_args] = parser.parse_known_args()
        self.args = known_args

        super(ArgSchemaParserPlus, self).__init__(args=extra_args, **kwargs)
