import os

from marshmallow import RAISE, ValidationError
from argschema.schemas import DefaultSchema


def write_or_print_outputs(data, parser):
    data.update({'input_parameters': parser.args})
    if 'output_json' in parser.args:
        parser.output(data, indent=2)
    else:
        print(parser.get_output_json(data))  


def check_write_access(path):
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL)
        os.close(fd)
        os.remove(path)
        return True
    except FileNotFoundError:
        check_write_access(os.path.dirname(path))
    except PermissionError:
        raise ValidationError(f'don\'t have permissions to write {path}')
    except FileExistsError:
        if not os.path.isdir(path):
            raise ValidationError(f'file at {path} already exists')
        return True


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