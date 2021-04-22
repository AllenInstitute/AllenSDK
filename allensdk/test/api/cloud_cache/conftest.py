import pytest


@pytest.fixture
def example_datasets():
    datasets = {}
    data = {}
    data['f1.txt'] = {'data': b'1234567',
                      'file_id': '1'}
    data['f2.txt'] = {'data': b'4567890',
                      'file_id': '2'}
    data['f3.txt'] = {'data': b'11121314',
                      'file_id': '3'}
    datasets['1.0'] = data

    data = {}
    data['f1.txt'] = {'data': b'abcdefg',
                      'file_id': '1'}
    data['f2.txt'] = {'data': b'4567890',
                      'file_id': '2'}
    data['f3.txt'] = {'data': b'11121314',
                      'file_id': '3'}

    datasets['2.0'] = data

    data = {}
    data['f1.txt'] = {'data': b'1234567',
                      'file_id': '1'}
    data['f2.txt'] = {'data': b'xyzabcde',
                      'file_id': '2'}
    data['f3.txt'] = {'data': b'hijklmnop',
                      'file_id': '3'}

    datasets['3.0'] = data
    return datasets
