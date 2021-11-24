import pytest
import copy


@pytest.fixture
def example_datasets():
    """
    A dict representing an example dataset that can
    be used for testing the CloudCache api.

    The key of the dict is the name of each file.
    The values of the dict are dicts in which
        'file_id' -> maps to the file_id used to describe the file
        'data' -> a bytestring representing the contents of the file
    """
    datasets = {}
    data = {}
    data['f1.txt'] = {'data': b'1234567',
                      'file_id': '1'}
    data['f2.txt'] = {'data': b'4567890',
                      'file_id': '2'}
    data['f3.txt'] = {'data': b'11121314',
                      'file_id': '3'}
    datasets['1.0.0'] = data

    data = {}
    data['f1.txt'] = {'data': b'abcdefg',
                      'file_id': '1'}
    data['f2.txt'] = {'data': b'4567890',
                      'file_id': '2'}
    data['f3.txt'] = {'data': b'11121314',
                      'file_id': '3'}

    datasets['2.0.0'] = data

    data = {}
    data['f1.txt'] = {'data': b'1234567',
                      'file_id': '1'}
    data['f2.txt'] = {'data': b'xyzabcde',
                      'file_id': '2'}
    data['f3.txt'] = {'data': b'hijklmnop',
                      'file_id': '3'}

    datasets['3.0.0'] = data
    return datasets


@pytest.fixture
def baseline_data_with_metadata():
    """
    Example dataset with example metadata for use in testing
    CloudCache API
    """
    data = {}
    data['f1.txt'] = {'file_id': '1', 'data': b'1234'}
    data['f2.txt'] = {'file_id': '2', 'data': b'2345'}
    data['f3.txt'] = {'file_id': '3', 'data': b'6789'}

    metadata = {}
    metadata['metadata_1.csv'] = b'abcdef'
    metadata['metadata_2.csv'] = b'ghijklm'
    metadata['metadata_3.csv'] = b'nopqrst'
    return {'data': data, 'metadata': metadata}


@pytest.fixture
def example_datasets_with_metadata(baseline_data_with_metadata):
    """
    Multiple versions of an example dataset that goes through
    all possible mutations (adding/deleting files; renaming files;
    changing existing files) for use in testing the CloudCache API
    """

    example = {}
    example['data'] = {}
    example['metadata'] = {}

    data = copy.deepcopy(baseline_data_with_metadata)
    example['data']['1.0.0'] = data['data']
    example['metadata']['1.0.0'] = data['metadata']

    # delete one data file
    data = copy.deepcopy(baseline_data_with_metadata)
    data['data'].pop('f2.txt')
    example['data']['2.0.0'] = data['data']
    example['metadata']['2.0.0'] = data['metadata']

    # rename one data file
    data = copy.deepcopy(baseline_data_with_metadata)
    old = data['data'].pop('f2.txt')
    data['data']['f4.txt'] = {'file_id': '4', 'data': old['data']}
    example['data']['3.0.0'] = data['data']
    example['metadata']['3.0.0'] = data['metadata']

    # change one data file
    data = copy.deepcopy(baseline_data_with_metadata)
    data['data']['f3.txt'] = {'file_id': '3', 'data': b'44556677'}
    example['data']['4.0.0'] = data['data']
    example['metadata']['4.0.0'] = data['metadata']

    # add a data file
    data = copy.deepcopy(baseline_data_with_metadata)
    data['data']['f4.txt'] = {'file_id': '4', 'data': b'44556677'}
    example['data']['5.0.0'] = data['data']
    example['metadata']['5.0.0'] = data['metadata']

    # delete a data file and change another
    data = copy.deepcopy(baseline_data_with_metadata)
    data['data'].pop('f2.txt')
    data['data']['f1.txt'] = {'file_id': '1', 'data': b'xxxxxx'}
    example['data']['6.0.0'] = data['data']
    example['metadata']['6.0.0'] = data['metadata']

    # delete a data file and rename another
    data = copy.deepcopy(baseline_data_with_metadata)
    data['data'].pop('f2.txt')
    old = data['data'].pop('f3.txt')
    data['data']['f5.txt'] = {'file_id': '5', 'data': old['data']}
    example['data']['7.0.0'] = data['data']
    example['metadata']['7.0.0'] = data['metadata']

    # delete a data file and add another
    data = copy.deepcopy(baseline_data_with_metadata)
    data['data'].pop('f2.txt')
    data['data']['f5.txt'] = {'file_id': '5', 'data': b'yyyyy'}
    example['data']['8.0.0'] = data['data']
    example['metadata']['8.0.0'] = data['metadata']

    # rename a data file and add another
    data = copy.deepcopy(baseline_data_with_metadata)
    old = data['data'].pop('f3.txt')
    data['data']['f4.txt'] = {'file_id': '4', 'data': old['data']}
    data['data']['f5.txt'] = {'file_id': '5', 'data': b'wwwwww'}
    example['data']['9.0.0'] = data['data']
    example['metadata']['9.0.0'] = data['metadata']

    # delete a metadata file
    data = copy.deepcopy(baseline_data_with_metadata)
    data['metadata'].pop('metadata_2.csv')
    example['data']['10.0.0'] = data['data']
    example['metadata']['10.0.0'] = data['metadata']

    # rename a metadata file
    data = copy.deepcopy(baseline_data_with_metadata)
    old = data['metadata'].pop('metadata_2.csv')
    data['metadata']['metadata_4.csv'] = old
    example['data']['11.0.0'] = data['data']
    example['metadata']['11.0.0'] = data['metadata']

    # change a metadata file
    data = copy.deepcopy(baseline_data_with_metadata)
    data['metadata']['metadata_3.csv'] = b'12345'
    example['data']['12.0.0'] = data['data']
    example['metadata']['12.0.0'] = data['metadata']

    # add a metadata file
    data = copy.deepcopy(baseline_data_with_metadata)
    data['metadata']['metadata_4.csv'] = b'12345'
    example['data']['13.0.0'] = data['data']
    example['metadata']['13.0.0'] = data['metadata']

    # delete a data file and change a metadata file
    data = copy.deepcopy(baseline_data_with_metadata)
    data['data'].pop('f2.txt')
    old = data['metadata'].pop('metadata_3.csv')
    data['metadata']['metadata_4.csv'] = old
    example['data']['14.0.0'] = data['data']
    example['metadata']['14.0.0'] = data['metadata']

    # rename a data file, add two data files
    # rename a metadata file and delete two metadata files
    data = copy.deepcopy(baseline_data_with_metadata)
    old = data['data'].pop('f1.txt')
    data['data']['f4.txt'] = old
    data['data']['f5.txt'] = {'file_id': '5', 'data': b'babababa'}
    data['data']['f6.txt'] = {'file_id': '6', 'data': b'neighneigh'}
    old = data['metadata'].pop('metadata_2.csv')
    data['metadata']['metadata_4.csv'] = old
    data['metadata'].pop('metadata_1.csv')
    data['metadata'].pop('metadata_3.csv')

    example['data']['15.0.0'] = data['data']
    example['metadata']['15.0.0'] = data['metadata']

    return example
