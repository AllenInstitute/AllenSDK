from typing import Union
import boto3
import json
import hashlib


def load_dataset(data_blobs: dict,
                 metadata_blobs: Union[dict, None],
                 manifest_version: str,
                 bucket_name: str,
                 project_name: str,
                 client: boto3.client) -> None:
    """
    Load a test dataset into moto's mocked S3

    Parameters
    ----------
    data_blobs: dict
        Maps filename to a dict
        'data': the bytes in the data file
        'file_id': the file_id of the data file

    metadata_blobs: Union[dict, None]
        A dict mapping metadata filename to bytes in the file

    manifest_version: str
        The version of the manifest (manifest will be
        uploaded to moto3 as manifest_{manifest_version}.json)

    bucket_name: str

    project_name: str

    client: boto3.client

    Returns
    -------
    None
        Uploads the provided data, generates the manifest,
        and uploads the manifest to moto3
    """

    for fname in data_blobs:
        client.put_object(Bucket=bucket_name,
                          Key=f'{project_name}/data/{fname}',
                          Body=data_blobs[fname]['data'])

    if metadata_blobs is not None:
        for fname in metadata_blobs:
            client.put_object(Bucket=bucket_name,
                              Key=f'{project_name}/project_metadata/{fname}',
                              Body=metadata_blobs[fname])

    response = client.list_object_versions(Bucket=bucket_name)
    fname_to_version = {}
    for obj in response['Versions']:
        if obj['IsLatest']:
            fname = obj['Key'].split('/')[-1]
            fname_to_version[fname] = obj['VersionId']

    manifest = {}
    manifest['manifest_version'] = manifest_version
    manifest['project_name'] = project_name
    manifest['metadata_file_id_column_name'] = 'file_id'
    manifest['metadata_files'] = {}
    manifest['data_pipeline'] = [{'name': 'AllenSDK', 'version': '1.1.1'}]

    data_file_dict = {}
    url_root = f'http://{bucket_name}.s3.amazonaws.com/{project_name}/data'
    for fname in data_blobs:
        url = f'{url_root}/{fname}'
        hasher = hashlib.blake2b()
        hasher.update(data_blobs[fname]['data'])
        checksum = hasher.hexdigest()

        data_file = {'url': url,
                     'version_id': fname_to_version[fname],
                     'file_hash': checksum}

        data_file_dict[data_blobs[fname]['file_id']] = data_file

    manifest['data_files'] = data_file_dict

    if metadata_blobs is not None:
        url_root = f'http://{bucket_name}.s3.amazonaws.com/{project_name}/'
        url_root += 'project_metadata'

        metadata_dict = {}
        for fname in metadata_blobs:
            url = f'{url_root}/{fname}'
            hasher = hashlib.blake2b()
            hasher.update(metadata_blobs[fname])
            metadata_dict[fname] = {'url': url,
                                    'file_hash': hasher.hexdigest(),
                                    'version_id': fname_to_version[fname]}

            manifest['metadata_files'] = metadata_dict

    manifest_k = f'{project_name}/manifests/'
    manifest_k += f'{project_name}_manifest_v{manifest_version}.json'
    client.put_object(Bucket=bucket_name,
                      Key=manifest_k,
                      Body=bytes(json.dumps(manifest), 'utf-8'))

    return None


def create_bucket(test_bucket_name: str,
                  project_name: str,
                  datasets: dict,
                  metadatasets: dict) -> None:
    """
    Create a bucket and populate it with example datasets

    Parameters
    ----------
    test_bucket_name: str
        Name of the bucket

    project_name: str
        Name of project

    datasets: dict
        Keyed on version names; values are dicts of individual
        data files to be loaded to the bucket

    metadatasets: dict
        Keyed on version names; values are dicts of individual
        metadata files to be loaded to the bucket (default: None)
    """

    conn = boto3.resource('s3', region_name='us-east-1')
    conn.create_bucket(Bucket=test_bucket_name, ACL='public-read')

    # https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/s3.html#bucketversioning
    bucket_versioning = conn.BucketVersioning(test_bucket_name)
    bucket_versioning.enable()

    client = boto3.client('s3', region_name='us-east-1')

    # upload first dataset
    for v in datasets.keys():
        if metadatasets is not None:
            m = metadatasets[v]
        else:
            m = None
        load_dataset(datasets[v],
                     m,
                     v,
                     test_bucket_name,
                     project_name,
                     client)

    return None
