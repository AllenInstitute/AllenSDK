import boto3
import json
import hashlib


def load_dataset(data_blobs: dict,
                 manifest_version: str,
                 bucket_name: str,
                 client: boto3.client) -> None:
    """
    Load a test dataset into moto's mocked S3

    Parameters
    ----------
    data_blobs: dict
        Maps filename to a dict
        'data': the bytes in the data file
        'file_id': the file_id of the data file

    manifest_version: str
        The version of the manifest (manifest will be
        uploaded to moto3 as manifest_{manifest_version}.json

    bucket_name: str

    client: boto3.client

    Returns
    -------
    None
        Uploads the provided data, generates the manifest,
        and uploads the manifest to moto3
    """

    project_name = 'project-x'

    for fname in data_blobs:
        client.put_object(Bucket=bucket_name,
                          Key=f'project-x/data/{fname}',
                          Body=data_blobs[fname]['data'])

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
    manifest['data_pipeline'] = 'placeholder'

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

    manifest_k = f'{project_name}/manifests/'
    manifest_k += f'{project_name}_manifest_v{manifest_version}.json'
    client.put_object(Bucket=bucket_name,
                      Key=manifest_k,
                      Body=bytes(json.dumps(manifest), 'utf-8'))

    return None


def create_bucket(test_bucket_name: str, datasets: dict) -> None:
    """
    Create a bucket and populate it with example datasets

    Parameters
    ----------
    test_bucket_name: str
        Name of the bucket

    datasets: dict
        Keyed on version names; values are dicts of individual
        data files to be loaded to the bucket
    """

    conn = boto3.resource('s3', region_name='us-east-1')
    conn.create_bucket(Bucket=test_bucket_name, ACL='public-read')

    # https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/s3.html#bucketversioning
    bucket_versioning = conn.BucketVersioning(test_bucket_name)
    bucket_versioning.enable()

    client = boto3.client('s3', region_name='us-east-1')

    # upload first dataset
    for v in datasets.keys():
        load_dataset(datasets[v],
                     v,
                     test_bucket_name,
                     client)

    return None
