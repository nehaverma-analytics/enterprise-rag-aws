"""Upload and download documents using the rag-app AWS profile."""

from functools import lru_cache
from pathlib import Path

import boto3
from botocore.client import BaseClient
from botocore.exceptions import ClientError


@lru_cache(maxsize=1)
def _get_s3_client() -> BaseClient:
    """Lazily create and reuse the S3 client for us-east-1."""
    session = boto3.Session(profile_name="rag-app")
    return session.client("s3", region_name="us-east-1")


def upload_document(local_path: str, bucket_name: str, object_key: str) -> str:
    """Upload a local file and return its S3 URI.

    Raises FileNotFoundError for missing files and IsADirectoryError for
    directories. AWS ClientError exceptions retain their original details.
    """
    try:
        with Path(local_path).open("rb") as document:
            client = _get_s3_client()
            client.upload_fileobj(document, bucket_name, object_key)
    except ClientError as error:
        error.add_note("S3 document upload failed in us-east-1.")
        raise

    return f"s3://{bucket_name}/{object_key}"


def download_document(bucket_name: str, object_key: str, local_path: str) -> str:
    """Download an S3 object and return the supplied local file path.

    The destination's parent directory must exist. AWS ClientError exceptions
    retain their original details.
    """
    try:
        client = _get_s3_client()
        client.download_file(bucket_name, object_key, local_path)
    except ClientError as error:
        error.add_note("S3 document download failed in us-east-1.")
        raise

    return local_path
