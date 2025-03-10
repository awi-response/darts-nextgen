"""Copernicus STAC utilities."""

import logging

logger = logging.getLogger(__name__)


def init_copernicus(profile_name: str = "default"):
    """Configure odc.stac and rio to authenticate with Copernicus cloud.

    This functions expects that credentials are present in the .aws/credentials file.
    Credentials can be optained from https://eodata-s3keysmanager.dataspace.copernicus.eu/

    Example credentials file:

    ```
    [default]
    AWS_ACCESS_KEY_ID=...
    AWS_SECRET_ACCESS_KEY=...
    ```

    Args:
        profile_name (str, optional): The boto3 profile name. Defaults to "default".

    References:
        - S3 access: https://documentation.dataspace.copernicus.eu/APIs/S3.html

    """
    import boto3
    import odc.stac

    session = boto3.Session(profile_name=profile_name)
    credentials = session.get_credentials()

    odc.stac.configure_rio(
        cloud_defaults=True,
        verbose=True,
        aws={
            "profile_name": profile_name,
            "aws_access_key_id": credentials.access_key,
            "aws_secret_access_key": credentials.secret_key,
            "region_name": "default",
            "endpoint_url": "eodata.ams.dataspace.copernicus.eu",
        },
        AWS_VIRTUAL_HOSTING=False,
    )
    logger.debug("Copernicus STAC initialized")
