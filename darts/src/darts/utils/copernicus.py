"""Copernicus STAC utilities."""

import logging
import os

logger = logging.getLogger(__name__)


def init_copernicus(profile_name: str = "default"):
    """Configure odc.stac and rio to authenticate with Copernicus cloud.

    This functions expects that credentials are present in the .aws/credentials file.
    Credentials can be obtained from https://eodata-s3keysmanager.dataspace.copernicus.eu/

    Example credentials file:

    ```
    [default]
    AWS_ACCESS_KEY_ID=...
    AWS_SECRET_ACCESS_KEY=...
    ```

    Args:
        profile_name (str, optional): The boto3 profile name. This must match with the name in the credentials file!.
            Defaults to "default".

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


def init_copernicus_from_keys(access_key: str, secret_key: str):
    """Set up the environment for accessing the Copernicus Data Space Ecosystem S3 storage.

    This will configure the necessary environment variables for accessing the S3 storage
    and calls configure_rio to set up the rasterio environment.

    Keys can be obtained from the Copernicus S3 Credentials manager:
    https://eodata-s3keysmanager.dataspace.copernicus.eu/panel/s3-credentials

    Args:
        access_key (str): The AWS access key ID.
        secret_key (str): The AWS secret access key.

    References:
        - S3 access: https://documentation.dataspace.copernicus.eu/APIs/S3.html

    """
    import boto3
    import odc.stac

    os.environ["GDAL_HTTP_TCP_KEEPALIVE"] = "YES"
    os.environ["AWS_S3_ENDPOINT"] = "eodata.dataspace.copernicus.eu"
    os.environ["AWS_ACCESS_KEY_ID"] = access_key
    os.environ["AWS_SECRET_ACCESS_KEY"] = secret_key
    os.environ["AWS_HTTPS"] = "YES"
    os.environ["AWS_VIRTUAL_HOSTING"] = "FALSE"
    os.environ["GDAL_HTTP_UNSAFESSL"] = "YES"

    session = boto3.session.Session(
        aws_access_key_id=access_key, aws_secret_access_key=secret_key, region_name="default"
    )
    odc.stac.configure_rio(cloud_defaults=False, aws={"session": session, "aws_unsigned": False})
    logger.debug("Copernicus STAC initialized")
