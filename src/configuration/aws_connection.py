import os
from pathlib import Path

import boto3
from dotenv import load_dotenv

from src.constants import AWS_ACCESS_KEY_ID_ENV_KEY, AWS_SECRET_ACCESS_KEY_ENV_KEY, REGION_NAME

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")


class S3Client:

    s3_client=None
    s3_resource = None
    def __init__(self, region_name=None):
        """ 
        This Class gets aws credentials from env_variable and creates an connection with s3 bucket 
        and raise exception when environment variable is not set
        """

        region_name = region_name or os.getenv("AWS_DEFAULT_REGION", REGION_NAME)

        if S3Client.s3_resource is None or S3Client.s3_client is None:
            __access_key_id = os.getenv(AWS_ACCESS_KEY_ID_ENV_KEY, "").strip()
            __secret_access_key = os.getenv(AWS_SECRET_ACCESS_KEY_ENV_KEY, "").strip()

            if __access_key_id and __secret_access_key:
                S3Client.s3_resource = boto3.resource(
                    "s3",
                    aws_access_key_id=__access_key_id,
                    aws_secret_access_key=__secret_access_key,
                    region_name=region_name,
                )
                S3Client.s3_client = boto3.client(
                    "s3",
                    aws_access_key_id=__access_key_id,
                    aws_secret_access_key=__secret_access_key,
                    region_name=region_name,
                )
            else:
                S3Client.s3_resource = boto3.resource("s3", region_name=region_name)
                S3Client.s3_client = boto3.client("s3", region_name=region_name)
        self.s3_resource = S3Client.s3_resource
        self.s3_client = S3Client.s3_client