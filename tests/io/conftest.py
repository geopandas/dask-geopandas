import os
from contextlib import contextmanager
import subprocess
import shlex
import logging
import time
import sys

import pytest


@contextmanager
def ensure_safe_environment_variables():
    """
    Get a context manager to safely set environment variables

    All changes will be undone on close, hence environment variables set
    within this contextmanager will neither persist nor change global state.
    """
    saved_environ = dict(os.environ)
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(saved_environ)


@pytest.fixture(scope="session")
def s3_server():
    """
    Fixture for mocking S3 interaction.

    Sets up moto server in separate process
    """
    pytest.importorskip("s3fs")
    pytest.importorskip("boto3")
    pytest.importorskip("moto", minversion="1.3.14")
    pytest.importorskip("flask")  # server mode needs flask too
    requests = pytest.importorskip("requests")
    logging.getLogger("requests").disabled = True

    endpoint_url = "http://127.0.0.1:5555/"

    with ensure_safe_environment_variables():
        os.environ["AWS_ACCESS_KEY_ID"] = "testing"
        os.environ["AWS_SECRET_ACCESS_KEY"] = "testing"
        os.environ["AWS_SECURITY_TOKEN"] = "testing"
        os.environ["AWS_SESSION_TOKEN"] = "testing"

        # Launching moto in server mode, i.e., as a separate process
        # with an S3 endpoint on localhost

        # pipe to null to avoid logging in terminal
        proc = subprocess.Popen(
            shlex.split("moto_server s3 -p 5555"),
            stdout=subprocess.DEVNULL,
        )

        timeout = 5
        while True:
            try:
                # OK to go once server is accepting connections
                r = requests.get(endpoint_url)
                if r.ok:
                    break
            except Exception:
                pass
            timeout -= 0.1
            time.sleep(0.1)
            assert timeout > 0, "Timed out waiting for moto server"
        yield endpoint_url

        # shut down external process
        proc.terminate()
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            proc.kill()
            if sys.platform == "win32":
                # belt & braces
                subprocess.call("TASKKILL /F /PID {pid} /T".format(pid=proc.pid))


@pytest.fixture
def s3_storage_options():
    return {"client_kwargs": {"endpoint_url": "http://127.0.0.1:5555/"}}


@pytest.fixture()
def s3_resource(s3_server):
    """
    Sets up S3 bucket 'geopandas-test'.
    """
    endpoint_url = s3_server

    import boto3
    import s3fs

    bucket = "geopandas-test"
    client = boto3.client("s3", endpoint_url=endpoint_url)

    client.create_bucket(Bucket=bucket, ACL="public-read-write")

    fs = s3fs.S3FileSystem(anon=True, client_kwargs={"endpoint_url": endpoint_url})
    s3fs.S3FileSystem.clear_instance_cache()
    fs.invalidate_cache()

    try:
        yield fs, endpoint_url
    finally:
        fs.rm(bucket, recursive=True)
