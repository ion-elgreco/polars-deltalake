from __future__ import annotations

import json
import socket
import urllib.request
from dataclasses import dataclass

import boto3
import polars as pl
import pytest
from azure.storage import blob as azure_blob
from deltalake import write_deltalake
from polars.testing import assert_frame_equal
from testcontainers.core.container import DockerContainer
from testcontainers.core.wait_strategies import LogMessageWaitStrategy

from _cloud_helpers import docker_available
from polars_deltalake import read_delta, scan_delta

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not docker_available(), reason="docker daemon unreachable"),
]


@dataclass
class Backend:
    name: str
    uri_root: str  # e.g. "s3://bucket"
    storage_options: dict[str, str]


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


_S3_IMAGE = "minio/minio:RELEASE.2024-12-18T13-15-44Z"
_S3_USER = "test-user"
_S3_PASS = "test-password-123"


@pytest.fixture(scope="module")
def _s3():
    container = (
        DockerContainer(_S3_IMAGE)
        .with_env("MINIO_ROOT_USER", _S3_USER)
        .with_env("MINIO_ROOT_PASSWORD", _S3_PASS)
        .with_command("server /data --address :9000")
        .with_exposed_ports(9000)
        .waiting_for(LogMessageWaitStrategy("API:").with_startup_timeout(30))
    )
    with container:
        host = container.get_container_host_ip()
        port = int(container.get_exposed_port(9000))
        endpoint = f"http://{host}:{port}"
        bucket = "polars-deltalake-test"
        boto3.client(
            "s3",
            endpoint_url=endpoint,
            aws_access_key_id=_S3_USER,
            aws_secret_access_key=_S3_PASS,
            region_name="us-east-1",
        ).create_bucket(Bucket=bucket)
        yield Backend(
            name="s3",
            uri_root=f"s3://{bucket}",
            storage_options={
                "aws_endpoint_url": endpoint,
                "aws_region": "us-east-1",
                "aws_access_key_id": _S3_USER,
                "aws_secret_access_key": _S3_PASS,
                "aws_allow_http": "true",
            },
        )


_AZURITE_IMAGE = "mcr.microsoft.com/azure-storage/azurite:3.33.0"
_AZURITE_ACCOUNT = "devstoreaccount1"
# Well-known Azurite key — public; not a secret.
_AZURITE_KEY = (
    "Eby8vdM02xNOcqFlqUwJPLlmEtlCDXJ1OUzFT50uSRZ6IFsuFq2UVErCz4I6tq/"
    "K1SZFPTOtr/KBHBeksoGMGw=="
)


@pytest.fixture(scope="module")
def _azure():
    container = (
        DockerContainer(_AZURITE_IMAGE)
        .with_command("azurite-blob --blobHost 0.0.0.0 --skipApiVersionCheck")
        .with_exposed_ports(10000)
        .waiting_for(
            LogMessageWaitStrategy(
                "Azurite Blob service successfully listens"
            ).with_startup_timeout(30)
        )
    )
    with container:
        host = container.get_container_host_ip()
        port = int(container.get_exposed_port(10000))
        endpoint = f"http://{host}:{port}/{_AZURITE_ACCOUNT}"
        container_name = "polars-deltalake-test"
        conn = (
            "DefaultEndpointsProtocol=http;"
            f"AccountName={_AZURITE_ACCOUNT};"
            f"AccountKey={_AZURITE_KEY};"
            f"BlobEndpoint={endpoint};"
        )
        azure_blob.BlobServiceClient.from_connection_string(conn).create_container(
            container_name
        )
        yield Backend(
            name="azure",
            uri_root=f"az://{container_name}",
            storage_options={
                "azure_storage_account_name": _AZURITE_ACCOUNT,
                "azure_storage_account_key": _AZURITE_KEY,
                "azure_storage_endpoint": endpoint,
                "azure_allow_http": "true",
            },
        )


# Patched fork by an `object_store` maintainer that backports the XML-API
# endpoints from fsouza/fake-gcs-server#1164 — same image delta-rs uses.
_GCS_IMAGE = "tustvold/fake-gcs-server"


@pytest.fixture(scope="module")
def _gcs():
    # The fork's XML-list handler only fires when `Host:` matches
    # `-public-host`; testcontainers' default dynamic-port mapping breaks
    # that, so we pick a port ourselves and bind it explicitly.
    bucket = "polars-deltalake-test"
    host_port = _free_port()
    public_host = f"localhost:{host_port}"
    container = (
        DockerContainer(_GCS_IMAGE)
        .with_command(
            f"-scheme http -port 4443 -public-host {public_host} -backend memory"
        )
        .with_bind_ports(4443, host_port)
        .waiting_for(
            LogMessageWaitStrategy("server started at").with_startup_timeout(30)
        )
    )
    with container:
        endpoint = f"http://{public_host}"
        req = urllib.request.Request(
            f"{endpoint}/storage/v1/b?project=test",
            data=json.dumps({"name": bucket}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req) as r:
            assert r.status in (200, 201)
        yield Backend(
            name="gcs",
            uri_root=f"gs://{bucket}",
            # `gcs_base_url` + `disable_oauth` inside the SA-JSON makes
            # object_store route to the emulator and skip the GCE-metadata
            # token dance (same trick delta-rs uses).
            storage_options={
                "google_service_account_key": json.dumps(
                    {
                        "gcs_base_url": endpoint,
                        "disable_oauth": True,
                        "client_email": "",
                        "private_key_id": "",
                        "private_key": "",
                    }
                ),
            },
        )


@pytest.fixture(scope="module", params=["s3", "azure", "gcs"])
def backend(request) -> Backend:
    return request.getfixturevalue(f"_{request.param}")


def test_write_then_read(backend: Backend):
    uri = f"{backend.uri_root}/simple/"
    df = pl.DataFrame({"id": [1, 2, 3], "name": ["a", "b", "c"]})
    write_deltalake(uri, df.to_arrow(), storage_options=backend.storage_options)

    out = read_delta(uri, storage_options=backend.storage_options).sort("id")
    assert_frame_equal(out, df)


def test_append(backend: Backend):
    uri = f"{backend.uri_root}/append/"
    write_deltalake(
        uri,
        pl.DataFrame({"x": [1, 2]}).to_arrow(),
        storage_options=backend.storage_options,
    )
    write_deltalake(
        uri,
        pl.DataFrame({"x": [3, 4]}).to_arrow(),
        mode="append",
        storage_options=backend.storage_options,
    )

    out = read_delta(uri, storage_options=backend.storage_options).sort("x")
    assert_frame_equal(out, pl.DataFrame({"x": [1, 2, 3, 4]}))


def test_partitioned(backend: Backend):
    uri = f"{backend.uri_root}/partitioned/"
    df = pl.DataFrame(
        {
            "region": ["eu", "us", "eu", "us"],
            "id": [1, 2, 3, 4],
            "value": [10, 20, 30, 40],
        }
    )
    write_deltalake(
        uri,
        df.to_arrow(),
        partition_by=["region"],
        storage_options=backend.storage_options,
    )

    out = scan_delta(uri, storage_options=backend.storage_options).collect().sort("id")
    assert_frame_equal(out, df)


def test_predicate_pushdown(backend: Backend):
    uri = f"{backend.uri_root}/pushdown/"
    write_deltalake(
        uri,
        pl.DataFrame({"g": ["a", "b", "c"], "v": [1, 2, 3]}).to_arrow(),
        storage_options=backend.storage_options,
    )

    out = (
        scan_delta(uri, storage_options=backend.storage_options)
        .filter(pl.col("g") == "b")
        .collect()
    )
    assert_frame_equal(out, pl.DataFrame({"g": ["b"], "v": [2]}))


def test_time_travel(backend: Backend):
    uri = f"{backend.uri_root}/tt/"
    write_deltalake(
        uri,
        pl.DataFrame({"x": [1]}).to_arrow(),
        storage_options=backend.storage_options,
    )
    write_deltalake(
        uri,
        pl.DataFrame({"x": [2]}).to_arrow(),
        mode="append",
        storage_options=backend.storage_options,
    )

    v0 = read_delta(uri, version=0, storage_options=backend.storage_options)
    assert_frame_equal(v0, pl.DataFrame({"x": [1]}))

    latest = read_delta(uri, storage_options=backend.storage_options).sort("x")
    assert_frame_equal(latest, pl.DataFrame({"x": [1, 2]}))
