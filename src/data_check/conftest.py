import pytest
import pandas as pd
import wandb
import os


def pytest_addoption(parser):
    parser.addoption("--csv", action="store")
    parser.addoption("--ref", action="store")
    parser.addoption("--kl_threshold", action="store", type=float)
    parser.addoption("--min_price", action="store", type=float)
    parser.addoption("--max_price", action="store", type=float)


@pytest.fixture(scope='session')
def data(request):
    run = wandb.init(job_type="data_tests", resume=True)

    # Download csv artifact
    artifact = run.use_artifact(request.config.option.csv)
    artifact_dir = artifact.download()
    files = [f for f in os.listdir(artifact_dir) if f.endswith('.csv')]
    data_path = os.path.join(artifact_dir, files[0])

    df = pd.read_csv(data_path)
    return df


@pytest.fixture(scope='session')
def ref_data(request):
    run = wandb.init(job_type="data_tests", resume=True)

    # Download ref artifact
    artifact = run.use_artifact(request.config.option.ref)
    artifact_dir = artifact.download()
    files = [f for f in os.listdir(artifact_dir) if f.endswith('.csv')]
    ref_path = os.path.join(artifact_dir, files[0])

    df = pd.read_csv(ref_path)
    return df


@pytest.fixture(scope='session')
def kl_threshold(request):
    return request.config.option.kl_threshold


@pytest.fixture(scope='session')
def min_price(request):
    return request.config.option.min_price


@pytest.fixture(scope='session')
def max_price(request):
    return request.config.option.max_price
