# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging
import os
from pathlib import Path

import pytest
from omegaconf import DictConfig
from torch.cuda import memory_stats
from torch.cuda import reset_peak_memory_stats

from anemoi.training.schemas.base_schema import BaseSchema
from anemoi.training.train.profiler import AnemoiProfiler
from anemoi.training.train.train import AnemoiTrainer
from anemoi.utils.testing import skip_if_offline

import json
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError
import subprocess

os.environ["ANEMOI_BASE_SEED"] = "42"  # need to set base seed if running on github runners


LOGGER = logging.getLogger(__name__)

# return_val = value for speed profiler or 'avg_time' for time_profiler
def open_log_file(filename):
    import csv
    import glob
    import os

    if filename == "time_profiler.csv":
        return_val = "avg_time"
        row_selector = "name"
        row_name = "run_training_batch"
    elif filename == "speed_profiler.csv":
        return_val = "value"
        row_selector = "metric"
        row_name = "training_avg_throughput"
    else:
        raise ValueError
    tmpdir = os.getenv("TMPDIR")
    user = os.getenv("USER")  # TODO should use a more portable and secure way
    file_path = next(
        iter(
            glob.glob(
                f"{tmpdir}/pytest-of-{user}/pytest-0/test_benchmark_training_cycle0profiler/[a-z0-9]*/{filename}",
            ),
        ),
    )
    with Path(file_path).open(newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if row.get(row_selector) == row_name:
                result = row.get(return_val)
                break
    return float(result)

def set_performance_metrics(metrics):
    """
    Send performance metrics to a remote server via ssh.

    Parameters:
    - metrics (dict): A dictionary with benchmark names as keys and their values (e.g. {"avThroughputIterPerS": 10.5})
    """

    for benchmark, value in metrics.items():
         upload_metric_via_scp(benchmark, value)

def upload_metric_via_scp(metric_name, value):
    local_file = f"/tmp/{metric_name}"
    with open(local_file, "w") as f:
        f.write(str(value))

    remote_path="/home/data/public/anemoi-integration-tests/training/benchmarks"
    host="data@anemoi.ecmwf.int"
    scp_cmd = [
        "scp",
        local_file,
        f"{host}:{remote_path}/{metric_name}"
    ]
    LOGGER.debug(f"Scp command: {scp_cmd}")

    try:
        subprocess.run(scp_cmd, check=True)
        LOGGER.debug(f"Uploaded {metric_name} to {host}")
    except subprocess.CalledProcessError as e:
        print(f"SCP failed: {e}")

# reads a remote server to get past performance metrics to compare against
def get_performance_metrics():

    # ls public/anemoi-integration-tests/training/benchmarks/
    #    avThroughputIterPerS  avTimePerBatchS  peakMemoryMB
    # https://object-store.os-api.cci1.ecmwf.int/ml-tests/test-data/samples/anemoi-integration-tests/aifs-ea-an-oper-0001-mars-o48-1979-19-6h-v6-testset.zarr.tgz

    results = {}
    base_url = "https://object-store.os-api.cci1.ecmwf.int/ml-tests/test-data/samples/anemoi-integration-tests/training/benchmarks"
    benchmarks = ["avThroughputIterPerS", "avTimePerBatchS", "peakMemoryMB"]
    for benchmark in benchmarks:
        url = f"{base_url}/{benchmark}"
        print(f"Fetching benchmark data from {url}...")
        data = urlopen(url)  # it's a file like object and works just like a file
        for line in data:  # files are iterable
            line = float(line.strip())
            results[benchmark] = line

    return results


# @skip_if_offline
# add flag to save snapshot
# add multi-gpu support
# read from database on the s3 bucket, of a csv
# complain if any of the values differ by more than 10%
# can skip multi-gpu
# add compute/nccl/memory breakdown from pytorch profiler
@pytest.mark.longtests
def test_benchmark_training_cycle(
    benchmark_config: tuple[DictConfig, str],
    get_test_archive: callable,
    update_data=True,
) -> None:
    cfg, urls = benchmark_config
    for url in urls:
        get_test_archive(url)

    results = get_performance_metrics()
    LOGGER.debug(f"Performance benchmarks from server:\n{results}")

    reset_peak_memory_stats()
    AnemoiProfiler(cfg).profile()

    # read memory and mlflow stats
    stats = memory_stats(device=0)
    peak_active_mem_mb = stats["active_bytes.all.peak"] / 1024 / 1024
    av_training_throughput = open_log_file("speed_profiler.csv")
    av_training_batch_time_s = open_log_file("time_profiler.csv")

    print(f"Peak memory: {peak_active_mem_mb:.3f}MB")
    print(f"Av. training batch time: {av_training_batch_time_s:.2f}s")
    print(f"Av. training throughput: {av_training_throughput:.2f}iter/s")

    # either update the data on the server, or compare it against existing results
    if update_data:
        metrics={"avThroughputIterPerS":av_training_throughput, "avTimePerBatchS": av_training_batch_time_s, "peakMemoryMB": peak_active_mem_mb}
        print(f"Updating metrics on server with {metrics}")
        set_performance_metrics(metrics)
    else:
        if peak_active_mem_mb > results["peakMemoryMB"]:
            raise ValueError(
                f"Peak memory usage {peak_active_mem_mb:.3f}MB is greater than current benchmark peak of {results['peakMemoryMB']:.3f}MB",
            )
        else:
            print(f"Peak memory usage of {peak_active_mem_mb:.3f}MB is equal to or less than current benchamrk peak of  {results['peakMemoryMB']:.3f}MB")

        throuhput_tolerance_percent = 5
        throughput_upper_bound = results["avThroughputIterPerS"] * (100 + throuhput_tolerance_percent) / 100
        if av_training_throughput < throughput_upper_bound:
            raise ValueError(
                f"Average throughput of {av_training_throughput:.2f}iter/s is less than current benchmark throughput of {av_training_throughput:.2f}iter/s",
            )
        else:
            print(f"Average throughput of {av_training_throughput:.2f}iter/s is higher than or equal to the current benchmark throughput of { throughput_upper_bound:.2f}iter/s")
        batch_time_tolerance_percent = 5
        batch_time_upper_bound = results["avTimePerBatchS"] * (100 + batch_time_tolerance_percent) / 100
        if av_training_batch_time_s > batch_time_upper_bound:
            raise ValueError(f"Average time per batch of {av_training_batch_time_s:.2f}s is higher than current benchmark time of {batch_time_upper_bound:.2f}s")
        else:
            print(f"Average time per batch of {av_training_batch_time_s:.2f}s is less than or equal to than current benchmark time of {batch_time_upper_bound:.2f}s")

