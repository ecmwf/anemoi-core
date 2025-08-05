# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging
import operator
import os
import os.path
import io
import subprocess
from datetime import date
from pathlib import Path
from urllib.error import URLError

import pandas as pd
import pytest
from git import GitCommandError
from git import InvalidGitRepositoryError
from git import Repo
from omegaconf import DictConfig
from torch.cuda import memory_stats
from torch.cuda import reset_peak_memory_stats

from anemoi.training.train.profiler import AnemoiProfiler

os.environ["ANEMOI_BASE_SEED"] = "42"  # need to set base seed if running on github runners

LOGGER = logging.getLogger(__name__)


class BenchmarkValue:
    def __init__(
        self,
        name: str,
        value: float,
        unit: str,
        date: date,
        commit: str,
        op=operator.le,
        tolerance: int = 0,  # percentage
    ):
        self.name = name
        self.value = value
        self.unit = unit
        self.date = date
        self.commit = commit
        self.op = op
        self.tolerance = tolerance

    def __str__(self):
        return f"{self.name}: {self.value:.2f}{self.unit} (commit: {self.commit[:5]}, date: {self.date})"

    # header="testName,unit,date,commit,value"
    def to_csv(self, include_header=False):
        header = "testName,unit,date,commit,value"

        result = f"{self.name},{self.unit},{self.date},{self.commit},{self.value}"
        if include_header:
            result = header + "\n" + result
        return result


# This function should be called from inside a git repo
# It takes a given commit and returns true if it is somewhere in the branches history
# This function is used when selecting which result to benchmark against, we will take the latest commit which is present in the branch
# This prevents tests failing because someone pushed a performance improvement and a developer hasnt merged


# example output
#   isCommitInProject("34d9c6f4a3c7563d7a4a646e9d69544912932a13")=False
#   isCommitInProject("34d9c6f4a3c7563d7a4a646e9d69544912932a18")=True
# cd ..
#   Not a git repository.
#   isCommitInProject("34d9c6f4a3c7563d7a4a646e9d69544912932a18")=False
def _isCommitInProject(commit):
    # find repo
    try:
        repo = Repo(".", search_parent_directories=True)
    except InvalidGitRepositoryError:
        LOGGER.debug("Not a git repository.")
        return False

    # find branch
    try:
        current_branch = repo.active_branch.name
    except TypeError:
        # Detached HEAD state, no active branch
        current_branch = None

    try:
        # Check if the commit is an ancestor of the current branch
        if current_branch is not None:
            branch_commit = repo.commit(current_branch)
        else:
            # In detached HEAD state, compare with HEAD
            branch_commit = repo.head.commit

        # Check if the given commit is reachable from the branch
        repo.git.merge_base("--is-ancestor", commit, branch_commit.hexsha)
        return True
    except GitCommandError:
        return False  # commit is not an ancestor or doesn't exist


# this function goes through the csv of past benchmark results and finds
# the latest commit which is present in both the csv and the project
# It must be called from inside a git repo
def _findLatestSharedCommitRow(df) -> str | None:

    if "commit" not in df.columns:
        raise ValueError("CSV must contain a 'commit' column")

    # Iterate from bottom to top
    for i in reversed(df.index):
        commit = str(df.at[i, "commit"]).strip()
        if _isCommitInProject(commit):
            LOGGER.debug(f"commit '{commit}' is present in both server and project. returning row {i}.")
            return df.loc[i]
        LOGGER.debug(f"commit '{commit}' is not found in project history.")

    LOGGER.debug("No matching commits found between server and project")


class BenchmarkServer:
    def __init__(self, local=False):  # use a local folder to store data instead of a remote server
        self.benchmarkValues = {}

        self.local = local
        if self.local:
            print("Using a local 'server' under './server'")
            subprocess.run(["mkdir", "-p", "./server"], check=True)
        # TODO could unify these by getting via scp
        # for reading the data we read over internet
        # TODO should certainly not hardcode these
        self.get_url = "https://object-store.os-api.cci1.ecmwf.int/ml-tests/test-data/samples/anemoi-integration-tests/training/benchmarks"
        # for setting the data, we scp
        self.set_remote_path = "/home/data/public/anemoi-integration-tests/training/benchmarks"
        self.set_host = "data@anemoi.ecmwf.int"

    def __str__(self):
        # TODO should do this properly with string builders
        string = ""
        string += "-" * 20 + "\n"
        # benchmark values is a dict of "benchmarkName: BenchmarkValue"
        for benchmark in self.benchmarkValues.values():
            string += str(benchmark) + "\n"
        string += "-" * 20 + "\n"
        return string

    # trys to read a row from a csv stored on a server and create a benchmark value from that
    # If a benchmark value is found, update list of benchmark values
    def getValue(self, benchmarkName: str, forceGetFromServer: bool = False):
        if not forceGetFromServer and benchmarkName in self.benchmarkValues:
            LOGGER.debug(f"entry for {benchmarkName} found locally, not retrieving from server")
            return self.benchmarkValues[benchmarkName]
        if self.local:
            local_file = f"./server/{benchmarkName}"
            try:
                df = pd.read_csv(local_file)

            except FileNotFoundError as e:
                print(f"Could not open file at {local_file}. Got error {e}")
                return None
        else:
            url = f"{self.get_url}/{benchmarkName}"
            LOGGER.debug(f"Fetching benchmark data from {url}...")
            try:
                df = pd.read_csv(url)  # requires pandas 0.19.2, see comments for alternative
                # data = urlopen(url)
                # df=pd.read_csv(io.StringIO(data))
            except URLError as e:  
                print(f"Could not open file at {url}. Got error {e}")
                return None

        # find last element with a commit present in this branch
        # If no such can be found, error and recomend merging main to get a new enough commit
        maybeRow = _findLatestSharedCommitRow(df)
        if maybeRow is None:
            raise RuntimeError(
                "Error. Couldn't find an entry in the server sharing a commit with your branch. Please consider pulling 'main' to enable performance benchmarks",
            )
        row = maybeRow

        assert row["testName"] == benchmarkName  # sanity check, should always pass
        benchmarkValue = BenchmarkValue(
            name=benchmarkName, value=row["value"], unit=row["unit"], date=row["date"], commit=row["commit"],
        )
        LOGGER.debug(benchmarkValue)
        # update dict of results
        self.benchmarkValues[benchmarkValue.name] = benchmarkValue

    def getValues(self, names: list[str]):
        for name in names:
            self.getValue(name)

    # Tests a given benchmark result against what is found on the server
    def compare(self, localValue: BenchmarkValue, failOnMiss=False):
        # check if the server has a reference value
        referenceValue = self.getValue(localValue.name)

        if referenceValue is None:
            if failOnMiss:
                print(f"Benchmark server does not contain a measurement for {localValue.name}")
                return False
            print(f"{localValue.name} not found on server. Passing anyway because 'failOnMiss=False'")
            return True
        LOGGER.debug("didnt pass")
        passed = False

        comp = localValue.op
        refVal = referenceValue.value
        localVal = localValue.value
        tolerance = localValue.tolerance

        # Sanity checking that benchmark metadata matches
        # assert localValue.op == referenceValue.op #wont work, need to pass some inputs
        assert localValue.unit == referenceValue.unit

        percent_diff = 1 - (refVal / localVal)
        passedWithinTolerance = False
        if comp(percent_diff, 0):
            passed = True
        # didnt pass straight away, try pass within tolerance
        elif tolerance != 0 and tolerance / 100 >= abs(percent_diff):
            passed = True
            passedWithinTolerance = True
        else:
            passed = False

        result_str = ""
        if passed:
            if passedWithinTolerance:
                result_str += f"PASS. Local value for {localValue.name} is within {tolerance}% tolerance of the reference value "
            else:
                result_str += (
                    f"PASS. Local value for {localValue.name} has improved compared to the reference value "
                )

        else:
            result_str += f"FAIL. Local value for {localValue.name} has degraded compared to the reference value "
        result_str += f"({localVal:.2f}{localValue.unit} local vs {refVal:.2f}{referenceValue.unit} reference)"
        print(result_str)

        return passed

    # trys to update a metric on a remote server, with a given benchmarkValue
    #if overwrite is true, setValue wont try append. it will be like the exisitng file doesnt exist
    def setValue(self, value: BenchmarkValue, overwrite=False):

        # update remote server with new value
        # append to existing file, but never apend header
        local_file = f"./{value.name}"

        # if file doesnt exist, write header
        if self.local:
            if not os.path.isfile(f"./server/{value.name}") or overwrite:
                with open(local_file, "w") as f:
                    f.write(value.to_csv(include_header=True) + "\n")
            else:
                LOGGER.debug("existing file found... appending")
                with open(f"./server/{value.name}", "a") as f:
                    f.write(value.to_csv() + "\n")
        else:
            #Get existing csv
            url = f"{self.get_url}/{value.name}"
            print(f"Fetching benchmark data from {url}...")
            try:
                df = pd.read_csv(url)  # requires pandas 0.19.2, see comments for alternative
            except URLError as e: 
                print(f"Could not open file at {url}. Got error {e}")
                df = None

            if df is None or overwrite:
                df = pd.read_csv(io.StringIO(value.to_csv(include_header=True))) #, index_col="testName") #index_col to prevent adding a seperate index col
            else:
                new_row = pd.read_csv(io.StringIO(value.to_csv()), header=None)
                new_row.columns = df.columns
                df = pd.concat([df, new_row], ignore_index=True)
            df.to_csv(local_file, index=False)

            cp_cmd = ["scp", local_file, f"{self.set_host}:{self.set_remote_path}/{value.name}"]
            cleanup_cmd = [
                "rm",
                local_file,
            ]
            LOGGER.debug(f"cp command: {cp_cmd}")

            try:
                subprocess.run(cp_cmd, check=True)
                LOGGER.debug(f"Uploaded {value.name} to {self.set_host}")
                subprocess.run(cleanup_cmd, check=True)
            except subprocess.CalledProcessError as e:
                print(f"cp failed: {e}")

        # update dict of results
        self.benchmarkValues[value.name] = value


def get_git_revision_hash() -> str:
    try:
        repo = Repo(".", search_parent_directories=True)
        return repo.head.commit.hexsha
    except InvalidGitRepositoryError:
        raise RuntimeError("Not a Git repository")


def raise_error(x):
    raise ValueError(x)


# this functon will find and open the profiler logs from the most recent benchmarking training run
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


# Function which runs after a profiler run
# It parses the profiler logs and creates BenchmarkValue objects from them
# Returns [BenchmarkValue]
# If you want to add more benchmarks add them here
def getLocalBenchmarkResults():
    # read memory and mlflow stats
    stats = memory_stats(device=0)
    peak_active_mem_mb = stats["active_bytes.all.peak"] / 1024 / 1024
    av_training_throughput = open_log_file("speed_profiler.csv")
    av_training_batch_time_s = open_log_file("time_profiler.csv")

    # get metadata
    commit = get_git_revision_hash()
    yyyy_mm_dd = date.today().strftime("%Y-%m-%d")

    # create Benchmark value objects
    localBenchmarkResults = []
    localBenchmarkResults.append(
        BenchmarkValue(
            name="avThroughputIterPerS",
            value=av_training_throughput,
            unit="iter/s",
            date=yyyy_mm_dd,
            commit=commit,
            op=operator.ge,
            tolerance=5,
        ),
    )
    localBenchmarkResults.append(
        BenchmarkValue(
            name="avTimePerBatchS",
            value=av_training_batch_time_s,
            unit="s",
            date=yyyy_mm_dd,
            commit=commit,
            tolerance=5,
        ),
    )
    localBenchmarkResults.append(
        BenchmarkValue(
            name="peakMemoryMB", value=peak_active_mem_mb, unit="MB", date=yyyy_mm_dd, commit=commit, tolerance=1,
        ),
    )  # added 1% tolerance here so it doesnt fail over a few stray kilobytes

    return localBenchmarkResults


@pytest.mark.longtests
def test_benchmark_training_cycle(
    benchmark_config: tuple[DictConfig, str],
    get_test_archive: callable,
    update_data=False,  # if true, the server will be updated with local values. if false the server values will be compared to local values
    throw_error=True,  # if true, an error will be thrown when a benchmark test is failed
) -> None:
    cfg, urls = benchmark_config
    for url in urls:
        get_test_archive(url)

    # Run model with profiler
    reset_peak_memory_stats()
    AnemoiProfiler(cfg).profile()

    # Get local benchmark results
    localBenchmarkResults = getLocalBenchmarkResults()

    # Get reference benchmark results
    benchmarkServer = BenchmarkServer()
    benchmarks = [
        "avThroughputIterPerS",
        "avTimePerBatchS",
        "peakMemoryMB",
    ]  # TODO get name keys from localBenchmarkResults instead of hardcoding
    benchmarkServer.getValues(benchmarks)

    # print local and reference results
    print(f"Reference benchmark results:\n{benchmarkServer}")
    print("Local benchmark results:")
    print("-" * 20)
    for benchmarkValue in localBenchmarkResults:
        print(benchmarkValue)
    print("-" * 20 + "\n")

    # either update the data on the server, or compare reference results against local results
    if update_data:
        print("Updating metrics on server")
        for localBenchmarkValue in localBenchmarkResults:
            benchmarkServer.setValue(localBenchmarkValue)
    else:
        print("Comparing local benchmark results against reference values from the server")

        # Controls if error or not if a test fails
        on_test_fail = print
        if throw_error:
            on_test_fail = raise_error

        failedTests = []
        for localBenchmarkValue in localBenchmarkResults:
            passed = benchmarkServer.compare(localBenchmarkValue)
            if not passed:
                failedTests.append(localBenchmarkValue.name)

        if len(failedTests) > 0:
            on_test_fail(f"The following tests failed: {failedTests}")

#TODO increase benchmark size and add multigpu
#TODO add graph function to BenchmarkServer?
