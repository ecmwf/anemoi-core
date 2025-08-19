# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import io
import logging
import operator
import os
import os.path
import shutil
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
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"  # reduce memory fragmentation

LOGGER = logging.getLogger(__name__)


BENCHMARK_SERVER_ARTIFACT_LIMIT=10


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

    def to_csv(self, include_header=False):
        header = "testName,unit,date,commit,value"

        result = f"{self.name},{self.unit},{self.date},{self.commit},{self.value}"
        if include_header:
            result = header + "\n" + result
        return result


def _make_tarfile(output_filename, source_dir):
    import os.path
    import tarfile

    with tarfile.open(output_filename, "w:gz") as tar:
        tar.add(source_dir, arcname=os.path.basename(source_dir))


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
    #def __init__(self, store:str="./local", testCase:str=""):  # use a local folder to store data instead of a remote server
    def __init__(self, store:str="ssh://data@anemoi.ecmwf.int:/home/data/public/anemoi-integration-tests/training/benchmarks", testCase:str=""):  # use a local folder to store data instead of a remote server
        self.benchmarkValues = {}
    
        self._parse_store_location(store)

        # TestCase creates an optional subdir under BenchmarkServer to store the results
        # So that you can store GNN_n320_1g and graphtransformer_n320_1g results under the same server
        # If testcase is "" then no subdirs are created
        self.testCase = testCase
        if self.testCase is not "":
            self.store = Path(f"{self.store}/{self.testCase}")

        if not self.local:
            self._mount_remote()

        if self.local:
            self.store.mkdir(parents=True, exist_ok=True)
        else: 
            self.fs.mkdir(str(self.store), create_parents=True)

        self.artifactLimit=BENCHMARK_SERVER_ARTIFACT_LIMIT #How many commits artifacts will be saved at once.
        #currently the trace file and memory snapshot are saved
        #When the artifactLimit is hit, the oldest commits artifacts are deleted
        #Artifacts can be reproduced by reverting to a given commit and running the pytests locally

    def _parse_store_location(self, store:str) -> None:
        """Parses an input string to determine where to store the benchmark servers files

        store: str -> either a local path or a remote path. Remote paths should be in the form
                    "ssh://<user>@<dest>:<remote_path>"

        retuns: None, but sets self.store and self.remote_user,self.remote_host if remote
        """

        import re

        #a string which starts with ".ssh" and has a "@" and ":" in the middle
        remote_pattern=r'^ssh://.*@.*:.*$'  
        if re.match(remote_pattern, store): 
            #looks like a remote string
            parts=store.strip("ssh://").split(":")
            remote=parts[0].split("@")
            self.remote_user=str(remote[0])
            self.remote_host=str(remote[1])
            self.store=Path(parts[1])
            #'%s' looks like a remote store pointing to %s on %s ssh://data@anemoi.ecmwf.int:/home/data/public/anemoi-integration-tests/training/benchmarks /home/data/public/anemoi-integration-tests/training/benchmark data@anemoi.ecmwf.int
            LOGGER.debug("'%s' looks like a remote store pointing to %s on %s", store, self.store, remote)
            self.local=False
        else:
            #TODO could write a regex to check if its a valid path
            self.local=True
            self.store=Path(store)
            LOGGER.info("'%s' is a local store pointing to %s", store, str(self.store))

    # mounts the remote server over sftp
    def _mount_remote(self):
        from sshfs import SSHFileSystem
        self.fs = SSHFileSystem(
            self.remote_host,
            username=self.remote_user
        )

    def __str__(self):
        # TODO should do this properly with string builders
        string = ""
        string += "-" * 20 + "\n"
        # benchmark values is a dict of "benchmarkName: BenchmarkValue"
        for benchmark in self.benchmarkValues.values():
            string += str(benchmark) + "\n"
        string += "-" * 20 + "\n"
        if self.local:
            string += f"(Server location: '{self.store}')\n"
        else:
            string += f"(Server location: '{self.remote_host}:{self.store}')\n"
        return string

    # trys to read a row from a csv stored on a server and create a benchmark value from that
    # If a benchmark value is found, update list of benchmark values
    def getValue(self, benchmarkName: str, forceGetFromServer: bool = False):
        if not forceGetFromServer and benchmarkName in self.benchmarkValues:
            LOGGER.debug(f"entry for {benchmarkName} found locally, not retrieving from server")
            return self.benchmarkValues[benchmarkName]

        bench_file = Path(f"{self.store}/{benchmarkName}")
        if self.local:
            if bench_file.exists():
                df = pd.read_csv(bench_file)
            else:
                LOGGER.info(f"Could not find file at {bench_file}.")
                return None
        else:
            local_file= Path(f"./{benchmarkName}")
            try:
                self.fs.get(str(bench_file), str(local_file))
                df = pd.read_csv(local_file)
            except IOError:
                LOGGER.info(f"Could not find file at {bench_file}.")
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
            name=benchmarkName,
            value=row["value"],
            unit=row["unit"],
            date=row["date"],
            commit=row["commit"],
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
                LOGGER.info(f"Benchmark server does not contain a measurement for {localValue.name}")
                return False
            LOGGER.info(f"{localValue.name} not found on server. Passing anyway because 'failOnMiss=False'")
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
                result_str += (
                    f"PASS. Local value for {localValue.name} is within {tolerance}% tolerance of the reference value "
                )
            else:
                result_str += f"PASS. Local value for {localValue.name} has improved compared to the reference value "

        else:
            result_str += f"FAIL. Local value for {localValue.name} has degraded compared to the reference value "
        result_str += f"({localVal:.2f}{localValue.unit} local vs {refVal:.2f}{referenceValue.unit} reference)"
        LOGGER.info(result_str)

        return passed

    # trys to update a metric on a remote server, with a given benchmarkValue
    # if overwrite is true, setValue wont try append. it will be like the exisitng file doesnt exist
    def setValue(self, value: BenchmarkValue, overwrite=False):

        #Check do we have an existing value
        output=Path(f"{self.store}/{value.name}")
        exists=True
        if overwrite:
            exists=False
        if self.local:
            exists = output.exists()
        else:
            exists = self.fs.exists(str(output))

        #If we have an existing copy, get it into local_file
        local_file=Path(f"./{value.name}")
        if exists:
            if self.local:
                shutil.copy(output, local_file)
            else:
                self.fs.get(str(output), str(local_file))

        #If the file exists just write value
        if exists:
            with open(local_file, "a") as f:
                f.write(value.to_csv() + "\n")
        else:
        # if file doesnt exist, write header
            with open(local_file, "w") as f:
                 f.write(value.to_csv(include_header=True) + "\n")     

        #Copy  local_file back to server and delete it
        if self.local:
            shutil.copy(local_file, output)
        else:
            LOGGER.info(f"Copying {local_file} to {self.store}/{value.name}")
            self.fs.put_file(str(local_file), str(output))
        local_file.unlink() # delete local file
            
        # update dict of results
        self.benchmarkValues[value.name] = value

    # takes a list of files and stores them on the server, under a commit folder
    # if the files exist already, by default nothing will be stored
    # Optionally (but strongly recomended) the artifacts will be tar-ed by default
    # tar-ing reduced the size of an artifact dir from 450MB (420MB was the trace) to 22MB
    def storeArtifacts(self, artifacts: list[Path], commit: str, tar=True) -> None:

        if not self.local and not tar:
            LOGGER.info("Uploading untarred to server not supported")
            return

        artifactDir=Path(f"{self.store}/artifacts")
        commitDir=Path(f"{artifactDir}/{commit}")
        if not self.local: #copy locally before tarring and sending to server
              commitDir=Path(f"./{commit}")
        commitTar=Path(f"{commitDir}.tar.gz")
        output = commitDir
        if tar:
            output = commitTar

        LOGGER.debug(f"Saving artifacts for commit {commit} under {output}")
        if output.exists():
            #TODO this doesnt work remote, but it should just overwrite
            LOGGER.info(f"Artifacts have already been saved for commit {commit} under {output}. Not saving...")
            # return
        else:
            commitDir.mkdir(parents=True) # might need to make artifacts too

            for artifact in artifacts:
                LOGGER.debug(f"Copying {artifact} to {commitDir}...")
                shutil.copy(artifact, commitDir)

            if tar:
                LOGGER.debug("Tar-ing artifacts {commitDir} to {commitTar}")
                _make_tarfile(commitTar, commitDir)
                # cleanup untar-ed file
                LOGGER.debug("Deleting {commitDir}")
                shutil.rmtree(commitDir)

        if not self.local:
            LOGGER.info(f"Copying tar file from {commitTar} to {artifactDir}")
            self.fs.mkdir(str(artifactDir), create_parents=True)
            self.fs.put_file(str(commitTar), str(artifactDir))
            commitTar.unlink() #delete local commit tar

        #cleanup oldest artifact if we are over artifact limit

        if self.local:
            remove=os.remove
            #listdir gets commit name, and the list compression makes it a complete path
            commits = [ f"{artifactDir}/{commit}" for commit in os.listdir(f"{artifactDir}")]
            commits.sort(key=os.path.getmtime) #sorts the list, oldest first
        else:
            remove=self.fs.rm_file
            commits = self.fs.listdir(f"{artifactDir}") #returns a list of info dicts
            commits = sorted(commits, key=lambda d: d['mtime'])
            commits = [commit['name'] for commit in commits] #commit is a dict of info, now that we've sorted drop to just paths

        if len(commits) >  self.artifactLimit:
            LOGGER.info(f"{len(commits)} commits stored under {artifactDir}, greater then server limit of {self.artifactLimit}")

            commitsToDelete = commits[:len(commits) - self.artifactLimit]
            LOGGER.info(f"Deleting {commitsToDelete}...")
            for commit in commitsToDelete:
                remove(commit)


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
def open_log_file(profilerPath: str, filename: str):
    import csv
    import glob

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

    # under /{profilerPath} there is a single random alphanumeric dir
    # this next(iter(glob(...))) gets us through this random dir
    file_path = next(
        iter(
            glob.glob(
                f"{profilerPath}/[a-z0-9]*/{filename}",
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
def getLocalBenchmarkResults(profilerPath: str) -> list[BenchmarkValue]:
    # read memory and mlflow stats
    stats = memory_stats(device=0)
    peak_active_mem_mb = stats["active_bytes.all.peak"] / 1024 / 1024
    av_training_throughput = open_log_file(profilerPath, "speed_profiler.csv")
    av_training_batch_time_s = open_log_file(profilerPath, "time_profiler.csv")

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
            name="peakMemoryMB",
            value=peak_active_mem_mb,
            unit="MB",
            date=yyyy_mm_dd,
            commit=commit,
            tolerance=1,
        ),
    )  # added 1% tolerance here so it doesnt fail over a few stray kilobytes

    return localBenchmarkResults


# Runs after a benchmark
# returns a list of files produced by the profiler
def getLocalBenchmarkArtifacts(profilerPath:str) -> list[Path]:
    import glob

    profilerDir = glob.glob(f"{profilerPath}/[a-z0-9]*/")[0]
    memory_snapshot = Path(f"{profilerDir}/memory_snapshot.pickle")
    if not memory_snapshot.exists():
        raise RuntimeError(f"Memory snapshot not found at: {memory_snapshot}")

    artifacts = [memory_snapshot]

    # get trace file
    # there can be multiple ${hostname}_${pid}\.None\.[0-9]+\.pt\.trace\.json files. 1 training + 1 valdation per device
    # but luckily if we take the first one thats always training on rank 0.
    trace_files = glob.glob(f"{profilerDir}/*.pt.trace.json")
    if len(trace_files) == 0:
        LOGGER.info(f"Can't find a trace file under {profilerDir}")
    else:
        trace_file = Path(trace_files[0])
        if not trace_file.exists():
            raise RuntimeError(f"trace file not found at: {trace_file}")
        artifacts.append(trace_file)
    return artifacts


@pytest.mark.multigpu
@pytest.mark.slow
def test_benchmark_training_cycle(
    benchmark_config: tuple[DictConfig, str],  # cfg, benchmarkTestCase
    get_test_archive: callable,
    update_data=False,  # if true, the server will be updated with local values. if false the server values will be compared to local values
    throw_error=True,  # if true, an error will be thrown when a benchmark test is failed
) -> None:
    cfg, testCase = benchmark_config
    LOGGER.info(f"Benchmarking the configuration: {testCase}")

    # Run model with profiler
    reset_peak_memory_stats()
    AnemoiProfiler(cfg).profile()

    # Get local benchmark results
    localBenchmarkResults = getLocalBenchmarkResults(cfg.hardware.paths.profiler)

    # Get reference benchmark results
    benchmarkServer = BenchmarkServer(testCase=testCase)

    benchmarks = [benchmarkValue.name for benchmarkValue in localBenchmarkResults]
    if not update_data:
        benchmarkServer.getValues(benchmarks)

    # print local and reference results
    LOGGER.info(f"Reference benchmark results:\n{benchmarkServer}")
    LOGGER.info("Local benchmark results:")
    LOGGER.info("-" * 20)
    for benchmarkValue in localBenchmarkResults:
        LOGGER.info(benchmarkValue)
    LOGGER.info("-" * 20 + "\n")

    # either update the data on the server, or compare reference results against local results
    if update_data:
        LOGGER.info("Updating metrics on server")
        for localBenchmarkValue in localBenchmarkResults:
            benchmarkServer.setValue(localBenchmarkValue)
        store_artifacts = True
        if store_artifacts:
            artifacts = getLocalBenchmarkArtifacts(cfg.hardware.paths.profiler)
            benchmarkServer.storeArtifacts(artifacts, localBenchmarkResults[0].commit)
    else:
        LOGGER.info("Comparing local benchmark results against reference values from the server")

        # Controls if error or not if a test fails
        on_test_fail = LOGGER.info
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
#TODO change hidden res for GT
#TODO refactor benchmark server into seperate file
#TODO when running multi-gpu, make sure only gpu 0 does benchmark server stuff
