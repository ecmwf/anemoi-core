# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import csv
import logging
import operator
import os
import os.path
import re
import shutil
import tarfile
from abc import ABC
from abc import abstractmethod
from collections.abc import Callable
from datetime import datetime
from datetime import timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from git import GitCommandError
from git import InvalidGitRepositoryError
from git import Repo
from omegaconf import DictConfig
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from torch.cuda import memory_stats

os.environ["ANEMOI_BASE_SEED"] = "42"  # need to set base seed if running on github runners
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"  # reduce memory fragmentation

LOGGER = logging.getLogger(__name__)

BENCHMARK_SERVER_ARTIFACT_LIMIT = 10


def parse_benchmark_config(path: Path) -> tuple[str, str, str]:
    with path.open("r") as f:
        benchmark_config = yaml.safe_load(f)
    user = benchmark_config["user"]
    hostname = benchmark_config["hostname"]
    path = benchmark_config["path"]
    return user, hostname, path


class BenchmarkValue:
    """A class which stores information about a benchmark, and functions to output them."""

    def __init__(
        self,
        name: str,
        value: float,
        unit: str,
        date: str,
        commit: str,
        op: Callable[[Any, Any], bool] = operator.le,
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
        return f"{self.name}: {self.value:.2f}{self.unit} (date: {self.date}, commit: {self.commit})"

    def to_csv(self, include_header: bool = False) -> str:
        header = "testName,unit,date,commit,value"

        result = f"{self.name},{self.unit},{self.date},{self.commit},{self.value}"
        if include_header:
            result = header + "\n" + result
        return result


def _make_tarfile(output_filename: str, source_dir: str) -> None:
    """Tars 'source_dir' to 'output_filename'."""
    with tarfile.open(output_filename, "w:gz") as tar:
        tar.add(source_dir, arcname=Path(source_dir).name)


def _tar_files(files: list[Path], tar_file: Path) -> None:
    """Takes a list of files and tars them, returns a path to the tar file."""
    tmp_dir = Path("./.tmp-tar-dir")
    tmp_dir.mkdir()
    for f in files:
        LOGGER.debug("Copying %s to %s...", f, tmp_dir)
        shutil.copy(f, tmp_dir)

    LOGGER.debug("Tar-ing files %s to %s", tmp_dir, tar_file)
    _make_tarfile(tar_file, tmp_dir)

    # clean up tmp_dir
    LOGGER.debug("Deleting %s", tmp_dir)
    shutil.rmtree(tmp_dir)


def _is_repo_on_branch(branch: str) -> bool:
    """Checks if a repo is on a given branch."""
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
        return False

    return branch == current_branch


# example output
#   isCommitInProject("34d9c6f4a3c7563d7a4a646e9d69544912932a13")=False
#   isCommitInProject("34d9c6f4a3c7563d7a4a646e9d69544912932a18")=True
#   cd .. # Not a git repository.
#   isCommitInProject("34d9c6f4a3c7563d7a4a646e9d69544912932a18")=False
def _is_commit_in_project(commit: str) -> bool:
    """Checks if a given commit is in the history of the repo this function was called inside.

    This function should be called from inside a git repo.
    It takes a given commit str and returns true if it is somewhere in the branches history
    This function is used when selecting which result to benchmark against,
    we will take the latest commit which is present in the branch
    This prevents tests failing because someone pushed a performance improvement and a developer hasnt merged
    """
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
        # if None -> In detached HEAD state, compare with HEAD
        branch_commit = repo.commit(current_branch) if current_branch is not None else repo.head.commit

        # Check if the given commit is reachable from the branch
        repo.git.merge_base("--is-ancestor", commit, branch_commit.hexsha)
    except GitCommandError:
        return False  # commit is not an ancestor or doesn't exist
    else:
        return True


def _find_latest_shared_commit(df: pd.DataFrame) -> str | None:
    """Finds the last shared commit betwen a csv and your current github repo.

    This function goes through the csv of past benchmark results and finds
    the latest commit which is present in both the csv and the project
    It must be called from inside a git repo
    """
    if "commit" not in df.columns:
        msg = "CSV must contain a 'commit' column"
        raise ValueError(msg)

    # Iterate from bottom to top
    for i in reversed(df.index):
        commit = str(df.at[i, "commit"]).strip()
        if _is_commit_in_project(commit):
            LOGGER.debug("commit '%s' is present in both server and project. returning row %s.", commit, i)
            return df.loc[i]
        LOGGER.debug("commit '%s' is not found in project history.", commit)

    LOGGER.debug("No matching commits found between server and project")
    return None


class BenchmarkServer(ABC):
    """Stores information from past benchmarks.

    includes methods to retrieve benchmark data and compare against new benchmark data.
    """

    def __init__(self, store: Path, test_case: str = ""):
        self.benchmark_values = {}

        # TestCase creates an optional subdir under BenchmarkServer to store the results
        # If testcase is "" then no subdirs are created
        self.test_case = test_case
        if self.test_case:
            self.store = Path(f"{store}/{self.test_case}")
        else:
            self.store = Path(f"{store}")

        self.artifactLimit = BENCHMARK_SERVER_ARTIFACT_LIMIT  # How many commits artifacts will be saved at once.
        # currently the trace file and memory snapshot are saved
        # When the artifactLimit is hit, the oldest commits artifacts are deleted
        # Artifacts can be reproduced by reverting to a given commit and running the pytests locally

    def __str__(self):
        string = ""
        string += "-" * 20 + "\n"
        # benchmark values is a dict of "benchmark_name: BenchmarkValue"
        for benchmark in self.benchmark_values.values():
            string += str(benchmark) + "\n"
        string += "-" * 20 + "\n"
        string += f"(Server location: '{self.store}')\n"
        return string

    def get_value(self, benchmark_name: str) -> None:
        """Retrieves a given benchmark from the store and stores it under self.benchmark_values[benchmark_name].

        If a benchmark value has already been loaded, return it.
        If the benchmark store is remote, the CSV is downloaded.
        Then the CSV is opend and the value retrieved.
        If the name cant be found in the benchmark store, the function returns
        trys to read a row from a csv and create a benchmark value from that
        If a benchmark value is found, update list of benchmark values.
        """
        if benchmark_name in self.benchmark_values:
            LOGGER.debug("entry for %s found locally, not retrieving from server", benchmark_name)
            return self.benchmark_values[benchmark_name]

        bench_file = Path(f"{self.store}/{benchmark_name}")
        local_file = Path(f"./{benchmark_name}")

        if self._exists(bench_file):
            self._get(bench_file, local_file)
            df = pd.read_csv(local_file)
            local_file.unlink()  # clean up local file
        else:
            LOGGER.info("Could not find file at %s.", bench_file)
            return None

        # find last element with a commit present in this branch
        # If no such can be found, error and recomend merging main to get a new enough commit
        maybe_row = _find_latest_shared_commit(df)
        if maybe_row is None:
            msg = "Error. Couldn't find an entry in the server sharing a commit with your branch.\n"
            msg += "Please consider merging 'main' to enable performance benchmarks"
            raise RuntimeError(msg)
        row = maybe_row

        assert row["testName"] == benchmark_name  # sanity check, should always pass
        benchmark_value = BenchmarkValue(
            name=benchmark_name,
            value=row["value"],
            unit=row["unit"],
            date=row["date"],
            commit=row["commit"],
        )
        LOGGER.debug(benchmark_value)
        # update dict of results
        self.benchmark_values[benchmark_value.name] = benchmark_value

        return None

    def get_values(self, names: list[str]) -> None:
        """Retrieves a list of benchmarks from the server."""
        for name in names:
            self.get_value(name)

    def compare(self, local_value: BenchmarkValue, fail_on_miss: bool = False) -> bool:
        """Tests a given benchmark result against what is found on the server.

        Takes a given benchmark value, and checks the server if there is a matching benchmark value.
        returns true if the given value is 'better' then the value on the benchmark

        """
        # check if the server has a reference value
        reference_value = self.get_value(local_value.name)
        if reference_value is None:
            if fail_on_miss:
                LOGGER.info("Benchmark server does not contain a measurement for %s", local_value.name)
                return False
            LOGGER.info("%s not found on server. Passing anyway because 'fail_on_miss=False'", local_value.name)
            return True

        passed = False

        comp = local_value.op
        ref_val = reference_value.value
        local_val = local_value.value
        tolerance = local_value.tolerance

        # Sanity checking that benchmark metadata matches
        assert local_value.unit == reference_value.unit

        # Check if tests pass outright
        percent_diff = 1 - (ref_val / local_val)
        passed_within_tolerance = False
        if comp(percent_diff, 0):
            passed = True
        # didnt pass straight away, try pass within tolerance
        elif tolerance != 0 and tolerance / 100 >= abs(percent_diff):
            passed = True
            passed_within_tolerance = True
        else:
            passed = False

        result_str = ""
        if passed:
            if passed_within_tolerance:
                result_str += (
                    f"PASS. Local value for {local_value.name} is within {tolerance}% tolerance of the reference value "
                )
            else:
                result_str += f"PASS. Local value for {local_value.name} has improved compared to the reference value "

        else:
            result_str += f"FAIL. Local value for {local_value.name} has degraded compared to the reference value "
        result_str += f"({local_val:.2f}{local_value.unit} local vs {ref_val:.2f}{reference_value.unit} reference)"
        LOGGER.info(result_str)

        return passed

    def set_value(self, value: BenchmarkValue, overwrite: bool = False) -> None:
        """Trys to update a metric on a remote server, with a given benchmark_value.

        if overwrite is true, set_value wont try append. it will be like the exisitng file doesnt exist
        """
        # Check do we have an existing value
        output = Path(f"{self.store}/{value.name}")
        exists = True
        if overwrite:
            exists = False
        exists = self._exists(output)

        # If we have an existing copy, get it into local_file
        local_file = Path(f"./{value.name}")
        if exists:
            self._get(output, local_file)

        # If the file exists just write value
        if exists:
            with local_file.open("a") as f:
                f.write(value.to_csv() + "\n")
        else:
            # if file doesnt exist, write header
            with local_file.open("w") as f:
                f.write(value.to_csv(include_header=True) + "\n")

        # Copy  local_file back to server and delete it
        self._put(local_file, output)
        local_file.unlink()  # delete local file

        # update dict of results
        self.benchmark_values[value.name] = value

        return

    def store_artifacts(self, artifacts: list[Path], commit: str) -> None:
        """Takes a list of files and stores them on the server, under a commit folder.

        if the files exist already, by default nothing will be stored
        tar-ing reduced the size of an artifact dir from 450MB (420MB was the trace) to 22MB
        """
        artifact_dir = Path(f"{self.store}/artifacts")
        commit_tar = Path(f"./{commit}.tar.gz")  # store commits locally before copuing them to the server

        LOGGER.debug("Saving artifacts for commit %s under %s", commit, commit_tar)
        if commit_tar.exists():
            LOGGER.info("Artifacts have already been saved for commit %s under %s. Not saving...", commit, commit_tar)
            return

        _tar_files(artifacts, commit_tar)

        # Move tar to server and delete local copy
        self._mkdir(artifact_dir)
        self._put(commit_tar, artifact_dir)
        commit_tar.unlink()  # delete local commit tar

        # cleanup oldest artifact if we are over artifact limit
        commits = self._sort_files_by_age(artifact_dir)

        if len(commits) > self.artifactLimit:
            LOGGER.info(
                "%s commits stored under %s, greater then server limit of %s",
                len(commits),
                artifact_dir,
                self.artifactLimit,
            )

            commits_to_delete = commits[: len(commits) - self.artifactLimit]
            LOGGER.info("Deleting %s...", commits_to_delete)
            for commit in commits_to_delete:
                self._rm(commit)

    @abstractmethod
    def _exists(self, path: Path) -> bool:
        """Check if a path exists on a server."""
        ...

    @abstractmethod
    def _mkdir(self, path: Path) -> None:
        """Creates a directory on the server."""
        ...

    @abstractmethod
    def _get(self, src: Path, dest: Path) -> None:
        """Retrieves a file from the server."""
        ...

    @abstractmethod
    def _put(self, src: Path, dest: Path) -> None:
        """Puts a file into the server."""
        ...

    @abstractmethod
    def _rm(self, path: Path) -> None:
        """Deletes a file from a server."""
        ...

    @abstractmethod
    def _sort_files_by_age(self, path: Path) -> list[str]:
        """Sorts files under path, oldest first."""
        ...


def parse_benchmark_location(store: str, test_case: str = "") -> BenchmarkServer:
    """Parses an input string to determine where to store the benchmark servers files.

    store: str -> either a local path or a remote path. Remote paths should be in the form
                "ssh://<user>@<dest>:<remote_path>"

    retuns: A local or remote benchmark server based on the store location.
    """
    # a string which starts with ".ssh" and has a "@" and ":" in the middle
    remote_pattern = r"^ssh://.*@.*:.*$"
    if re.match(remote_pattern, store):
        # looks like a remote string
        parts = store.removeprefix("ssh://").split(":")
        remote = parts[0].split("@")
        remote_user = str(remote[0])
        remote_host = str(remote[1])
        store_path = Path(parts[1])
        LOGGER.debug("'%s' looks like a remote store pointing to %s on %s", store, store_path, remote)
        return RemoteBenchmarkServer(store_path, remote_user, remote_host, test_case=test_case)
    store = Path(store)
    return LocalBenchmarkServer(store, test_case=test_case)


def get_git_revision_hash() -> str:
    """Gets the commit of a given git repo."""
    try:
        repo = Repo(".", search_parent_directories=True)
    except InvalidGitRepositoryError as e:
        msg = "Not a Git repository"
        raise RuntimeError(msg) from e
    else:
        return repo.head.commit.hexsha


class LocalBenchmarkServer(BenchmarkServer):

    def __init__(self, store: Path, test_case: str = ""):
        super().__init__(store, test_case=test_case)

        # create store locally
        self.store.mkdir(parents=True, exist_ok=True)

    def _exists(self, path: Path) -> bool:
        """Check if a path exists on a local server."""
        return path.exists()

    def _mkdir(self, path: Path) -> None:
        """Creates a directory on the local server."""
        path.mkdir(parents=True)

    def _get(self, src: Path, dest: Path) -> None:
        """Retrieves a file from the server."""
        shutil.copy(src, dest)

    def _put(self, src: Path, dest: Path) -> None:
        """Puts a file into the server."""
        shutil.copy(src, dest)

    def _rm(self, path: Path) -> None:
        """Deletes a file from local server."""
        path.unlink()

    def _sort_files_by_age(self, path: Path) -> list[str]:
        """Sorts files under path, oldest first."""
        # listdir gets file name name, and the list compression makes it a complete path
        files = [str(p) for p in path.iterdir()]
        files.sort(key=os.path.getmtime)  # sorts the list, oldest first
        return files


class RemoteBenchmarkServer(BenchmarkServer):

    def __init__(
        self,
        store: str,
        remote_user: str,
        remote_host: str,
        test_case: str = "",
    ):
        super().__init__(store, test_case=test_case)

        self.remote_user = remote_user
        self.remote_host = remote_host

        # mount the remote server
        self._mount_remote()

        # create store remotely
        self._mkdir(self.store)

    def _mkdir(self, path: Path) -> None:
        """Creates a directory on the remote server."""
        if self.fs is None:
            msg = f"Error. Tried to make a directory at {path} on an unmounted remote server"
            raise ValueError(msg)
        self.fs.mkdir(str(path), create_parents=True)

    def _get(self, src: Path, dest: Path) -> None:
        """Retrieves a file from the server.

        src: path to file on the remote server
        dest: path to file locally
        """
        if self.fs is None:
            msg = f"Error. Tried to retrieve {src} from an unmounted remote server"
            raise ValueError(msg)
        self.fs.get(str(src), str(dest))

    def _put(self, src: Path, dest: Path) -> None:
        """Puts a file into the server.

        src: path to file on local server
        dest: path to file on remote server
        """
        if self.fs is None:
            msg = f"Error. Tried to put a file to {dest} on an unmounted remote server"
            raise ValueError(msg)
        self.fs.put_file(str(src), str(dest))

    def _exists(self, path: Path) -> bool:
        """Checks if a path exists on a remote server."""
        if self.fs is None:
            msg = f"Error. Tried to check the existance of {path} on an unmounted remote server"
            raise ValueError(msg)
        return self.fs.exists(str(path))

    def _rm(self, path: Path) -> None:
        """Deletes a file from a remote server."""
        if self.fs is None:
            msg = f"Error. Tried to delete a file at {path} on an unmounted remote server"
            raise ValueError(msg)
        self.fs.rm_file(path)

    def _sort_files_by_age(self, path: Path) -> list[str]:
        """Sorts files under path, oldest first."""
        if self.fs is None:
            msg = f"Error. Tried to sort files under {path} on an unmounted remote server"
            raise ValueError(msg)
        files = self.fs.listdir(str(path))  # returns a list of info dicts
        files = sorted(files, key=lambda d: d["mtime"])
        return [f["name"] for f in files]  # f is a dict of info, f["name"] is just the path

    def _mount_remote(self) -> None:
        """Mounts the remote server over sftp using self.remote_host and self.remote_user."""
        from sshfs import SSHFileSystem

        self.fs = SSHFileSystem(self.remote_host, username=self.remote_user)
        return


# this functon will find and open the profiler logs from the most recent benchmarking training run
# return_val = value for speed profiler or 'avg_time' for time_profiler
def open_log_file(profiler_path: str, filename: str) -> float:
    if filename == "time_profiler.csv":
        return_val = "avg_time"
        row_selector = "name"
        row_name = "run_training_batch"
    elif filename == "speed_profiler.csv":
        return_val = "value"
        row_selector = "metric"
        row_name = "training_avg_throughput"
    else:
        msg = f"Tried to open unknown log file: {filename}"
        raise ValueError(msg)

    # under /{profiler_path} there is a single random alphanumeric dir
    try:
        profiler_dir = next(iter(Path(profiler_path).glob("[a-z0-9]*/")))
    except IndexError as e:
        msg = f"Could not find a profiler dir under {profiler_path}."
        raise IndexError(msg) from e
    file_path = f"{profiler_dir}/{filename}"
    with Path(file_path).open(newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if row.get(row_selector) == row_name:
                result = row.get(return_val)
                break
    return float(result)


def get_local_benchmark_results(profiler_path: str) -> list[BenchmarkValue]:
    """Function which runs after a profiler run.

    It parses the profiler logs and creates BenchmarkValue objects from them

    Returns [BenchmarkValue]
    If you want to add more benchmarks add them here
    """
    # read memory and mlflow stats
    stats = memory_stats(device=0)
    peak_active_mem_mb = stats["active_bytes.all.peak"] / 1024 / 1024
    av_training_throughput = open_log_file(profiler_path, "speed_profiler.csv")

    # get metadata
    commit = get_git_revision_hash()
    yyyy_mm_dd = datetime.now(tz=timezone.utc).date()

    # create Benchmark value objects
    local_benchmark_results = []
    local_benchmark_results.append(
        BenchmarkValue(
            name="avThroughputIterPerS",
            value=av_training_throughput,
            unit="iter/s",
            date=yyyy_mm_dd,
            commit=commit,
            op=operator.ge,
            tolerance=10,
        ),
    )
    local_benchmark_results.append(
        BenchmarkValue(
            name="peakMemoryMB",
            value=peak_active_mem_mb,
            unit="MB",
            date=yyyy_mm_dd,
            commit=commit,
            tolerance=1,
        ),
    )  # added 1% tolerance here so it doesnt fail over a few stray kilobytes

    return local_benchmark_results


def get_local_benchmark_artifacts(profiler_path: str) -> list[Path]:
    """Runs after a benchmark and returns a list of paths to artifacts produced by the profiler.

    Currently it captures the pytorch trace file and the memory snapshot
    """
    profiler_path = Path(profiler_path)
    profiler_dir = next(iter(profiler_path.glob("[a-z0-9]*/")))

    # get memory snapshot
    memory_snapshot = Path(f"{profiler_dir}/memory_snapshot.pickle")
    if not memory_snapshot.exists():
        msg = f"Memory snapshot not found at: {memory_snapshot}"
        raise RuntimeError(msg)

    artifacts = [memory_snapshot]

    # get trace file
    # there can be multiple ${hostname}_${pid}\.None\.[0-9]+\.pt\.trace\.json files. 1 training + 1 valdation per device
    # but luckily if we take the first one thats always training on rank 0.
    trace_files = list(profiler_dir.glob("*.pt.trace.json"))
    if len(trace_files) == 0:
        LOGGER.info("Can't find a trace file under %s", profiler_dir)
    else:
        trace_file = Path(trace_files[0])
        if not trace_file.exists():
            msg = f"trace file not found at: {trace_file}"
            raise RuntimeError(msg)
        artifacts.append(trace_file)

    return artifacts


def _print_local_benchmark_results(local_benchmark_results: list[BenchmarkValue]) -> str:
    local_results_str = "Local benchmark results:\n"
    local_results_str += "-" * 20 + "\n"
    for benchmark_value in local_benchmark_results:
        local_results_str += str(benchmark_value) + "\n"
    local_results_str += "-" * 20 + "\n"
    return local_results_str


@rank_zero_only
def benchmark(
    cfg: DictConfig,
    test_case: str,
    store: str,
    update_data: bool = False,  # when this is true, data is always updated
) -> None:
    local_benchmark_results = get_local_benchmark_results(cfg.hardware.paths.profiler)

    # Get reference benchmark results
    benchmark_server = parse_benchmark_location(store, test_case=test_case)

    benchmarks = [benchmark_value.name for benchmark_value in local_benchmark_results]
    benchmark_server.get_values(benchmarks)

    # print local and reference results
    LOGGER.info("Reference benchmark results:\n%s", benchmark_server)
    LOGGER.info(_print_local_benchmark_results(local_benchmark_results))

    # compare reference results against local results
    LOGGER.info("Comparing local benchmark results against reference values from the server")

    failed_tests = []
    for local_benchmark_value in local_benchmark_results:
        passed = benchmark_server.compare(local_benchmark_value)
        if not passed:
            failed_tests.append(local_benchmark_value.name)

    if len(failed_tests) > 0:
        msg = f"The following tests failed: {failed_tests}"
        artifacts = get_local_benchmark_artifacts(cfg.hardware.paths.profiler)
        artifacts_tar = Path(f"./{test_case}_artifacts.tar.gz")
        _tar_files(artifacts, artifacts_tar)
        LOGGER.info("Profiling artifacts from failed run stored under: %s", artifacts_tar)
        raise ValueError(msg)
    # the tests have passed, possibly update the data on the server
    update_data = update_data or _is_repo_on_branch("main")  # update if our branch is main
    if update_data:
        LOGGER.info("Updating metrics on server")
        for local_benchmark_value in local_benchmark_results:
            benchmark_server.set_value(local_benchmark_value)
            artifacts = get_local_benchmark_artifacts(cfg.hardware.paths.profiler)
            benchmark_server.store_artifacts(artifacts, local_benchmark_results[0].commit)
