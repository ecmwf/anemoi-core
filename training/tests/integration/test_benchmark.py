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
import os.path
from pathlib import Path

import pytest
from omegaconf import DictConfig
from torch.cuda import memory_stats
from torch.cuda import reset_peak_memory_stats

from anemoi.training.train.profiler import AnemoiProfiler

import json
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError
import subprocess
from datetime import date
import operator
import pandas as pd
import io 

os.environ["ANEMOI_BASE_SEED"] = "42"  # need to set base seed if running on github runners

LOGGER = logging.getLogger(__name__)

class BenchmarkValue():
    def __init__(
        self,
        name: str,
        value: float,
        unit: str,
        date: date,
        commit: str,
        op=operator.le,
        tolerance:int=0 #percentage 
            ):
        self.name=name
        self.value=value
        self.unit=unit
        self.date=date
        self.commit=commit
        self.op=op
        self.tolerance=tolerance

    def __str__(self):
        return f"{self.name}: {self.value}{self.unit}"

    #header="testName,unit,date,commit,value" 
    def to_csv(self):
        return f"{self.name},{self.unit},{self.date},{self.commit},{self.value}"

class BenchmarkServer():
    def __init__(self, 
            local=True #use a local folder to store data instead of a remote server
            ):
        self.benchmarkValues={}


        self.local=local
        if self.local:
            print("Using a local 'server' under './server'")
            subprocess.run(["mkdir", "-p", "./server"], check=True)
        #TODO could unify these by getting via scp
        #for reading the data we read over internet
        self.get_url="https://object-store.os-api.cci1.ecmwf.int/ml-tests/test-data/samples/anemoi-integration-tests/training/benchmarks"
        #for setting the data, we scp
        self.set_remote_path="/home/data/public/anemoi-integration-tests/training/benchmarks"
        self.set_host="data@anemoi.ecmwf.int"

    def __str__(self):
        #TODO should do this properly with string builders
        string =""
        string += "-"*20 + "\n"
        #benchmark values is a dict of "benchmarkName: BenchmarkValue"
        for benchmark in self.benchmarkValues.values():
            string += str(benchmark) + "\n"
        string += "-"*20 + "\n"
        return string

    #trys to read a metric from 'self.get_url'
    #If the metric exists, update list of benchmark values
    #assumes file is just a single line with a single value
    # TODO return none if value cant be found
    def getValue(self, benchmarkName:str, forceGetFromServer:bool = False):
        if not forceGetFromServer and benchmarkName in self.benchmarkValues:
            LOGGER.debug(f"entry for {benchmarkName} found locally, not retrieving from server")
            return self.benchmarkValues[benchmarkName]
        else:
            if self.local:
                local_file = f"./server/{benchmarkName}"
                try:
                    df = pd.read_csv(local_file)
                    #with open(local_file, "r") as f:
                        #data=f.read()

                except FileNotFoundError as e:
                    print(f"Could not open file at {local_file}. Got error {e}")
                    return None
            else:
                url = f"{self.get_url}/{benchmarkName}"
                print(f"Fetching benchmark data from {url}...")
                try:
                    df = pandas.read_csv(url) #requires pandas 0.19.2, see comments for alternative
                    #data = urlopen(url)  
                    #df=pd.read_csv(io.StringIO(data))
                except URLError as e: #TODO test this
                    print(f"Could not open file at {url}. Got error {e}")
                    return None

            #header="testName,unit,date,commit,value"
            assert df["testName"].iloc[-1] == benchmarkName
            #TODO instead of finding last element, find last element with a commit present in this branch
            # If no such can be found, error and recomend merging main to get a new enough commit
            benchmarkValue = BenchmarkValue(name=benchmarkName,value=df["value"].iloc[-1], unit=df["unit"].iloc[-1], date=df["date"].iloc[-1], commit=df["commit"].iloc[-1])
            LOGGER.debug( benchmarkValue)
            #update dict of results
            self.benchmarkValues[benchmarkValue.name] = benchmarkValue

    def getValues(self, names: list[str]):
        for name in names:
            self.getValue(name)

    #Tests a given benchmark result against what is found on the server
    def compare(self, localValue: BenchmarkValue, failOnMiss=False):
        #check if the server has a reference value
        referenceValue = self.getValue(localValue.name)

        if referenceValue is None:
            if failOnMiss:
                #TODO I should aggregate this too
                raise ValueError(f"Benchmark server does not contain a measurement for {localValue.name}")
            else:
                print(f"{localValue.name} not found on server. Passing anyway because 'failOnMiss=False'")
                return True
        else:

            #TODO add sanity checking once this info is on server
            #assert localValue.op == referenceValue.op #wont work, need to pass some inputs
            #assert localValue.tolerance == referenceValue.tolerance
            #assert localValue.unit == referenceValue.unit

            #select correct comparison operation and optionally apply tolerance
            #e.g. memory is 'local > ref => fail', throughput is 'local < (ref + tol) => fail'
            comp=localValue.op
            
            refVal=referenceValue.value
            localVal=cplocalValue.value
            tolerance=localValue.tolerance

            #This code is complicated because we need to account for
            # different comparsions >,<,<= etc
            # the possibility of tolerance 
            # so we apply the comparison and then if that doesnt pass we check for absolute tolerance difference
            # In this way, we dont need to encode which value to apply the tolerance too
            #I'd be open to hardcoding '>=' and '<=' tho

            percent_diff = 1 - (refVal / localVal )
            if comp(percent_diff, 0):
                LOGGER.debug("passed outright")
                passed=True
            #didnt pass straight away, try pass within tolerance
            elif tolerance != 0 and tolerance/100 >= abs(percent_diff):
                LOGGER.debug(f"Passed within {tolerance}% tolerance")
                passed=True
            else:
                LOGGER.debug("didnt pass")
                passed=False

            result_str=""
            if passed:
                result_str += f"PASS. Local  value for {localValue.name} is within tolerance of the reference value "
            else:
                result_str += f"FAIL. Local value for {localValue.name} has degraded compared to the reference value "
            #TODO replace with referenceValue.unit once i have that on the server
            result_str += f"({localVal:.2f}{localValue.unit} local vs {refVal:.2f}{localValue.unit} reference)"
            print(result_str)

            return passed

    #trys to update a metric on a remote server, with a given benchmarkValue
    def setValue(self, value:BenchmarkValue):

        #update remote server with new value
        header="testName,unit,date,commit,value"
        #append to existing file, but never apend header
        #TODO get it working for remote

        local_file = f"./{value.name}"
        
        #if file doesnt exist, write header
        if self.local:
            if not os.path.isfile(f"./server/{value.name}"):
                with open(local_file, "w") as f:
                    f.write(header + "\n")
                    f.write(value.to_csv() + "\n")
            else: 
                LOGGER.debug("existing file found... appending")
                with open(f"./server/{value.name}", "a") as f:
                    f.write(value.to_csv() + "\n")
        else:

            cp_cmd = [
                "scp",
                local_file,
                f"{self.set_host}:{self.set_remote_path}/{value.name}"
            ]
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

        #update dict of results
        self.benchmarkValues[value.name] = value

def get_git_revision_hash() -> str:
    return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()

def raise_error(x):
    raise ValueError(x)


#this functon will find and open the profiler logs from the most recent benchmarking training run
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

@pytest.mark.longtests
def test_benchmark_training_cycle(
    benchmark_config: tuple[DictConfig, str],
    get_test_archive: callable,
    update_data=True,
    throw_error=True,
) -> None:
    cfg, urls = benchmark_config
    for url in urls:
        get_test_archive(url)

    reset_peak_memory_stats()
    AnemoiProfiler(cfg).profile()

    # read memory and mlflow stats
    stats = memory_stats(device=0)
    peak_active_mem_mb = stats["active_bytes.all.peak"] / 1024 / 1024
    av_training_throughput = open_log_file("speed_profiler.csv")
    av_training_batch_time_s = open_log_file("time_profiler.csv")

    #create Benchmark value objects
    commit=get_git_revision_hash() #TODO check what happens if this cant find git hash
    yyyy_mm_dd=date.today().strftime('%Y-%m-%d')
    localBenchmarkResults=[]
    localBenchmarkResults.append(BenchmarkValue(name="avThroughputIterPerS", value=av_training_throughput, unit="iter/s", date=yyyy_mm_dd, commit=commit, op=operator.lt, tolerance=5))
    localBenchmarkResults.append(BenchmarkValue(name="avTimePerBatchS", value=av_training_batch_time_s, unit="s", date=yyyy_mm_dd, commit=commit, tolerance=5))
    localBenchmarkResults.append(BenchmarkValue(name="peakMemoryMB", value=peak_active_mem_mb, unit="MB", date=yyyy_mm_dd, commit=commit))

    #Get reference benchmark results
    benchmarkServer=BenchmarkServer()
    benchmarks = ["avThroughputIterPerS", "avTimePerBatchS", "peakMemoryMB"]
    benchmarkServer.getValues(benchmarks)

    print(f"Reference benchmark results:\n{benchmarkServer}")
    print("Local benchmark results:")
    for benchmarkValue in localBenchmarkResults:
        print(benchmarkValue)

    # either update the data on the server, or compare it against existing results
    if update_data:
        print(f"Updating metrics on server")
        for localBenchmarkValue in localBenchmarkResults:
            benchmarkServer.setValue(localBenchmarkValue)
    else:
        print(f"Comparing local benchmark results against reference values from the server")

        on_test_fail=print
        if throw_error:
            on_test_fail=raise_error

        failedTests=[]
        for localBenchmarkValue in localBenchmarkResults:
            passed = benchmarkServer.compare(localBenchmarkValue)
            if not passed:
                failedTests.append(localBenchmarkValue.name)

        if len(failedTests) > 0:
            on_test_fail(f"The following tests failed: {failedTests}")
