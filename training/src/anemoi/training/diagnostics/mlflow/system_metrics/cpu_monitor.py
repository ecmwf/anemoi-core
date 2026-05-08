# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import psutil
from mlflow.system_metrics.metrics.base_metrics_monitor import BaseMetricsMonitor


class CPUMonitor(BaseMetricsMonitor):
    """Class for monitoring CPU stats.

    Extends default CPUMonitor, to also measure total \
            memory and a different formula for calculating used memory.

    """

    def collect_metrics(self) -> None:
        # Get CPU metrics.
        cpu_percent = psutil.cpu_percent()
        self._metrics["cpu_utilization_percentage"].append(cpu_percent)

        system_memory = psutil.virtual_memory()
        # Change the formula for measuring CPU memory usage
        # By default Mlflow uses psutil.virtual_memory().used
        # Tests have shown that "used" underreports memory usage by as much as a factor of 2,
        #   "used" also misses increased memory usage from using a higher prefetch factor
        self._metrics["system_memory_usage_megabytes"].append(
            (system_memory.total - system_memory.available) / 1e6,
        )
        self._metrics["system_memory_usage_percentage"].append(system_memory.percent)

        # QOL: report the total system memory in raw numbers
        self._metrics["system_memory_total_megabytes"].append(system_memory.total / 1e6)

    def aggregate_metrics(self) -> dict[str, int]:
        return {k: round(sum(v) / len(v), 1) for k, v in self._metrics.items()}
