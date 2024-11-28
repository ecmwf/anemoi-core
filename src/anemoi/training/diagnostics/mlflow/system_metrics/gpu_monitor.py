# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import contextlib
import sys

from mlflow.system_metrics.metrics.base_metrics_monitor import BaseMetricsMonitor

with contextlib.suppress(ImportError):
    import pynvml
with contextlib.suppress(ImportError):
    from pyrsmi import rocml


class GreenGPUMonitor(BaseMetricsMonitor):
    """Class for monitoring Nvidia GPU stats.

    Requires pynvml to be installed.
    Extends default GPUMonitor, to also measure total \
            memory

    """

    def __init__(self):
        if "pynvml" not in sys.modules:
            # Only instantiate if `pynvml` is installed.
            import_error_msg = "`pynvml` is not installed, if you are running on an Nvidia GPU \
                and want to log GPU metrics please run `pip install pynvml`."
            raise ImportError(import_error_msg)
        try:
            # `nvmlInit()` will fail if no GPU is found.
            pynvml.nvmlInit()
        except pynvml.NVMLError as e:
            runtime_error_msg = "Failed to initalize Nvidia GPU monitor: "
            raise RuntimeError(runtime_error_msg) from e

        super().__init__()
        self.num_gpus = pynvml.nvmlDeviceGetCount()
        self.gpu_handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(self.num_gpus)]

    def collect_metrics(self) -> None:
        # Get GPU metrics.
        for i, handle in enumerate(self.gpu_handles):
            memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
            self._metrics[f"gpu_{i}_memory_usage_percentage"].append(
                round(memory.used / memory.total * 100, 1),
            )
            self._metrics[f"gpu_{i}_memory_usage_megabytes"].append(memory.used / 1e6)

            # Only record total device memory on GPU 0 to prevent spam
            # Unlikely for GPUs on the same node to have different total memory
            if i == 0:
                self._metrics["gpu_memory_total_megabytes"].append(memory.total / 1e6)

            # Monitor PCIe usage
            tx_kilobytes = pynvml.nvmlDeviceGetPcieThroughput(handle, pynvml.NVML_PCIE_UTIL_TX_BYTES)
            rx_kilobytes = pynvml.nvmlDeviceGetPcieThroughput(handle, pynvml.NVML_PCIE_UTIL_RX_BYTES)
            self._metrics[f"gpu_{i}_pcie_tx_megabytes"].append(tx_kilobytes / 1e3)
            self._metrics[f"gpu_{i}_pcie_rx_megabytes"].append(rx_kilobytes / 1e3)

            device_utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
            self._metrics[f"gpu_{i}_utilization_percentage"].append(device_utilization.gpu)

            power_milliwatts = pynvml.nvmlDeviceGetPowerUsage(handle)
            power_capacity_milliwatts = pynvml.nvmlDeviceGetEnforcedPowerLimit(handle)
            self._metrics[f"gpu_{i}_power_usage_watts"].append(power_milliwatts / 1000)
            self._metrics[f"gpu_{i}_power_usage_percentage"].append(
                (power_milliwatts / power_capacity_milliwatts) * 100,
            )

    def aggregate_metrics(self) -> dict[str, int]:
        return {k: round(sum(v) / len(v), 1) for k, v in self._metrics.items()}


class RedGPUMonitor(BaseMetricsMonitor):
    """Class for monitoring AMD GPU stats.

    Requires that pyrsmi is installed
    Logs utilization and memory usage.

    """

    def __init__(self):
        if "pyrsmi" not in sys.modules:
            import_error_msg = "`pyrsmi` is not installed, if you are running on an AMD GPU \
                and want to log GPU metrics please run `pip install pyrsmi`."
            # Only instantiate if `pyrsmi` is installed.
            raise ImportError(import_error_msg)
        try:
            # `rocml.smi_initialize()()` will fail if no GPU is found.
            rocml.smi_initialize()
        except RuntimeError as e:
            runtime_error_msg = "Failed to initalize AMD GPU monitor: "
            raise RuntimeError(runtime_error_msg) from e

        super().__init__()
        self.num_gpus = rocml.smi_get_device_count()

    def collect_metrics(self) -> None:
        # Get GPU metrics.
        for device in range(self.num_gpus):
            memory_used = rocml.smi_get_device_memory_used(device)
            memory_total = rocml.smi_get_device_memory_total(device)
            memory_busy = rocml.smi_get_device_memory_busy(device)
            self._metrics[f"gpu_{device}_memory_usage_percentage"].append(
                round(memory_used / memory_total * 100, 1),
            )
            self._metrics[f"gpu_{device}_memory_usage_megabytes"].append(memory_used / 1e6)

            self._metrics[f"gpu_{device}_memory_busy_percentage"].append(memory_busy)

            # Only record total device memory on GPU 0 to prevent spam
            # Unlikely for GPUs on the same node to have different total memory
            if device == 0:
                self._metrics["gpu_memory_total_megabytes"].append(memory_total / 1e6)

            utilization = rocml.smi_get_device_utilization(device)
            self._metrics[f"gpu_{device}_utilization_percentage"].append(utilization)

    def aggregate_metrics(self) -> dict[str, int]:
        return {k: round(sum(v) / len(v), 1) for k, v in self._metrics.items()}
