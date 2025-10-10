import json
import logging
import os
from collections import defaultdict

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import torch
from rich.console import Console

LOGGER = logging.getLogger(__name__)
console = Console(record=True, width=200)

filename = "/ec/res4/scratch/naco/aifs/outputs/raps/train/4N_16gpn_transformer_512c_o1280.jobname-619791/train-outputs/profiler/e5c4bd0471a2494f9183bdf493fafc7a/ac6-306.bullx_621045.None.1743008762865142923.pt.trace.json"


def find_first_trace_file(dirpath: str) -> str | None:
    """Search the given directory (non-recursively) for the first file ending with '.pt.trace.json'.

    Args:
        dirpath (str): Directory to search.

    Returns
    -------
        str | None: Full path to the first matching file, or None if not found.
    """
    for filename in sorted(os.listdir(dirpath)):
        if filename.endswith(".pt.trace.json"):
            return os.path.join(dirpath, filename)
    return None


def analyze_anemoi_durations(json_file_path):
    allowed_names = {"anemoi-encoder", "anemoi-decoder", "anemoi-processor"}
    # durations = dict()
    durations = defaultdict(list)

    with open(json_file_path) as f:
        data = json.load(f)

        # Some trace files have events inside 'traceEvents'
        events = data.get("traceEvents", data)

        for event in events:
            if event.get("cat") == "gpu_user_annotation" and event.get("name") in allowed_names:
                durations[event["name"]].append(event.get("dur", 0))

    # Compute total and average durations
    summary = {}
    for name, dur_list in durations.items():
        total = sum(dur_list)
        avg = total / len(dur_list) if dur_list else 0
        summary[name] = {"total_duration": total, "average_duration": avg, "count": len(dur_list)}

    return summary


def list_unique_kernel_names(json_file_path):
    unique_names = set()

    with open(json_file_path) as f:
        data = json.load(f)

        # Handle traceEvents wrapping
        events = data.get("traceEvents", data)

        for event in events:
            if event.get("cat") == "kernel":
                name = event.get("name")
                if name:
                    unique_names.add(name)

    console.print("Unique kernel event names:")
    for name in sorted(unique_names):
        console.print(f"- {name}")


def extract_unique_kernel_function_names(json_file_path):
    unique_funcs = set()

    with open(json_file_path) as f:
        data = json.load(f)
        events = data.get("traceEvents", data)

        for event in events:
            if event.get("cat") == "kernel" or event.get("cat") == "gpu_memcpy" or event.get("cat") == "gpu_memset":
                name = event.get("name", "")

                # Step 1: Remove leading "void " if present
                name = name.removeprefix("void ")

                # Step 2: Truncate at the first "<", if present
                name = name.split("<")[0]

                unique_funcs.add(name)

    # print("Unique kernel function names:\n")
    # for func in sorted(unique_funcs):
    #    print(f"- {func}")
    return unique_funcs


def sum_kernel_durations(kernel_durations):
    # Sort by total duration (descending)
    sorted_durations = sorted(kernel_durations.items(), key=lambda x: x[1], reverse=True)

    # print("Total durations per kernel (sorted by duration):")
    # for kernel_name, total_dur in sorted_durations:
    #    print(f"- {kernel_name}: {total_dur/1000:.3f}s")

    return sorted_durations


def plot_kernel_durations_pie(sorted_durations, top_n=10):
    # Split top N and 'Other'
    top_kernels = sorted_durations[:top_n]
    other_total = sum(dur for _, dur in sorted_durations[top_n:])

    labels = [name for name, _ in top_kernels]
    durations = [dur for _, dur in top_kernels]

    if other_total > 0:
        labels.append("Other")
        durations.append(other_total)

    # Generate distinct colors using colormap
    cmap = cm.get_cmap("tab20", len(labels))
    colors = [cmap(i) for i in range(len(labels))]

    # Generate hatching for special kernels
    hatches = []
    for name in labels:
        if "nccl" in name:
            hatches.append("////")  # Diagonal lines for comms
        elif "Memcpy" in name:
            hatches.append("....")  # Dots for memcpys
        else:
            hatches.append(None)

    # Plot
    explode = [0.05] * top_n + [0]

    # Create pie chart
    fig, ax = plt.subplots(figsize=(10, 8))
    # wedges, texts, autotexts = ax.pie(
    wedges, texts = ax.pie(
        durations,
        labels=None,  # disable inline labels
        # autopct='%1.1f%%',
        startangle=140,
        colors=colors,
        # explode=explode,
        wedgeprops=dict(width=0.4, edgecolor="black", linewidth=2),
        pctdistance=0.75,
    )

    # Apply hatches
    for wedge, hatch in zip(wedges, hatches, strict=False):
        if hatch:
            wedge.set_hatch(hatch)

    # Add legend outside the plot
    ax.legend(wedges, labels, title="Kernel Names", loc="center left", bbox_to_anchor=(1, 0.5), fontsize="small")

    ax.set_title(f"Kernel Duration Distribution (Top {top_n} Shown)", fontsize=14)
    plt.tight_layout()
    plt.savefig("kernels.png")
    # plt.show()


def classify_kernel(name: str) -> str:
    name_lower = name.lower()
    if "memcpy" in name_lower or "memset" in name_lower:
        return "memory"
    if "nccl" in name_lower:
        return "comms"
    # elif 'cublas' in name_lower or 'cutlass' in name_lower or 'gemm' in name_lower or 'kernel' in name_lower or 'flash' in name_lower:
    #    return 'compute'
    return "compute"  # Default fallback


def print_kernel_table(
    data: list[tuple[str, float]],
    kernel_counts,
    kernel_weighted_occupancies,
    top_n: int = 10,
    num_iterations=20,
):
    total_duration_us = sum(duration for _, duration in data) / num_iterations

    # Sort by duration descending
    data_sorted = sorted(data, key=lambda x: x[1], reverse=True)

    top_kernels = data_sorted[:top_n]
    remaining = data_sorted[top_n:]

    # Prepare table rows
    rows = []

    # for (name, duration), count, occupancy in zip(top_kernels, kernel_counts, kernel_weighted_occupancies):
    total_count = 0
    for name, duration_us in top_kernels:
        count = kernel_counts[name] / num_iterations
        total_count += count
        occupancy = kernel_weighted_occupancies[name] / duration_us * 100
        category = classify_kernel(name)
        percent = ((duration_us / num_iterations) / total_duration_us) * 100
        rows.append((name, category, duration_us / 1000000 / num_iterations, percent, count, occupancy))

    # Aggregate "other"
    if remaining:
        other_duration_us = sum(d for _, d in remaining) / num_iterations
        other_percent = (other_duration_us / total_duration_us) * 100
        other_count = sum(kernel_counts[name] for name, _ in remaining) / num_iterations
        total_count += other_count
        other_occupancy = 0.0
        rows.append(
            ("other", "-", other_duration_us / 1000000 / num_iterations, other_percent, other_count, other_occupancy),
        )

    # Add total row
    rows.append(
        ("total (kernels on different streams can overlap)", "-", total_duration_us / 1000000, 100.0, total_count, 0.0),
    )

    # Print formatted table
    console.print(
        f"{'Kernel':60} {'Category':10} {'Duration (s)':>15} {'% Time':>10} {'Count':>10} {'% Occupancy':>10}",
    )
    console.print("-" * 100)
    for name, category, duration, percent, count, occupancy in rows:
        console.print(f"{name[:58]:60} {category:10} {duration:15.2f} {percent:10.2f} {count:10} {occupancy:10.2f}")


def count_iterations(json_file_path):
    with open(json_file_path) as f:
        data = json.load(f)
        events = data.get("traceEvents", data)

        for event in events:
            iteration_count = 0
            if event.get("cat") == "python_function" and ("transfer_batch_to_device" in event.get("name")):
                iteration_count += 1
    return iteration_count


def compute_av_time_per_iter_and_dl_stalls(iteration_durations_us, dataloading_stall_durations_us):
    # print(f"{iteration_durations_us=}, {dataloading_stall_durations_us=}")
    dataloading_stall_durations_s = np.array(dataloading_stall_durations_us) / 1000000
    iteration_durations_s = np.array(iteration_durations_us) / 1000000
    # iteration_durations = [iteration_start_times[i+1] - iteration_start_times[i] for i in range(len(iteration_start_times) - 1)]
    dataloading_stall_percentages = dataloading_stall_durations_s / iteration_durations_s * 100
    # print(f"{iteration_durations_s=} {dataloading_stall_durations_s=} {dataloading_stall_percentages=}%")
    # using median here to minimise the impact of long tails in the distribution
    av_iteration_duration_s = np.median(iteration_durations_s)
    av_throughput = 1 / av_iteration_duration_s
    av_dataloading_stall_duration_s = np.median(dataloading_stall_durations_s)
    av_dataloading_stall_percentage = np.median(dataloading_stall_percentages)
    console.print(
        f"Each training iteration took an average of {av_iteration_duration_s:.2f}s ({av_throughput:.2f} iterations per second)",
    )
    console.print(
        f"An average of {av_dataloading_stall_duration_s:.2f}s ({av_dataloading_stall_percentage:.2f}%) of each iteration was spent idling while loading data",
    )
    if av_dataloading_stall_percentage > 5.0:
        console.print(
            "Warning! Dataloading stall times are high. You can try increase the number of dataloader workers. If you are limited by CPU memory, you can try decrease prefetch factor to 1 to further increase the number of workers",
        )
    return iteration_durations_s, dataloading_stall_durations_s, dataloading_stall_percentages


def analyse_HtoD_memcpy(batch_sizes_GB, batch_transfer_bw_GBs, batch_transfer_durations_us):
    #    "ph": "X", "cat": "gpu_memcpy", "name": "Memcpy HtoD (Pinned -> Device)", "pid": 0, "tid": 7, "ts": 7376383457086.139, "dur": 19558.546, "args": {  "External id": 77569,      "device": 0, "context": 1,      "stream": 7, "correlation": 436228,      "bytes": 499925760, "memory bandwidth (GB/s)": 25.560476734824768}

    batch_transfer_durations_us = np.array(batch_transfer_durations_us)

    # print(f"{batch_sizes_GB=} {batch_transfer_durations_us/1000000=} {batch_transfer_bw_GBs=}")
    av_batch_size_GB = np.mean(batch_sizes_GB)
    av_batch_transfer_bw_GBs = np.mean(batch_transfer_bw_GBs)
    av_batch_transfer_durations_s = np.mean(batch_transfer_durations_us) / 1000000
    console.print(
        f"{av_batch_size_GB=:.2f}GB, {av_batch_transfer_durations_s=:.2f}s, ({av_batch_transfer_bw_GBs=:.2f}GB/s)",
    )
    return av_batch_size_GB, av_batch_transfer_bw_GBs, av_batch_transfer_durations_s


def parse_json_trace_file(json_file_path):
    """This function iterates once over a json trace file and returns all the information we will later analyse"""
    with open(json_file_path) as f:
        data = json.load(f)
        events = data.get("traceEvents", data)

    # HtoD memcpy analysis
    batch_sizes_GB = []
    batch_transfer_bw_GBs = []
    batch_transfer_durations_us = []

    # Av. iter time and dataloading stall analysis
    dataloading_stall_durations_us = []
    iteration_durations_us = []

    # kernel analysis
    # unique_kernel_names = set()
    kernel_durations = defaultdict(float)
    kernel_counts = defaultdict(int)
    kernel_weighted_occupancies = defaultdict(float)

    main_stream = 7  # assumption
    gpu_idle_time = 0
    prev_kernel_end_time = 0

    iteration_count = 0

    for event in events:
        if event.get("cat") == "kernel" or event.get("cat") == "gpu_memcpy" or event.get("cat") == "gpu_memset":
            # get unique kernel names
            name = event.get("name", "")
            # Step 1: Remove leading "void " if present
            name = name.removeprefix("void ")
            # Step 2: Truncate at the first "<", if present
            name = name.split("<")[0]
            # unique_kernel_names.add(name)

            # get durations for each kernel
            kernel_duration = event.get("dur", 0)
            kernel_occupancy_pct = event.get("args").get("est. achieved occupancy %", 0) / 100
            kernel_durations[name] += kernel_duration
            kernel_counts[name] += 1
            kernel_weighted_occupancies[name] += kernel_occupancy_pct * kernel_duration

        # analyse iter time and dataloading stall time
        if event.get("cat") == "user_annotation" and ("train_dataloader_next" in event.get("name")):
            dataloading_stall_durations_us.append(event.get("dur"))
        if event.get("cat") == "user_annotation" and ("run_training_batch" in event.get("name")):
            iteration_durations_us.append(event.get("dur"))

        # if event.get("cat") == "python_function" and ("transfer_batch_to_device" in event.get("name")):
        if event.get("cat") == "user_annotation" and ("transfer_batch_to_device" in event.get("name")):
            iteration_count += 1

        # analyse HtoD memcpy
        if event.get("cat") == "gpu_memcpy" and ("Memcpy HtoD (Pinned" in event.get("name")):
            batch_transfer_durations_us.append(event.get("dur"))
            batch_sizes_GB.append(
                event.get("args")["bytes"] / 1000 / 1000 / 1000,
            )  # use 1000 not 1024 as it matches the numbers reported by the tracer
            batch_transfer_bw_GBs.append(event.get("args")["memory bandwidth (GB/s)"])

        # should i include memcpy here?
        if event.get("cat") == "kernel" and event.get("args").get("stream") == main_stream:
            kernel_end_time = event.get("ts") + event.get("dur")
            kernel_start_time = event.get("ts")
            if prev_kernel_end_time != 0:
                diff = kernel_start_time - prev_kernel_end_time
                if diff > 0:
                    gpu_idle_time += diff
            prev_kernel_end_time = kernel_end_time

    print(f"gpu_idle_time = {gpu_idle_time/1000000}s")
    print(f"gpu_idle_time per iteration = {gpu_idle_time/1000000/iteration_count}s")
    return (
        batch_sizes_GB,
        batch_transfer_bw_GBs,
        batch_transfer_durations_us,
        dataloading_stall_durations_us,
        iteration_durations_us,
        kernel_durations,
        kernel_counts,
        kernel_weighted_occupancies,
        iteration_count,
    )


def analyse_gpu_memory_usage():
    max_available_memory_GB = (
        torch.cuda.get_device_properties().total_memory / 1024 / 1024 / 1024
    )  # TODO should guard this incase the syntax is different on AMD
    max_reserved_memory_GB = torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024
    max_allocated_memory_GB = torch.cuda.max_memory_allocated() / 1024 / 1024 / 1024
    console.print(f"{max_available_memory_GB=:.2f}, {max_reserved_memory_GB=:.2f}, {max_allocated_memory_GB=:.2f}")

    max_reserved_but_unused_memory_GB = max_reserved_memory_GB - max_allocated_memory_GB
    if max_reserved_but_unused_memory_GB > 2:
        console.print(
            f"Warning! you have {max_reserved_but_unused_memory_GB}GB of memory reserved by pytorch but not actively allocated. This memory fragmentation can result in avoidable Out-Of-Memory errors. You can try 'export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True' to reduce this memory fragmentation.",
        )

    max_allocated_memory_percentage = max_allocated_memory_GB / max_available_memory_GB * 100
    console.print(f"Peak (allocated) memory usage was {max_allocated_memory_percentage:.2f}% of total device memory")
    if max_allocated_memory_percentage < 50.0:
        console.print(
            "Warning! Your peak device memory usage is low. You could try increasing the batch size or reducing the number of GPUs",
        )


# analyze_anemoi_durations(filename)
# unique_funcs=extract_unique_kernel_function_names(filename)
# durations =sum_kernel_durations(filename, unique_funcs)
# print(durations)


def analyse_trace(dirpath):
    filename = find_first_trace_file(dirpath)
    console.print(f"Analysing {filename}")
    (
        batch_sizes_GB,
        batch_transfer_bw_GBs,
        batch_transfer_durations_us,
        dataloading_stall_durations_us,
        iteration_durations_us,
        kernel_durations,
        kernel_counts,
        kernel_weighted_occupancies,
        iteration_count,
    ) = parse_json_trace_file(filename)
    kernel_durations = sum_kernel_durations(kernel_durations)
    print_kernel_table(
        kernel_durations,
        kernel_counts,
        kernel_weighted_occupancies,
        top_n=10,
        num_iterations=iteration_count,
    )
    console.print("\n")
    compute_av_time_per_iter_and_dl_stalls(iteration_durations_us, dataloading_stall_durations_us)
    console.print("\n")
    analyse_HtoD_memcpy(batch_sizes_GB, batch_transfer_bw_GBs, batch_transfer_durations_us)
    console.print("\n")
    analyse_gpu_memory_usage()


if __name__ == "__main__":
    # filename="/ec/res4/scratch/naco/aifs/outputs/raps/train/4N_16gpn_transformer_512c_o1280.jobname-619791/train-outputs/profiler/e5c4bd0471a2494f9183bdf493fafc7a/ac6-306.bullx_621045.None.1743008762865142923.pt.trace.json" # old o2560 512c with multi gpu
    # filename="/ec/res4/hpcperm/naco/aifs/tests/profiler_update/outputs/profiler/c78cfd5d829340549407acb1e5b36426/ac6-302.bullx_3183374.None.1751544480627283549.pt.trace.json" #20 steps o96 1 gpu
    filename = "/ec/res4/hpcperm/naco/aifs/tests/profiler_update/outputs/profiler/9a57351d759c428b92964af00c75b767/ac6-301.bullx_3890366.None.1751551147735771561.pt.trace.json"  # 20 steps o96 4 gpus
    (
        batch_sizes_GB,
        batch_transfer_bw_GBs,
        batch_transfer_durations_us,
        dataloading_stall_durations_us,
        iteration_durations_us,
        kernel_durations,
        kernel_counts,
        kernel_weighted_occupancies,
        iteration_count,
    ) = parse_json_trace_file(filename)
    # print(kernel_durations)
    kernel_durations = sum_kernel_durations(kernel_durations)
    print_kernel_table(
        kernel_durations,
        kernel_counts,
        kernel_weighted_occupancies,
        top_n=10,
        num_iterations=iteration_count,
    )
    # print(durations)
    # plot_kernel_durations_pie(durations, top_n=20)
    print("\n")
    compute_av_time_per_iter_and_dl_stalls(iteration_durations_us, dataloading_stall_durations_us)
    print("\n")
    analyse_HtoD_memcpy(batch_sizes_GB, batch_transfer_bw_GBs, batch_transfer_durations_us)
    print("\n")
    x = torch.empty(1024, 1024, 1024, device="cuda", dtype=torch.float32)
    analyse_gpu_memory_usage()
    # print(torch.cuda.get_device_properties())
