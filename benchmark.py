#!/usr/bin/env python3

import argparse
import json
import os
import random
import re
import struct
import subprocess
import sys
import urllib.request
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml
from dotenv import load_dotenv
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent.absolute()
BENCHMARKS_DIR = PROJECT_ROOT / "benchmarks"
RESULTS_DIR = PROJECT_ROOT / "results"
DEPS_DIR = PROJECT_ROOT / "deps"
MACHINE_DIR = DEPS_DIR / "machine"
IMAGES_DIR = PROJECT_ROOT / "images"

DEFAULT_LINUX_IMAGE_URL = "https://github.com/cartesi/machine-linux-image/releases/download/v0.20.0/linux-6.5.13-ctsi-1-v0.20.0.bin"
DEFAULT_ROOTFS_IMAGE_URL = "https://github.com/cartesi/machine-rootfs-image/releases/download/v0.20.0-test1/rootfs-ubuntu.ext2"

load_dotenv(PROJECT_ROOT / ".env")


def load_config() -> dict:
    config_path = PROJECT_ROOT / "config.yaml"
    if config_path.exists():
        with open(config_path) as f:
            return yaml.safe_load(f) or {}
    return {}


def get_image_url(image_type: str) -> str:
    config = load_config()

    if image_type == "linux":
        return (
            os.getenv("LINUX_IMAGE_URL")
            or config.get("linux_image_url")
            or DEFAULT_LINUX_IMAGE_URL
        )
    elif image_type == "rootfs":
        return (
            os.getenv("ROOTFS_IMAGE_URL")
            or config.get("rootfs_image_url")
            or DEFAULT_ROOTFS_IMAGE_URL
        )
    return ""


def get_hardware_profile(hardware_key: str) -> dict:
    """Get hardware profile from config."""
    config = load_config()
    profiles = config.get("hardware_profiles", {})

    if hardware_key not in profiles:
        raise ValueError(f"Unknown hardware profile: {hardware_key}. Available: {list(profiles.keys())}")

    return profiles[hardware_key]


def estimate_proving_time(cycles: int, throughput_khz: float) -> float:
    """Estimate proving time in seconds based on cycles and throughput."""
    return cycles / (throughput_khz * 1000)


def download_file(url: str, dest: Path) -> bool:
    print(f"Downloading {url}...")
    print(f"  -> {dest}")

    dest.parent.mkdir(parents=True, exist_ok=True)

    try:
        urllib.request.urlretrieve(url, dest)
        print(f"  Download complete.")
        return True
    except Exception as e:
        print(f"  Download failed: {e}")
        return False


def ensure_images() -> tuple[Path | None, Path | None]:
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    linux_url = get_image_url("linux")
    rootfs_url = get_image_url("rootfs")

    linux_filename = linux_url.split("/")[-1]
    rootfs_filename = rootfs_url.split("/")[-1]

    linux_path = IMAGES_DIR / linux_filename
    rootfs_path = IMAGES_DIR / rootfs_filename

    if not linux_path.exists():
        if not download_file(linux_url, linux_path):
            return None, None

    if not rootfs_path.exists():
        if not download_file(rootfs_url, rootfs_path):
            return None, None

    return linux_path, rootfs_path


def load_benchmark(yaml_path: Path) -> dict:
    with open(yaml_path) as f:
        config = yaml.safe_load(f)
    return config


def list_benchmarks() -> list[Path]:
    return sorted(BENCHMARKS_DIR.glob("*.yaml"))


def build_machine_emulator() -> bool:
    print("Building machine emulator...")
    try:
        subprocess.run(
            ["make"],
            cwd=MACHINE_DIR,
            check=True,
            capture_output=True,
        )
        print("Machine emulator built successfully.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Failed to build machine emulator: {e}")
        print(e.stderr.decode() if e.stderr else "")
        return False


def build_risc0_prover() -> bool:
    print("Building RISC0 prover...")
    risc0_dir = MACHINE_DIR / "risc0"
    try:
        subprocess.run(
            ["make"],
            cwd=risc0_dir,
            check=True,
            capture_output=True,
        )
        print("RISC0 prover built successfully.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Failed to build RISC0 prover: {e}")
        print(e.stderr.decode() if e.stderr else "")
        return False


def check_builds() -> bool:
    cartesi_so = MACHINE_DIR / "src" / "cartesi.so"
    risc0_cli = MACHINE_DIR / "risc0" / "rust" / "target" / "debug" / "cartesi-risc0-cli"

    needs_build = False

    if not cartesi_so.exists():
        print(f"Cartesi Lua module not found at {cartesi_so}")
        needs_build = True
        if not build_machine_emulator():
            return False

    if not risc0_cli.exists():
        print(f"RISC0 CLI not found at {risc0_cli}")
        needs_build = True
        if not build_risc0_prover():
            return False

    if not needs_build:
        print("All binaries found.")

    return True


def run_cartesi_machine(
    command: str,
    step_size: int,
    max_mcycle: int,
    log_path: Path,
    rootfs_path: Path,
    linux_path: Path,
) -> tuple[str | None, str | None]:
    machine_lua = MACHINE_DIR / "src" / "cartesi-machine.lua"

    cmd = [
        str(machine_lua),
        f"--ram-image={linux_path}",
        f"--flash-drive=label:root,data_filename:{rootfs_path}",
        "--hash-tree=hash_function:sha256",
        f"--max-mcycle={max_mcycle + step_size + 1000}",
        f"--log-step={step_size},{log_path}",
        "--",
        command,
    ]

    print(f"  Running Cartesi machine with step_size={step_size}...")

    env = os.environ.copy()
    env["LUA_CPATH_5_4"] = f"{MACHINE_DIR / 'src'}/?.so;;"
    env["LUA_PATH_5_4"] = f"{MACHINE_DIR / 'src'}/?.lua;;"

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,
            env=env,
        )
    except subprocess.TimeoutExpired:
        print("  Cartesi machine timed out.")
        return None, None

    stderr_output = result.stderr

    hashes = []
    for line in stderr_output.split("\n"):
        match = re.match(r"^(\d+): ([a-f0-9]{64})$", line)
        if match:
            cycle = int(match.group(1))
            hash_value = match.group(2)
            hashes.append((cycle, hash_value))

    if len(hashes) < 2:
        print(f"  Failed to extract hashes from machine output.")
        print(f"  stderr: {stderr_output[:500]}...")
        return None, None

    start_cycle, start_hash = hashes[0]
    end_cycle, end_hash = hashes[1]

    return start_hash, end_hash


def run_risc0_prover(
    start_hash: str,
    end_hash: str,
    log_path: Path,
    step_size: int,
) -> dict | None:
    risc0_cli = MACHINE_DIR / "risc0" / "rust" / "target" / "debug" / "cartesi-risc0-cli"
    receipt_path = log_path.parent / "receipt.bin"

    env = os.environ.copy()
    env["RISC0_DEV_MODE"] = "1"
    env["RUST_LOG"] = "info"
    env["RISC0_INFO"] = "1"

    cmd = [
        str(risc0_cli),
        "prove",
        start_hash,
        str(log_path),
        str(step_size),
        end_hash,
        str(receipt_path),
    ]

    print(f"  Running RISC0 prover...")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=1800,
            env=env,
        )
    except subprocess.TimeoutExpired:
        print("  RISC0 prover timed out.")
        return None

    output = result.stdout + result.stderr

    if result.returncode != 0:
        print(f"  Prover failed with exit code {result.returncode}")
        print(f"  Output: {output[:1000]}")

    metrics = parse_prover_output(output)
    metrics["step"] = step_size

    return metrics


def parse_prover_output(output: str) -> dict:
    metrics = {
        "execution_time": None,
        "number_of_segments": None,
        "total_cycles": None,
        "user_cycles": None,
    }

    patterns = {
        "execution_time": r"execution time:\s*([\d.]+\w*)",
        "number_of_segments": r"number of segments:\s*(\d+)",
        "total_cycles": r"(\d+)\s+total cycles",
        "user_cycles": r"(\d+)\s+user cycles",
    }

    for key, pattern in patterns.items():
        match = re.search(pattern, output, re.IGNORECASE)
        if match:
            value = match.group(1)
            if key in ["number_of_segments", "total_cycles", "user_cycles"]:
                metrics[key] = int(value)
            else:
                metrics[key] = value

    return metrics


def read_page_count(log_path: Path) -> int | None:
    try:
        with open(log_path, "rb") as f:
            f.read(8)
            data = f.read(8)
            if len(data) < 8:
                return None
            page_count = struct.unpack("<Q", data)[0]
            return page_count
    except Exception:
        return None


def generate_plots(results: list[dict], benchmark_name: str, output_dir: Path, hardware_profiles: list[dict] | None = None):
    if not results:
        print("No results to plot.")
        return

    step_lengths = [r["step"] for r in results]

    execution_times = []
    for r in results:
        et = r.get("execution_time")
        if et and isinstance(et, str):
            match = re.match(r"([\d.]+)", et)
            if match:
                execution_times.append(float(match.group(1)))
            else:
                execution_times.append(0)
        else:
            execution_times.append(0)

    num_segments = [r.get("number_of_segments", 0) or 0 for r in results]
    total_cycles = [r.get("total_cycles", 0) or 0 for r in results]
    user_cycles = [r.get("user_cycles", 0) or 0 for r in results]
    page_counts = [r.get("page_count", 0) or 0 for r in results]

    fig, axs = plt.subplots(3, 2, figsize=(12, 14))
    plt.suptitle(f"Benchmark: {benchmark_name}")

    axs[0, 0].plot(step_lengths, execution_times, color="blue", marker="o")
    axs[0, 0].set_title("Execution Time")
    axs[0, 0].set_xlabel("Step Length (cycles)")
    axs[0, 0].set_ylabel("Execution Time (s)")
    axs[0, 0].grid(True)

    axs[0, 1].plot(step_lengths, num_segments, color="red", marker="o")
    axs[0, 1].set_title("Number of Segments")
    axs[0, 1].set_xlabel("Step Length (cycles)")
    axs[0, 1].set_ylabel("Segments")
    axs[0, 1].grid(True)

    axs[1, 0].plot(step_lengths, total_cycles, color="green", marker="o")
    axs[1, 0].set_title("Total Cycles")
    axs[1, 0].set_xlabel("Step Length (cycles)")
    axs[1, 0].set_ylabel("Cycles")
    axs[1, 0].grid(True)

    axs[1, 1].plot(step_lengths, user_cycles, color="orange", marker="o")
    axs[1, 1].set_title("User Cycles")
    axs[1, 1].set_xlabel("Step Length (cycles)")
    axs[1, 1].set_ylabel("Cycles")
    axs[1, 1].grid(True)

    axs[2, 0].plot(step_lengths, page_counts, color="purple", marker="o")
    axs[2, 0].set_title("Page Count")
    axs[2, 0].set_xlabel("Step Length (cycles)")
    axs[2, 0].set_ylabel("Pages Touched")
    axs[2, 0].grid(True)

    # Plot estimated proving times for each hardware profile
    if hardware_profiles:
        colors = ["crimson", "dodgerblue", "forestgreen", "darkorange", "purple"]
        markers = ["o", "s", "^", "D", "v"]
        for i, profile in enumerate(hardware_profiles):
            throughput = profile["throughput_khz"]
            estimated_times = [estimate_proving_time(c, throughput) for c in total_cycles]
            color = colors[i % len(colors)]
            marker = markers[i % len(markers)]
            axs[2, 1].plot(step_lengths, estimated_times, color=color, marker=marker, label=profile["name"])
        axs[2, 1].set_title("Est. Proving Time")
        axs[2, 1].set_xlabel("Step Length (cycles)")
        axs[2, 1].set_ylabel("Time (seconds)")
        axs[2, 1].legend(loc="upper left", fontsize=8)
        axs[2, 1].grid(True)
    else:
        axs[2, 1].axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.97])

    plot_path = output_dir / "plots.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Plots saved to {plot_path}")


def get_total_cycles(
    command: str,
    rootfs_path: Path,
    linux_path: Path,
    max_mcycle: int | None = None,
) -> int | None:
    machine_lua = MACHINE_DIR / "src" / "cartesi-machine.lua"

    cmd = [
        str(machine_lua),
        f"--ram-image={linux_path}",
        f"--flash-drive=label:root,data_filename:{rootfs_path}",
    ]
    if max_mcycle:
        cmd.append(f"--max-mcycle={max_mcycle}")
    cmd.extend(["--", command])

    print(f"Getting total cycles for command: {command}")
    print(f"  cmd: {' '.join(cmd)}")

    env = os.environ.copy()
    env["LUA_CPATH_5_4"] = f"{MACHINE_DIR / 'src'}/?.so;;"
    env["LUA_PATH_5_4"] = f"{MACHINE_DIR / 'src'}/?.lua;;"

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=1200,
            env=env,
        )
    except subprocess.TimeoutExpired:
        print("  Machine timed out while getting total cycles.")
        return None

    output = result.stderr + result.stdout
    for line in output.split("\n"):
        match = re.match(r"^Cycles:\s*(\d+)", line)
        if match:
            cycles = int(match.group(1))
            print(f"  Total cycles: {cycles}")
            return cycles

    print("  Failed to extract total cycles from output.")
    print(f"  Return code: {result.returncode}")
    print(f"  stderr (last 500 chars): {result.stderr[-500:]}")
    return None


def measure_boot_cycles(
    rootfs_path: Path,
    linux_path: Path,
) -> int | None:
    print("Measuring boot cycles (running 'ls' payload)...")
    cycles = get_total_cycles(
        command="ls",
        rootfs_path=rootfs_path,
        linux_path=linux_path,
        max_mcycle=100000000,
    )
    if cycles:
        boot_cycles = int(cycles * 1.1)
        print(f"  Boot cycles (with 10% buffer): {boot_cycles}")
        return boot_cycles
    return None


def run_window_sample(
    command: str,
    start_cycle: int,
    window_size: int,
    log_path: Path,
    rootfs_path: Path,
    linux_path: Path,
) -> tuple[int | None, str | None, str | None]:
    machine_lua = MACHINE_DIR / "src" / "cartesi-machine.lua"

    cmd = [
        str(machine_lua),
        f"--ram-image={linux_path}",
        f"--flash-drive=label:root,data_filename:{rootfs_path}",
        "--hash-tree=hash_function:sha256",
        f"--max-mcycle={start_cycle + window_size + 1000}",
        f"--log-step={window_size},{log_path}",
        "--",
        command,
    ]

    env = os.environ.copy()
    env["LUA_CPATH_5_4"] = f"{MACHINE_DIR / 'src'}/?.so;;"
    env["LUA_PATH_5_4"] = f"{MACHINE_DIR / 'src'}/?.lua;;"

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,
            env=env,
        )
    except subprocess.TimeoutExpired:
        return None, None, None

    hashes = []
    for line in result.stderr.split("\n"):
        match = re.match(r"^(\d+): ([a-f0-9]{64})$", line)
        if match:
            hashes.append(match.group(2))

    if len(hashes) < 2:
        return None, None, None

    page_count = read_page_count(log_path)
    return page_count, hashes[0], hashes[1]


def store_metrics_jsonl(metrics: dict, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "a") as f:
        f.write(json.dumps(metrics) + "\n")


def generate_histograms(
    data_path: Path,
    benchmark_name: str,
    window_size: int,
    output_dir: Path,
    hardware_profiles: list[dict] | None = None,
):
    if not data_path.exists():
        print(f"No data file found at {data_path}")
        return

    records = []
    with open(data_path) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    if not records:
        print("No records to plot.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    fields = [
        ("total_cycles", "Total Cycles", "steelblue"),
        ("page_count", "Page Count", "purple"),
        ("number_of_segments", "Number of Segments", "darkgreen"),
    ]

    # Add user_cycles if no hardware profiles
    if not hardware_profiles:
        fields.append(("user_cycles", "User Cycles", "darkorange"))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, (field, title, color) in enumerate(fields):
        ax = axes[idx]
        values = [r.get(field) for r in records if r.get(field) is not None]

        if not values:
            ax.text(0.5, 0.5, f"No data for {title}", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(title)
            continue

        ax.hist(values, bins=20, color=color, edgecolor="black", alpha=0.7)

        mean_val = np.mean(values)
        median_val = np.median(values)
        p10 = np.percentile(values, 10)
        p90 = np.percentile(values, 90)

        ax.axvline(mean_val, color="red", linestyle="--", linewidth=2, label=f"Mean: {mean_val:.0f}")
        ax.axvline(median_val, color="green", linestyle="--", linewidth=2, label=f"Median: {median_val:.0f}")
        ax.axvline(p10, color="purple", linestyle=":", linewidth=1.5, label=f"P10: {p10:.0f}")
        ax.axvline(p90, color="brown", linestyle=":", linewidth=1.5, label=f"P90: {p90:.0f}")

        ax.set_title(title)
        ax.set_xlabel(title)
        ax.set_ylabel("Frequency")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)

    # 4th subplot: estimated proving times for all hardware profiles
    if hardware_profiles:
        ax = axes[3]
        total_cycles = [r.get("total_cycles") for r in records if r.get("total_cycles")]
        colors = ["crimson", "dodgerblue", "forestgreen", "darkorange", "purple"]

        for i, profile in enumerate(hardware_profiles):
            throughput = profile["throughput_khz"]
            estimated_times = [estimate_proving_time(c, throughput) for c in total_cycles]
            color = colors[i % len(colors)]
            ax.hist(estimated_times, bins=20, color=color, edgecolor="black", alpha=0.5, label=profile["name"])

        ax.set_title("Est. Proving Time")
        ax.set_xlabel("Time (seconds)")
        ax.set_ylabel("Frequency")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"{benchmark_name} - Monte Carlo Distribution (window={window_size}, n={len(records)})", fontsize=14)
    plt.tight_layout()

    plot_path = output_dir / f"{benchmark_name}_{window_size}_histograms.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Histograms saved to {plot_path}")


def run_monte_carlo(
    config: dict,
    linux_path: Path,
    rootfs_path: Path,
    num_samples: int = 100,
    window_size: int = 100000,
    boot_cycles: int = 0,
    hardware_profiles: list[dict] | None = None,
) -> list[dict]:
    name = config["name"]
    command = config["command"]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = RESULTS_DIR / f"{name}_monte_carlo_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    log_path = output_dir / "log.bin"
    data_path = output_dir / f"{name}_{window_size}.jsonl"

    total_cycles = get_total_cycles(
        command=command,
        rootfs_path=rootfs_path,
        linux_path=linux_path,
    )

    if not total_cycles:
        print("ERROR: Could not determine total cycles for benchmark.")
        return []

    payload_cycles = total_cycles - boot_cycles
    if payload_cycles < window_size:
        print(f"ERROR: Payload too short for Monte Carlo sampling.")
        print(f"  Total cycles: {total_cycles}")
        print(f"  Boot cycles: {boot_cycles}")
        print(f"  Payload cycles: {payload_cycles}")
        print(f"  Window size: {window_size}")
        return []

    print(f"\n{'='*60}")
    print(f"Monte Carlo Experiment: {name}")
    print(f"Command: {command}")
    print(f"Total cycles: {total_cycles}")
    print(f"Boot cycles: {boot_cycles}")
    print(f"Payload cycles: {payload_cycles}")
    print(f"Window size: {window_size}")
    print(f"Samples: {num_samples}")
    print(f"{'='*60}\n")

    results = []
    successful = 0

    for i in tqdm(range(num_samples), desc=f"Sampling {name}"):
        if log_path.exists():
            log_path.unlink()

        max_start = max(boot_cycles, total_cycles - window_size - 1)
        start_cycle = random.randint(boot_cycles, max_start)

        page_count, start_hash, end_hash = run_window_sample(
            command=command,
            start_cycle=start_cycle,
            window_size=window_size,
            log_path=log_path,
            rootfs_path=rootfs_path,
            linux_path=linux_path,
        )

        if not start_hash or not end_hash:
            continue

        metrics = run_risc0_prover(
            start_hash=start_hash,
            end_hash=end_hash,
            log_path=log_path,
            step_size=window_size,
        )

        if metrics:
            metrics["benchmark"] = name
            metrics["page_count"] = page_count
            metrics["window_size"] = window_size
            metrics["start_cycle"] = start_cycle
            results.append(metrics)
            store_metrics_jsonl(metrics, data_path)
            successful += 1

    print(f"\nCompleted {successful}/{num_samples} samples successfully.")

    generate_histograms(data_path, name, window_size, output_dir, hardware_profiles)

    return results


def run_benchmark(
    config: dict,
    linux_path: Path,
    rootfs_path: Path,
    min_step: int | None = None,
    max_step: int | None = None,
    increment: int | None = None,
    boot_cycles: int = 0,
    hardware_profiles: list[dict] | None = None,
) -> list[dict]:
    name = config["name"]
    command = config["command"]
    max_mcycle = boot_cycles

    step_config = config.get("step_sizes", {})
    min_step = min_step or step_config.get("min", 50000)
    max_step = max_step or step_config.get("max", 150000)
    increment = increment or step_config.get("increment", 10000)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = RESULTS_DIR / f"{name}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    log_path = output_dir / "log.bin"

    results = []

    print(f"\n{'='*60}")
    print(f"Running benchmark: {name}")
    print(f"Command: {command}")
    print(f"Boot cycles: {boot_cycles}")
    print(f"Step range: {min_step} to {max_step} (increment: {increment})")
    print(f"{'='*60}\n")

    for step_size in range(min_step, max_step + 1, increment):
        print(f"\n--- Step size: {step_size} ---")

        if log_path.exists():
            log_path.unlink()

        start_hash, end_hash = run_cartesi_machine(
            command=command,
            step_size=step_size,
            max_mcycle=max_mcycle,
            log_path=log_path,
            rootfs_path=rootfs_path,
            linux_path=linux_path,
        )

        if not start_hash or not end_hash:
            print(f"  Skipping step {step_size} due to machine failure.")
            continue

        page_count = read_page_count(log_path)
        print(f"  Start hash: {start_hash[:16]}...")
        print(f"  End hash: {end_hash[:16]}...")
        print(f"  Page count: {page_count}")

        metrics = run_risc0_prover(
            start_hash=start_hash,
            end_hash=end_hash,
            log_path=log_path,
            step_size=step_size,
        )

        if metrics:
            metrics["benchmark"] = name
            metrics["page_count"] = page_count
            results.append(metrics)
            print(f"  Metrics: {metrics}")
        else:
            print(f"  Failed to get metrics for step {step_size}.")

    results_path = output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    generate_plots(results, name, output_dir, hardware_profiles)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="ZK Benchmarks - RISC Zero proving benchmarks for Cartesi machine"
    )
    parser.add_argument(
        "benchmark",
        nargs="?",
        help="Benchmark name to run (without .yaml extension). Runs all if not specified.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available benchmarks",
    )
    parser.add_argument(
        "--build-only",
        action="store_true",
        help="Only build dependencies, don't run benchmarks",
    )
    parser.add_argument(
        "--min-step",
        type=int,
        help="Override minimum step size",
    )
    parser.add_argument(
        "--max-step",
        type=int,
        help="Override maximum step size",
    )
    parser.add_argument(
        "--increment",
        type=int,
        help="Override step increment",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Clean build artifacts",
    )
    parser.add_argument(
        "--monte-carlo",
        action="store_true",
        help="Run Monte Carlo experiment (random window sampling)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=100,
        help="Number of Monte Carlo samples (default: 100)",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=100000,
        help="Window size in cycles for Monte Carlo sampling (default: 100000)",
    )
    parser.add_argument(
        "--hardware",
        type=str,
        help="Hardware profile key from config.yaml for proving time estimation (e.g., risc0_rtx_3090_ti)",
    )

    args = parser.parse_args()

    if args.list:
        print("Available benchmarks:")
        for path in list_benchmarks():
            config = load_benchmark(path)
            mode = config.get("mode", "sweep")
            print(f"  - {config['name']} [{mode}]: {config.get('description', 'No description')}")
        return 0

    if args.clean:
        print("Cleaning build artifacts...")
        subprocess.run(["make", "clean"], cwd=MACHINE_DIR, capture_output=True)
        subprocess.run(["make", "clean"], cwd=MACHINE_DIR / "risc0", capture_output=True)
        print("Clean complete.")
        return 0

    print("Ensuring required images are available...")
    linux_path, rootfs_path = ensure_images()
    if not linux_path or not rootfs_path:
        print("ERROR: Failed to obtain required images.")
        return 1
    print(f"Linux image: {linux_path}")
    print(f"Rootfs image: {rootfs_path}")

    if not check_builds():
        print("ERROR: Failed to build required dependencies.")
        return 1

    if args.build_only:
        print("Build complete.")
        return 0

    boot_cycles = measure_boot_cycles(rootfs_path, linux_path)
    if not boot_cycles:
        print("ERROR: Failed to measure boot cycles.")
        return 1

    # Get hardware profile if specified
    hardware_profile = None
    if args.hardware:
        try:
            hardware_profile = get_hardware_profile(args.hardware)
            print(f"Using hardware profile: {hardware_profile['name']} ({args.hardware})")
        except ValueError as e:
            print(f"ERROR: {e}")
            return 1

    benchmarks_to_run = []

    if args.benchmark:
        yaml_path = BENCHMARKS_DIR / f"{args.benchmark}.yaml"
        if not yaml_path.exists():
            print(f"ERROR: Benchmark '{args.benchmark}' not found at {yaml_path}")
            return 1
        benchmarks_to_run.append(yaml_path)
    else:
        benchmarks_to_run = list_benchmarks()
        if not benchmarks_to_run:
            print("ERROR: No benchmarks found in benchmarks/ directory.")
            return 1

    all_results = {}
    ran_monte_carlo = False

    for yaml_path in benchmarks_to_run:
        config = load_benchmark(yaml_path)
        mode = config.get("mode", "sweep")

        # Get hardware profiles: CLI arg overrides benchmark config
        bench_hardware_profiles = []
        if hardware_profile:
            bench_hardware_profiles = [hardware_profile]
        elif config.get("hardware_profiles"):
            for hw_key in config["hardware_profiles"]:
                try:
                    bench_hardware_profiles.append(get_hardware_profile(hw_key))
                except ValueError as e:
                    print(f"WARNING: {e}")

        if args.monte_carlo:
            mode = "monte-carlo"

        if mode == "monte-carlo":
            ran_monte_carlo = True
            mc_config = config.get("monte_carlo", {})
            num_samples = args.num_samples if args.num_samples != 100 else mc_config.get("num_samples", 100)
            window_size = args.window_size if args.window_size != 100000 else mc_config.get("window_size", 100000)
            results = run_monte_carlo(
                config,
                linux_path=linux_path,
                rootfs_path=rootfs_path,
                num_samples=num_samples,
                window_size=window_size,
                boot_cycles=boot_cycles,
                hardware_profiles=bench_hardware_profiles,
            )
        else:
            results = run_benchmark(
                config,
                linux_path=linux_path,
                rootfs_path=rootfs_path,
                min_step=args.min_step,
                max_step=args.max_step,
                increment=args.increment,
                boot_cycles=boot_cycles,
                hardware_profiles=bench_hardware_profiles,
            )
        all_results[config["name"]] = results

    print("\n" + "="*60)
    if ran_monte_carlo:
        print("Monte Carlo experiment complete!")
    else:
        print("Benchmark run complete!")
    print("="*60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
