#!/usr/bin/env python3

import argparse
import json
import os
import re
import struct
import subprocess
import sys
import urllib.request
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import yaml
from dotenv import load_dotenv

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


def generate_plots(results: list[dict], benchmark_name: str, output_dir: Path):
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

    axs[2, 1].axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.97])

    plot_path = output_dir / "plots.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Plots saved to {plot_path}")


def run_benchmark(
    config: dict,
    linux_path: Path,
    rootfs_path: Path,
    min_step: int | None = None,
    max_step: int | None = None,
    increment: int | None = None,
) -> list[dict]:
    name = config["name"]
    command = config["command"]
    max_mcycle = config.get("max_mcycle", 60000000)

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

    generate_plots(results, name, output_dir)

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

    args = parser.parse_args()

    if args.list:
        print("Available benchmarks:")
        for path in list_benchmarks():
            config = load_benchmark(path)
            print(f"  - {config['name']}: {config.get('description', 'No description')}")
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

    for yaml_path in benchmarks_to_run:
        config = load_benchmark(yaml_path)
        results = run_benchmark(
            config,
            linux_path=linux_path,
            rootfs_path=rootfs_path,
            min_step=args.min_step,
            max_step=args.max_step,
            increment=args.increment,
        )
        all_results[config["name"]] = results

    print("\n" + "="*60)
    print("Benchmark run complete!")
    print("="*60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
