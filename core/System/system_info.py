import json
import os
import platform
import shutil
import subprocess
import sys

import psutil

from core.logging import write_log


COMPONENT = "system/scanner"


# ============================================================
# Utility Functions
# ============================================================

def run_command(command, timeout=5):
    """Run a system command and return stdout."""

    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )

        if result.returncode == 0:
            return result.stdout.strip()

    except Exception:
        pass

    return None


def bytes_to_gb(value):
    """Convert bytes to GB."""

    return value / (1024 ** 3)


def format_gb(value):
    """Format bytes as GB."""

    return f"{bytes_to_gb(value):.1f} GB"


def print_section(title):
    """Print a clean section header."""

    print()
    print(f"┌─ {title}")
    print("│")


def print_item(label, value):
    """Print a formatted information item."""

    print(f"│  {label:<23} {value}")


# ============================================================
# Operating System
# ============================================================

def get_system_info():
    """Get operating system information."""

    system_name = platform.system()

    if system_name == "Windows":

        edition = run_command([
            "powershell",
            "-NoProfile",
            "-Command",
            "(Get-CimInstance Win32_OperatingSystem).Caption"
        ])

        build = run_command([
            "powershell",
            "-NoProfile",
            "-Command",
            "(Get-CimInstance Win32_OperatingSystem).BuildNumber"
        ])

        display_version = run_command([
            "powershell",
            "-NoProfile",
            "-Command",
            "(Get-ItemProperty "
            "'HKLM:\\SOFTWARE\\Microsoft\\Windows NT\\CurrentVersion'"
            ").DisplayVersion"
        ])

        return {
            "name": edition or "Windows",
            "version": display_version or platform.release(),
            "build": build or "Unknown",
            "architecture": platform.machine(),
            "python": platform.python_version(),
        }

    return {
        "name": system_name,
        "version": platform.release(),
        "build": platform.version(),
        "architecture": platform.machine(),
        "python": platform.python_version(),
    }


# ============================================================
# CPU
# ============================================================

def get_cpu_info():
    """Get detailed CPU information."""

    cpu_name = run_command([
        "powershell",
        "-NoProfile",
        "-Command",
        "(Get-CimInstance Win32_Processor).Name"
    ])

    physical_cores = psutil.cpu_count(logical=False)
    logical_threads = psutil.cpu_count(logical=True)

    frequency = psutil.cpu_freq()

    current_frequency = "Unknown"
    max_frequency = "Unknown"

    if frequency:

        if frequency.current:
            current_frequency = (
                f"{frequency.current / 1000:.2f} GHz"
            )

        if frequency.max:
            max_frequency = (
                f"{frequency.max / 1000:.2f} GHz"
            )

    reported_clock = run_command([
        "powershell",
        "-NoProfile",
        "-Command",
        "(Get-CimInstance Win32_Processor).MaxClockSpeed"
    ])

    if reported_clock:

        try:
            reported_clock = (
                f"{int(reported_clock) / 1000:.2f} GHz"
            )

        except ValueError:
            reported_clock = "Unknown"

    else:
        reported_clock = "Unknown"

    return {
        "model": (
            cpu_name
            or platform.processor()
            or "Unknown"
        ),
        "architecture": platform.machine(),
        "physical_cores": physical_cores or "Unknown",
        "logical_threads": logical_threads or "Unknown",
        "reported_clock": reported_clock,
        "current_frequency": current_frequency,
        "max_frequency": max_frequency,
    }


# ============================================================
# CPU Features
# ============================================================

def get_cpu_features():
    """
    Detect CPU instruction sets where possible.

    Exact instruction-set detection is platform dependent.
    """

    features = []

    if platform.system() == "Linux":

        try:

            with open("/proc/cpuinfo", "r") as file:
                cpuinfo = file.read().lower()

            possible_features = [
                ("AVX-512", "avx512"),
                ("AVX2", "avx2"),
                ("AVX", "avx"),
                ("FMA", "fma"),
                ("SSE4.2", "sse4_2"),
                ("SSE4.1", "sse4_1"),
            ]

            for display_name, cpu_flag in possible_features:

                if cpu_flag in cpuinfo:
                    features.append(display_name)

        except Exception:
            pass

    elif platform.system() == "Windows":

        architecture = platform.machine().lower()

        if architecture in ("amd64", "x86_64"):
            features.append("x86-64")

        elif architecture == "arm64":
            features.append("ARM64")

    return features


# ============================================================
# RAM
# ============================================================

def get_memory_info():
    """Get RAM usage information."""

    memory = psutil.virtual_memory()

    return {
        "total": memory.total,
        "available": memory.available,
        "used": memory.used,
        "usage_percent": memory.percent,
    }


def get_ram_modules():
    """
    Get physical RAM module information on Windows.
    """

    if platform.system() != "Windows":
        return []

    command = [
        "powershell",
        "-NoProfile",
        "-Command",
        """
        Get-CimInstance Win32_PhysicalMemory |
        Select-Object Manufacturer,
                      PartNumber,
                      Capacity,
                      Speed,
                      ConfiguredClockSpeed,
                      SMBIOSMemoryType,
                      DeviceLocator |
        ConvertTo-Json -Compress
        """
    ]

    output = run_command(command)

    if not output:
        return []

    try:

        data = json.loads(output)

        if isinstance(data, dict):
            data = [data]

        memory_types = {
            20: "DDR",
            21: "DDR2",
            22: "DDR2",
            24: "DDR3",
            26: "DDR4",
            27: "DDR5",
        }

        modules = []

        for module in data:

            capacity = module.get("Capacity")
            speed = module.get("Speed")
            configured_speed = module.get(
                "ConfiguredClockSpeed"
            )
            memory_type = module.get(
                "SMBIOSMemoryType"
            )

            if capacity:

                capacity_text = (
                    f"{bytes_to_gb(int(capacity)):.1f} GB"
                )

            else:
                capacity_text = "Unknown"

            speed_text = (
                f"{speed} MT/s"
                if speed
                else "Unknown"
            )

            configured_speed_text = (
                f"{configured_speed} MT/s"
                if configured_speed
                else "Unknown"
            )

            ram_type = memory_types.get(
                int(memory_type)
                if memory_type
                else -1,
                "Unknown",
            )

            modules.append({
                "manufacturer": (
                    module.get("Manufacturer")
                    or "Unknown"
                ),
                "part_number": (
                    module.get("PartNumber")
                    or "Unknown"
                ),
                "capacity": capacity_text,
                "speed": speed_text,
                "configured_speed": (
                    configured_speed_text
                ),
                "type": ram_type,
                "slot": (
                    module.get("DeviceLocator")
                    or "Unknown"
                ),
            })

        return modules

    except Exception:
        return []


def get_memory_channels():
    """
    Determine memory channel configuration.

    WMI does not expose this reliably across all systems.
    """

    return "Unknown"


# ============================================================
# NVIDIA GPU
# ============================================================

def get_nvidia_info():
    """
    Get NVIDIA GPU information using nvidia-smi.
    """

    command = [
        "nvidia-smi",
        "--query-gpu="
        "name,"
        "driver_version,"
        "memory.total,"
        "memory.used,"
        "memory.free,"
        "compute_cap",
        "--format=csv,noheader,nounits",
    ]

    output = run_command(command)

    if not output:
        return []

    gpus = []

    for line in output.splitlines():

        parts = [
            item.strip()
            for item in line.split(",")
        ]

        if len(parts) < 6:
            continue

        gpus.append({
            "name": parts[0],
            "driver": parts[1],
            "vram_total": f"{parts[2]} MB",
            "vram_used": f"{parts[3]} MB",
            "vram_free": f"{parts[4]} MB",
            "compute_capability": parts[5],
        })

    return gpus


def get_cuda_version():
    """Get CUDA version reported by NVIDIA driver."""

    output = run_command(["nvidia-smi"])

    if not output:
        return None

    for line in output.splitlines():

        if "CUDA Version" in line:

            try:

                value = line.split(
                    "CUDA Version:",
                    1
                )[1]

                return value.strip().split()[0]

            except Exception:
                pass

    return None


# ============================================================
# Storage
# ============================================================

def get_storage_info():
    """Get storage information for mounted drives."""

    drives = []

    if platform.system() == "Windows":

        for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":

            drive = f"{letter}:\\"

            if not os.path.exists(drive):
                continue

            try:

                total, used, free = (
                    shutil.disk_usage(drive)
                )

                drives.append({
                    "drive": drive,
                    "total": total,
                    "used": used,
                    "free": free,
                })

            except Exception:
                pass

    else:

        try:

            total, used, free = (
                shutil.disk_usage("/")
            )

            drives.append({
                "drive": "/",
                "total": total,
                "used": used,
                "free": free,
            })

        except Exception:
            pass

    return drives


# ============================================================
# System Scan
# ============================================================

def scan_system():
    """
    Collect the complete system hardware/software profile.

    Returns:
        dict: Machine-readable system profile.
    """

    write_log(
        level="INFO",
        component=COMPONENT,
        action="scan",
        message="System hardware scan started",
    )

    try:

        system = get_system_info()
        cpu = get_cpu_info()
        cpu_features = get_cpu_features()
        memory = get_memory_info()
        ram_modules = get_ram_modules()
        gpus = get_nvidia_info()
        cuda_version = get_cuda_version()
        storage = get_storage_info()

        profile = {
            "system": system,

            "cpu": {
                **cpu,
                "features": cpu_features,
            },

            "memory": {
                "total_gb": round(
                    bytes_to_gb(memory["total"]),
                    2,
                ),
                "available_gb": round(
                    bytes_to_gb(memory["available"]),
                    2,
                ),
                "used_gb": round(
                    bytes_to_gb(memory["used"]),
                    2,
                ),
                "usage_percent": memory[
                    "usage_percent"
                ],
                "modules": ram_modules,
                "channels": get_memory_channels(),
            },

            "gpu": {
                "count": len(gpus),
                "devices": gpus,
                "cuda_version": cuda_version,
            },

            "storage": [
                {
                    "drive": drive["drive"],
                    "total_gb": round(
                        bytes_to_gb(drive["total"]),
                        2,
                    ),
                    "used_gb": round(
                        bytes_to_gb(drive["used"]),
                        2,
                    ),
                    "free_gb": round(
                        bytes_to_gb(drive["free"]),
                        2,
                    ),
                }
                for drive in storage
            ],
        }

        write_log(
            level="INFO",
            component=COMPONENT,
            action="scan",
            message="System hardware scan completed",
            details=profile,
        )

        return profile

    except Exception as error:

        write_log(
            level="ERROR",
            component=COMPONENT,
            action="scan",
            message="System hardware scan failed",
            details={
                "error_type": type(error).__name__,
                "error": str(error),
            },
        )

        raise


# ============================================================
# Display
# ============================================================

def display_system_info(profile):
    """Display the system profile in a clean terminal layout."""

    system = profile["system"]
    cpu = profile["cpu"]
    memory = profile["memory"]
    gpu = profile["gpu"]
    storage = profile["storage"]

    print()
    print("╔══════════════════════════════════════════════════════╗")
    print("║              LOCAL AI SYSTEM SCANNER                ║")
    print("╚══════════════════════════════════════════════════════╝")

    # --------------------------------------------------------
    # System
    # --------------------------------------------------------

    print_section("SYSTEM")

    print_item(
        "Operating System",
        system["name"],
    )

    print_item(
        "Version",
        system["version"],
    )

    print_item(
        "Build",
        system["build"],
    )

    print_item(
        "Architecture",
        system["architecture"],
    )

    print_item(
        "Python",
        system["python"],
    )

    # --------------------------------------------------------
    # CPU
    # --------------------------------------------------------

    print_section("CPU")

    print_item(
        "Model",
        cpu["model"],
    )

    print_item(
        "Architecture",
        cpu["architecture"],
    )

    print_item(
        "Physical Cores",
        cpu["physical_cores"],
    )

    print_item(
        "Logical Threads",
        cpu["logical_threads"],
    )

    print_item(
        "Reported Clock",
        cpu["reported_clock"],
    )

    print_item(
        "Current Clock",
        cpu["current_frequency"],
    )

    print_item(
        "Max Clock",
        cpu["max_frequency"],
    )

    # --------------------------------------------------------
    # CPU Features
    # --------------------------------------------------------

    print_section("CPU FEATURES")

    if cpu["features"]:

        print_item(
            "Instruction Sets",
            ", ".join(cpu["features"]),
        )

    else:

        print_item(
            "Instruction Sets",
            "Not detected",
        )

    # --------------------------------------------------------
    # Memory
    # --------------------------------------------------------

    print_section("MEMORY")

    print_item(
        "Total RAM",
        f"{memory['total_gb']:.1f} GB",
    )

    print_item(
        "Available RAM",
        f"{memory['available_gb']:.1f} GB",
    )

    print_item(
        "Used RAM",
        f"{memory['used_gb']:.1f} GB",
    )

    print_item(
        "Usage",
        f"{memory['usage_percent']:.1f}%",
    )

    if memory["modules"]:

        print_item(
            "Memory Type",
            memory["modules"][0]["type"],
        )

        print_item(
            "Module Count",
            len(memory["modules"]),
        )

        for index, module in enumerate(
            memory["modules"],
            start=1,
        ):

            print()

            print_item(
                f"Module {index}",
                module["capacity"],
            )

            print_item(
                "Speed",
                module["speed"],
            )

            print_item(
                "Configured Speed",
                module["configured_speed"],
            )

            print_item(
                "Manufacturer",
                module["manufacturer"],
            )

            print_item(
                "Slot",
                module["slot"],
            )

        print_item(
            "Memory Channels",
            memory["channels"],
        )

    # --------------------------------------------------------
    # GPU
    # --------------------------------------------------------

    print_section("GPU")

    if gpu["devices"]:

        print_item(
            "GPU Count",
            gpu["count"],
        )

        for index, device in enumerate(
            gpu["devices"],
            start=1,
        ):

            print()

            print_item(
                f"GPU {index}",
                device["name"],
            )

            print_item(
                "VRAM Total",
                device["vram_total"],
            )

            print_item(
                "VRAM Used",
                device["vram_used"],
            )

            print_item(
                "VRAM Free",
                device["vram_free"],
            )

            print_item(
                "Driver",
                device["driver"],
            )

            print_item(
                "Compute Capability",
                device["compute_capability"],
            )

        if gpu["cuda_version"]:

            print_item(
                "CUDA",
                gpu["cuda_version"],
            )

    else:

        print_item(
            "NVIDIA GPU",
            "Not detected",
        )

    # --------------------------------------------------------
    # Storage
    # --------------------------------------------------------

    print_section("STORAGE")

    if storage:

        for drive in storage:

            print_item(
                drive["drive"],
                f"{drive['free_gb']:.1f} GB free / "
                f"{drive['total_gb']:.1f} GB total",
            )

    else:

        print_item(
            "Storage",
            "Not detected",
        )

    print()
    print("└─ System scan completed")
    print()


# ============================================================
# Main
# ============================================================

def main():

    try:

        profile = scan_system()

        display_system_info(profile)

    except Exception as error:

        print()
        print("System scan failed.")
        print(f"Error: {error}")
        print()


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":
    main()