import json
import os
import platform
import shutil
import subprocess

import psutil


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
                "configured_speed": configured_speed_text,
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