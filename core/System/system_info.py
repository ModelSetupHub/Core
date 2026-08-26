import os
import platform
import shutil
import subprocess
import sys

import psutil


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
            timeout=timeout
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
    """Get detailed operating system information."""

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
            "python": platform.python_version()
        }

    return {
        "name": system_name,
        "version": platform.release(),
        "build": platform.version(),
        "architecture": platform.machine(),
        "python": platform.python_version()
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

    # Get processor base/max clock from Windows
    clock_speed = run_command([
        "powershell",
        "-NoProfile",
        "-Command",
        "(Get-CimInstance Win32_Processor).MaxClockSpeed"
    ])

    if clock_speed:

        try:
            base_frequency = f"{int(clock_speed) / 1000:.2f} GHz"

        except ValueError:
            base_frequency = "Unknown"

    else:
        base_frequency = "Unknown"

    return {
        "model": cpu_name or platform.processor() or "Unknown",
        "architecture": platform.machine(),
        "physical_cores": physical_cores or "Unknown",
        "logical_threads": logical_threads or "Unknown",
        "base_frequency": base_frequency,
        "current_frequency": current_frequency,
        "max_frequency": max_frequency
    }


# ============================================================
# CPU Features
# ============================================================

def get_cpu_features():
    """
    Detect CPU instruction sets.

    These can be important for CPU-based inference.
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
                ("SSE4.1", "sse4_1")
            ]

            for display_name, cpu_flag in possible_features:

                if cpu_flag in cpuinfo:
                    features.append(display_name)

        except Exception:
            pass

    elif platform.system() == "Windows":

        # Windows does not expose all CPU instruction sets
        # reliably through WMI.
        # PowerShell is used here for basic architecture detection.

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
    """Get RAM capacity and usage."""

    memory = psutil.virtual_memory()

    return {
        "total": memory.total,
        "available": memory.available,
        "used": memory.used,
        "usage_percent": memory.percent
    }


def get_ram_modules():
    """
    Get physical RAM module information on Windows.

    Requires PowerShell / WMI.
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
        import json

        data = json.loads(output)

        if isinstance(data, dict):
            data = [data]

        modules = []

        memory_types = {
            20: "DDR",
            21: "DDR2",
            22: "DDR2",
            24: "DDR3",
            26: "DDR4",
            27: "DDR5"
        }

        for module in data:

            capacity = module.get("Capacity")
            speed = module.get("Speed")
            configured_speed = module.get("ConfiguredClockSpeed")
            memory_type = module.get("SMBIOSMemoryType")

            if capacity:
                capacity_gb = bytes_to_gb(int(capacity))
                capacity_text = f"{capacity_gb:.1f} GB"
            else:
                capacity_text = "Unknown"

            if speed:
                speed_text = f"{speed} MT/s"
            else:
                speed_text = "Unknown"

            if configured_speed:
                configured_speed_text = (
                    f"{configured_speed} MT/s"
                )
            else:
                configured_speed_text = "Unknown"

            ram_type = memory_types.get(
                int(memory_type) if memory_type else -1,
                "Unknown"
            )

            modules.append({
                "manufacturer": module.get(
                    "Manufacturer"
                ) or "Unknown",

                "part_number": module.get(
                    "PartNumber"
                ) or "Unknown",

                "capacity": capacity_text,
                "speed": speed_text,
                "configured_speed": configured_speed_text,
                "type": ram_type,

                "slot": module.get(
                    "DeviceLocator"
                ) or "Unknown"
            })

        return modules

    except Exception:
        return []


def get_memory_channels():
    """
    Try to determine memory channel configuration.

    Windows does not expose this consistently, so this
    may return Unknown.
    """

    if platform.system() != "Windows":
        return "Unknown"

    # There is no universally reliable WMI field for
    # active memory channel configuration.
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

        "--format=csv,noheader,nounits"
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
            "compute_capability": parts[5]
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
                value = line.split("CUDA Version:")[1]
                return value.strip().split()[0]

            except Exception:
                pass

    return None


# ============================================================
# Storage
# ============================================================

def get_storage_info():
    """Get available storage on all mounted drives."""

    drives = []

    if platform.system() == "Windows":

        for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":

            drive = f"{letter}:\\"

            if not os.path.exists(drive):
                continue

            try:

                total, used, free = shutil.disk_usage(drive)

                drives.append({
                    "drive": drive,
                    "total": total,
                    "used": used,
                    "free": free
                })

            except Exception:
                pass

    else:

        try:

            total, used, free = shutil.disk_usage("/")

            drives.append({
                "drive": "/",
                "total": total,
                "used": used,
                "free": free
            })

        except Exception:
            pass

    return drives


# ============================================================
# Main
# ============================================================

def main():

    print()
    print("╔══════════════════════════════════════════════════════╗")
    print("║              LOCAL AI SYSTEM SCANNER                ║")
    print("╚══════════════════════════════════════════════════════╝")

    # --------------------------------------------------------
    # System
    # --------------------------------------------------------

    system = get_system_info()

    print_section("SYSTEM")

    print_item(
        "Operating System",
        system["name"]
    )

    print_item(
        "Version",
        system["version"]
    )

    print_item(
        "Build",
        system["build"]
    )

    print_item(
        "Architecture",
        system["architecture"]
    )

    print_item(
        "Python",
        system["python"]
    )

    # --------------------------------------------------------
    # CPU
    # --------------------------------------------------------

    cpu = get_cpu_info()

    print_section("CPU")

    print_item(
        "Model",
        cpu["model"]
    )

    print_item(
        "Architecture",
        cpu["architecture"]
    )

    print_item(
        "Physical Cores",
        cpu["physical_cores"]
    )

    print_item(
        "Logical Threads",
        cpu["logical_threads"]
    )

    print_item(
        "Base Clock",
        cpu["base_frequency"]
    )

    print_item(
        "Current Clock",
        cpu["current_frequency"]
    )

    print_item(
        "Max Clock",
        cpu["max_frequency"]
    )

    # --------------------------------------------------------
    # CPU Features
    # --------------------------------------------------------

    features = get_cpu_features()

    print_section("CPU FEATURES")

    if features:
        print_item(
            "Instruction Sets",
            ", ".join(features)
        )
    else:
        print_item(
            "Instruction Sets",
            "Not detected"
        )

    # --------------------------------------------------------
    # Memory
    # --------------------------------------------------------

    memory = get_memory_info()

    ram_modules = get_ram_modules()

    print_section("MEMORY")

    print_item(
        "Total RAM",
        format_gb(memory["total"])
    )

    print_item(
        "Available RAM",
        format_gb(memory["available"])
    )

    print_item(
        "Used RAM",
        format_gb(memory["used"])
    )

    print_item(
        "Usage",
        f"{memory['usage_percent']:.1f}%"
    )

    if ram_modules:

        print_item(
            "Memory Type",
            ram_modules[0]["type"]
        )

        print_item(
            "Module Count",
            len(ram_modules)
        )

        for index, module in enumerate(
            ram_modules,
            start=1
        ):

            print()
            print_item(
                f"Module {index}",
                module["capacity"]
            )

            print_item(
                "Speed",
                module["speed"]
            )

            print_item(
                "Configured Speed",
                module["configured_speed"]
            )

            print_item(
                "Manufacturer",
                module["manufacturer"]
            )

            print_item(
                "Slot",
                module["slot"]
            )

        print_item(
            "Memory Channels",
            get_memory_channels()
        )

    # --------------------------------------------------------
    # GPU
    # --------------------------------------------------------

    gpus = get_nvidia_info()

    print_section("GPU")

    if gpus:

        print_item(
            "GPU Count",
            len(gpus)
        )

        for index, gpu in enumerate(
            gpus,
            start=1
        ):

            print()

            print_item(
                f"GPU {index}",
                gpu["name"]
            )

            print_item(
                "VRAM Total",
                gpu["vram_total"]
            )

            print_item(
                "VRAM Used",
                gpu["vram_used"]
            )

            print_item(
                "VRAM Free",
                gpu["vram_free"]
            )

            print_item(
                "Driver",
                gpu["driver"]
            )

            print_item(
                "Compute Capability",
                gpu["compute_capability"]
            )

        cuda = get_cuda_version()

        if cuda:

            print_item(
                "CUDA",
                cuda
            )

    else:

        print_item(
            "NVIDIA GPU",
            "Not detected"
        )

    # --------------------------------------------------------
    # Storage
    # --------------------------------------------------------

    storage = get_storage_info()

    print_section("STORAGE")

    if storage:

        for drive in storage:

            print_item(
                drive["drive"],
                f"{format_gb(drive['free'])} free / "
                f"{format_gb(drive['total'])} total"
            )

    else:

        print_item(
            "Storage",
            "Not detected"
        )

    # --------------------------------------------------------
    # Finish
    # --------------------------------------------------------

    print()
    print("└─ System scan completed")
    print()


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":
    main()