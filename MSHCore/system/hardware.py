"""Hardware information collection and diagnostics for system resources."""

import json
import os
import platform
import shutil
import subprocess

import psutil


def run_command(command: list[str], timeout: int = 5) -> str | None:
    """Run a system command and return stripped stdout if successful.

    Args:
        command: Command list to execute via subprocess.
        timeout: Maximum execution time in seconds. Defaults to 5.

    Returns:
        str | None: Stripped stdout output if exit code is 0, otherwise None.
    """
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            # This process's own stdin may carry a stream the caller depends
            # on, and PowerShell in particular reads stdin when it has one, so
            # a child inheriting it would consume those bytes.
            stdin=subprocess.DEVNULL,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        # Hardware probing is best-effort: a missing tool, a timeout or a
        # permission error all mean "unknown" rather than a failed scan.
        pass

    return None


def bytes_to_gb(value: int | float) -> float:
    """Convert bytes to gigabytes.

    Args:
        value: Size in bytes.

    Returns:
        float: Size in gigabytes (GiB).
    """
    return value / (1024 ** 3)


def format_gb(value: int | float) -> str:
    """Format bytes as a human-readable gigabyte string.

    Args:
        value: Size in bytes.

    Returns:
        str: Formatted string (e.g., '16.0 GB').
    """
    return f"{bytes_to_gb(value):.1f} GB"


def _linux_os_release(field: str) -> str | None:
    """Read one field from the Linux ``os-release`` file.

    Args:
        field: Variable name to look up, such as 'PRETTY_NAME'.

    Returns:
        str | None: The value with its quotes stripped, or None when neither
        the file nor the field is present.
    """
    for path in ("/etc/os-release", "/usr/lib/os-release"):
        try:
            with open(path, "r", encoding="utf-8") as file:
                for line in file:
                    name, separator, value = line.partition("=")

                    if separator and name.strip() == field:
                        return value.strip().strip('"') or None
        except OSError:
            continue

    return None


def _proc_cpuinfo(field: str) -> str | None:
    """Read one per-processor field from ``/proc/cpuinfo``.

    Args:
        field: Field name as the kernel prints it, such as 'model name'.

    Returns:
        str | None: The first value found, or None when the file cannot be
        read or the field is absent.
    """
    try:
        with open("/proc/cpuinfo", "r", encoding="utf-8") as file:
            for line in file:
                name, separator, value = line.partition(":")

                if separator and name.strip().lower() == field.lower():
                    return value.strip() or None
    except OSError:
        pass

    return None


def get_system_info() -> dict:
    """Get operating system identification and version details.

    Each platform is asked in its own terms: WMI through PowerShell on
    Windows, and the distribution's ``os-release`` file with the kernel
    release on Linux.

    Returns:
        dict: Dictionary containing OS name, version, build number,
            architecture, and Python version.
    """
    system_name = platform.system()

    if system_name == "Windows":
        edition = run_command([
            "powershell",
            "-NoProfile",
            "-Command",
            "(Get-CimInstance Win32_OperatingSystem).Caption",
        ])

        build = run_command([
            "powershell",
            "-NoProfile",
            "-Command",
            "(Get-CimInstance Win32_OperatingSystem).BuildNumber",
        ])

        display_version = run_command([
            "powershell",
            "-NoProfile",
            "-Command",
            "(Get-ItemProperty 'HKLM:\\SOFTWARE\\Microsoft\\Windows NT"
            "\\CurrentVersion').DisplayVersion",
        ])

        return {
            "name": edition or "Windows",
            "version": display_version or platform.release(),
            "build": build or "Unknown",
            "architecture": platform.machine(),
            "python": platform.python_version(),
        }

    if system_name == "Linux":
        distribution = (
            _linux_os_release("PRETTY_NAME") or _linux_os_release("NAME")
        )

        return {
            "name": distribution or "Linux",
            "version": _linux_os_release("VERSION_ID") or platform.release(),
            # The kernel release, which is the build number's counterpart.
            "build": platform.release(),
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


def get_cpu_info() -> dict:
    """Get detailed CPU hardware and clock frequency information.

    The processor model and its rated clock are read through WMI on Windows
    and from ``lscpu`` and ``/proc/cpuinfo`` on Linux.

    Returns:
        dict: Dictionary containing CPU model name, architecture, physical core
            count, logical thread count, reported clock speed, current
            frequency, and max frequency.
    """
    cpu_name = None
    reported_clock = None

    if platform.system() == "Windows":
        cpu_name = run_command([
            "powershell",
            "-NoProfile",
            "-Command",
            "(Get-CimInstance Win32_Processor).Name",
        ])

        reported_clock = run_command([
            "powershell",
            "-NoProfile",
            "-Command",
            "(Get-CimInstance Win32_Processor).MaxClockSpeed",
        ])

    elif platform.system() == "Linux":
        cpu_name = _proc_cpuinfo("model name")

        # lscpu's parsable output is numeric and locale-independent, which the
        # human-readable table is not; /proc/cpuinfo's own clock figure is the
        # fallback when util-linux is not installed.
        max_clock = run_command(["lscpu", "--parse=MAXMHZ"])

        if max_clock:
            for line in max_clock.splitlines():
                if line.strip() and not line.startswith("#"):
                    reported_clock = line.strip()
                    break

        reported_clock = reported_clock or _proc_cpuinfo("cpu MHz")

    physical_cores = psutil.cpu_count(logical=False)
    logical_threads = psutil.cpu_count(logical=True)

    frequency = psutil.cpu_freq()
    current_frequency = "Unknown"
    max_frequency = "Unknown"

    if frequency:
        if frequency.current:
            current_frequency = f"{frequency.current / 1000:.2f} GHz"
        if frequency.max:
            max_frequency = f"{frequency.max / 1000:.2f} GHz"

    if reported_clock:
        try:
            reported_clock = f"{float(reported_clock) / 1000:.2f} GHz"
        except ValueError:
            reported_clock = "Unknown"
    else:
        reported_clock = "Unknown"

    return {
        "model": cpu_name or platform.processor() or "Unknown",
        "architecture": platform.machine(),
        "physical_cores": physical_cores or "Unknown",
        "logical_threads": logical_threads or "Unknown",
        "reported_clock": reported_clock,
        "current_frequency": current_frequency,
        "max_frequency": max_frequency,
    }


def get_cpu_features() -> list[str]:
    """Detect available CPU instruction set features (e.g., AVX2, AVX-512).

    Exact instruction-set detection is platform dependent.

    Returns:
        list[str]: List of detected CPU feature and instruction set flags.
    """
    features = []

    if platform.system() == "Linux":
        try:
            with open("/proc/cpuinfo", "r", encoding="utf-8") as file:
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


def get_memory_info() -> dict:
    """Get system RAM capacity and utilization statistics.

    Returns:
        dict: Dictionary containing total, available, used memory in bytes,
            and usage percentage.
    """
    memory = psutil.virtual_memory()

    return {
        "total": memory.total,
        "available": memory.available,
        "used": memory.used,
        "usage_percent": memory.percent,
    }


def get_ram_modules() -> list[dict]:
    """Get physical RAM module specifications and slot layout.

    Modules are enumerated through WMI on Windows and read from the SMBIOS
    tables with ``dmidecode`` on Linux, which needs root for those tables; a
    machine where the read is refused reports no modules rather than failing
    the scan.

    Returns:
        list[dict]: List of RAM module dictionaries with manufacturer, part
            number, capacity, speed, configured speed, memory type, and slot
            location.
    """
    system_name = platform.system()

    if system_name == "Windows":
        return _windows_ram_modules()

    if system_name == "Linux":
        return _linux_ram_modules()

    return []


def _windows_ram_modules() -> list[dict]:
    """Read RAM module details from WMI through PowerShell.

    Returns:
        list[dict]: One dictionary per populated slot.
    """
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
        """,
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
            configured_speed = module.get("ConfiguredClockSpeed")
            memory_type = module.get("SMBIOSMemoryType")

            if capacity:
                capacity_text = f"{bytes_to_gb(int(capacity)):.1f} GB"
            else:
                capacity_text = "Unknown"

            speed_text = f"{speed} MT/s" if speed else "Unknown"
            configured_speed_text = (
                f"{configured_speed} MT/s" if configured_speed else "Unknown"
            )

            ram_type = memory_types.get(
                int(memory_type) if memory_type else -1,
                "Unknown",
            )

            modules.append({
                "manufacturer": module.get("Manufacturer") or "Unknown",
                "part_number": module.get("PartNumber") or "Unknown",
                "capacity": capacity_text,
                "speed": speed_text,
                "configured_speed": configured_speed_text,
                "type": ram_type,
                "slot": module.get("DeviceLocator") or "Unknown",
            })

        return modules

    except Exception:
        return []


def _linux_ram_modules() -> list[dict]:
    """Read RAM module details from SMBIOS with ``dmidecode``.

    Returns:
        list[dict]: One dictionary per populated slot, using the same keys as
        the Windows path so a caller reads one shape everywhere. Empty when
        dmidecode is missing or the SMBIOS tables cannot be read.
    """
    output = run_command(["dmidecode", "--type", "17"], timeout=10)

    if not output:
        return []

    # dmidecode's memory type names already match the strings the Windows
    # path reports, so only the older unnamed values need mapping.
    memory_types = {
        "DDR": "DDR",
        "DDR2": "DDR2",
        "DDR3": "DDR3",
        "DDR4": "DDR4",
        "DDR5": "DDR5",
    }

    modules = []
    current: dict[str, str] = {}

    def flush() -> None:
        """Append the module just parsed, when the slot holds one."""
        if current.get("Size", "No Module Installed") in (
            "No Module Installed",
            "Unknown",
            "",
        ):
            return

        size = current.get("Size", "")

        try:
            capacity_text = (
                f"{float(size.split()[0]):.1f} GB"
                if size.lower().endswith("gb")
                else size
            )
        except ValueError:
            capacity_text = size or "Unknown"

        # dmidecode reports the unit itself, while some tables carry the bare
        # number; both are normalised to the "3200 MT/s" form the Windows path
        # produces.
        def rate(value: str) -> str:
            """Format one memory speed field."""
            stripped = value.strip()

            if not stripped or stripped.lower() == "unknown":
                return "Unknown"

            return (
                stripped
                if "mt/s" in stripped.lower()
                else f"{stripped} MT/s"
            )

        speed = rate(current.get("Speed", "Unknown"))
        configured_speed = rate(
            current.get("Configured Memory Speed", "") or speed
        )
        ram_type = memory_types.get(
            current.get("Type", "").strip(),
            "Unknown",
        )

        modules.append({
            "manufacturer": (
                current.get("Manufacturer", "Unknown").strip() or "Unknown"
            ),
            "part_number": (
                current.get("Part Number", "Unknown").strip() or "Unknown"
            ),
            "capacity": capacity_text,
            "speed": speed,
            "configured_speed": configured_speed,
            "type": ram_type,
            "slot": (
                current.get("Locator", "Unknown").strip() or "Unknown"
            ),
        })

    for line in output.splitlines():
        if line.startswith("Memory Device"):
            flush()
            current = {}
            continue

        if ":" not in line:
            continue

        name, _, value = line.partition(":")
        current[name.strip()] = value.strip()

    flush()

    return modules


def get_memory_channels() -> str:
    """Determine the memory channel configuration.

    WMI does not expose this reliably across all hardware configurations.

    Returns:
        str: Channel topology description or 'Unknown'.
    """
    return "Unknown"


# ============================================================
# NVIDIA GPU
# ============================================================

def get_nvidia_info() -> list[dict]:
    """Get NVIDIA GPU device information and memory stats using nvidia-smi.

    Returns:
        list[dict]: List of GPU dictionaries containing name, driver version,
            vram total/used/free, and compute capability.
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
        parts = [item.strip() for item in line.split(",")]

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


def get_cuda_version() -> str | None:
    """Get the CUDA version reported by the NVIDIA driver.

    Returns:
        str | None: Installed CUDA version string (e.g., '12.2') or None.
    """
    output = run_command(["nvidia-smi"])

    if not output:
        return None

    for line in output.splitlines():
        if "CUDA Version" in line:
            try:
                value = line.split("CUDA Version:", 1)[1]
                return value.strip().split()[0]
            except Exception:
                pass

    return None


def get_vram_used() -> list[int]:
    """Sample the currently used VRAM of every NVIDIA GPU, in megabytes.

    One `nvidia-smi --query-gpu=memory.used` call, one snapshot. The number is
    what the driver reports for the whole GPU — every process on the machine
    is in it — so a caller that wants a model's own footprint samples while
    that model alone is the thing that changed.

    Returns:
        list[int]: Megabytes in use, one entry per GPU in nvidia-smi's order.
        Empty when nvidia-smi is missing, fails, or reports nothing — the same
        'no NVIDIA layer here' answer every other GPU probe gives.
    """
    output = run_command([
        "nvidia-smi",
        "--query-gpu=memory.used",
        "--format=csv,noheader,nounits",
    ])

    if not output:
        return []

    readings = []

    for line in output.splitlines():
        try:
            readings.append(int(float(line.strip())))
        except ValueError:
            continue

    return readings


def get_gpu_thermal() -> list[dict]:
    """Sample each NVIDIA GPU's temperature and current SM clock.

    One `nvidia-smi` call, one snapshot. The clock is the one the driver
    reports at that moment — after a GPU has throttled, the falling figure is
    the trace it leaves, which is what makes the reading worth pairing with
    the temperature that caused it.

    Returns:
        list[dict]: One entry per GPU carrying 'temperature_c' and
        'sm_clock_mhz', either None when that figure was not reported.
        Empty when nvidia-smi is missing, fails, or reports nothing — the
        same 'no NVIDIA layer here' answer every other GPU probe gives.
    """
    output = run_command([
        "nvidia-smi",
        "--query-gpu=temperature.gpu,clocks.current.sm",
        "--format=csv,noheader,nounits",
    ])

    if not output:
        return []

    def _number(value: str) -> int | None:
        try:
            return int(float(value))
        except ValueError:
            return None

    readings = []

    for line in output.splitlines():
        parts = [item.strip() for item in line.split(",")]

        if len(parts) < 2:
            continue

        temperature = _number(parts[0])
        clock = _number(parts[1])

        if temperature is not None or clock is not None:
            readings.append({
                "temperature_c": temperature,
                "sm_clock_mhz": clock,
            })

    return readings


# ============================================================
# Storage
# ============================================================

def get_storage_info() -> list[dict]:
    """Get storage drive metrics and disk usage for mounted filesystems.

    Windows enumerates drive letters; Linux reports the distinct physical
    filesystems behind its mount points, so a drive mounted in two places is
    counted once.

    Returns:
        list[dict]: List of drive dictionaries with drive path, total bytes,
            used bytes, and free bytes.
    """
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
                    "free": free,
                })
            except Exception:
                pass
    else:
        # One entry per device rather than per mount point, so /, /home and a
        # bind mount of either do not each report the same capacity.
        seen: set[tuple[int, int]] = set()

        try:
            partitions = psutil.disk_partitions(all=False)
        except Exception:
            partitions = []

        for partition in partitions:
            # Loopback and squashfs mounts — snaps, Flatpak runtimes, images —
            # are read-only payloads rather than storage the caller can use,
            # and reporting each one buries the real drives.
            if partition.fstype in ("squashfs", "iso9660") or partition.device.startswith("/dev/loop"):
                continue

            try:
                total, used, free = shutil.disk_usage(partition.mountpoint)
            except Exception:
                # An unreadable or permission-denied mount is skipped; the
                # rest of the machine is still worth reporting.
                continue

            identity = (total, free)

            if identity in seen:
                continue

            seen.add(identity)
            drives.append({
                "drive": partition.mountpoint,
                "total": total,
                "used": used,
                "free": free,
            })

        if not drives:
            # No usable partition table: the root filesystem is still a valid
            # answer, and the one the caller almost certainly wanted.
            try:
                total, used, free = shutil.disk_usage("/")
                drives.append({
                    "drive": "/",
                    "total": total,
                    "used": used,
                    "free": free,
                })
            except Exception:
                pass

    return drives
