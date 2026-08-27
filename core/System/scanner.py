from core.logging import write_log

from .hardware import (
    bytes_to_gb,
    get_cpu_features,
    get_cpu_info,
    get_cuda_version,
    get_memory_channels,
    get_memory_info,
    get_nvidia_info,
    get_ram_modules,
    get_storage_info,
    get_system_info,
)


COMPONENT = "system/scanner"


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
                    bytes_to_gb(
                        memory["total"]
                    ),
                    2,
                ),

                "available_gb": round(
                    bytes_to_gb(
                        memory["available"]
                    ),
                    2,
                ),

                "used_gb": round(
                    bytes_to_gb(
                        memory["used"]
                    ),
                    2,
                ),

                "usage_percent": (
                    memory["usage_percent"]
                ),

                "modules": ram_modules,

                "channels": (
                    get_memory_channels()
                ),
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
                        bytes_to_gb(
                            drive["total"]
                        ),
                        2,
                    ),

                    "used_gb": round(
                        bytes_to_gb(
                            drive["used"]
                        ),
                        2,
                    ),

                    "free_gb": round(
                        bytes_to_gb(
                            drive["free"]
                        ),
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
