import subprocess
import tempfile
from pathlib import Path
import urllib
import json

from core.logging import write_log


def run_command(command):
    """
    Execute an Ollama command.
    """

    return subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )


def list_models():
    """
    List installed Ollama models.
    """

    result = run_command(
        ["ollama", "list"]
    )

    return result.stdout.strip()


def show_model_info(model: str):
    """
    Show information about a model.
    """

    result = run_command(
        ["ollama", "show", model]
    )

    if result.returncode != 0:
        raise RuntimeError(
            result.stderr.strip()
            or f"Failed to get information for model: {model}"
        )

    return result.stdout.strip()


def add_model(
    model_name: str,
    model_path: str,
):
    """
    Add a local model to Ollama.
    """

    model_file = Path(
        model_path
    ).expanduser().resolve()

    if not model_file.is_file():
        raise FileNotFoundError(
            f"Model file not found: {model_file}"
        )

    with tempfile.TemporaryDirectory() as temp_dir:
        modelfile = Path(temp_dir) / "Modelfile"

        modelfile.write_text(
            f"FROM {model_file}\n",
            encoding="utf-8",
        )

        result = run_command(
            [
                "ollama",
                "create",
                model_name,
                "-f",
                str(modelfile),
            ]
        )

    if result.returncode != 0:
        write_log(
            level="ERROR",
            component="ollama/model",
            action="add",
            message="Failed to add model",
            details={
                "model": model_name,
                "error": result.stderr.strip(),
            },
        )

        raise RuntimeError(
            result.stderr.strip()
            or f"Failed to add model: {model_name}"
        )

    write_log(
        level="INFO",
        component="ollama/model",
        action="add",
        message="Model added successfully",
        details={
            "model": model_name,
            "path": str(model_file),
        },
    )

    return result.stdout.strip()


def remove_model(model: str):
    """
    Remove a model from Ollama.
    """

    result = run_command(
        [
            "ollama",
            "rm",
            model,
        ]
    )

    if result.returncode != 0:
        write_log(
            level="ERROR",
            component="ollama/model",
            action="remove",
            message="Failed to remove model",
            details={
                "model": model,
                "error": result.stderr.strip(),
            },
        )

        raise RuntimeError(
            result.stderr.strip()
            or f"Failed to remove model: {model}"
        )

    write_log(
        level="INFO",
        component="ollama/model",
        action="remove",
        message="Model removed successfully",
        details={
            "model": model,
        },
    )

    return result.stdout.strip()


def run_model(
    model: str,
    prompt: str,
):
    """
    Run a prompt using a model.
    """

    result = run_command(
        [
            "ollama",
            "run",
            model,
            prompt,
        ]
    )

    if result.returncode != 0:
        write_log(
            level="ERROR",
            component="ollama/model",
            action="run",
            message="Failed to run model",
            details={
                "model": model,
                "error": result.stderr.strip(),
            },
        )

        raise RuntimeError(
            result.stderr.strip()
            or f"Failed to run model: {model}"
        )

    write_log(
        level="INFO",
        component="ollama/model",
        action="run",
        message="Model executed successfully",
        details={
            "model": model,
            "prompt_length": len(prompt),
        },
    )

    return result.stdout.strip()


def stop_model(model: str):
    """
    Stop a running model.
    """

    result = run_command(
        [
            "ollama",
            "stop",
            model,
        ]
    )

    if result.returncode != 0:
        write_log(
            level="ERROR",
            component="ollama/model",
            action="stop",
            message="Failed to stop model",
            details={
                "model": model,
                "error": result.stderr.strip(),
            },
        )

        raise RuntimeError(
            result.stderr.strip()
            or f"Failed to stop model: {model}"
        )

    write_log(
        level="INFO",
        component="ollama/model",
        action="stop",
        message="Model stopped successfully",
        details={
            "model": model,
        },
    )

    return result.stdout.strip()


def load_model(
    model: str,
    keep_alive: str = "10m",
):
    """
    Load a model into memory if it is not already loaded.
    """

    if not model.strip():
        raise ValueError(
            "Model name is required"
        )

    # Check whether the model is already loaded.
    try:
        with urllib.request.urlopen(
            "http://127.0.0.1:11434/api/ps",
            timeout=5,
        ) as response:
            data = json.loads(
                response.read().decode("utf-8")
            )

    except (
        urllib.error.URLError,
        TimeoutError,
        OSError,
        json.JSONDecodeError,
    ) as error:
        write_log(
            level="ERROR",
            component="ollama/model",
            action="load",
            message="Failed to check loaded models",
            details={
                "model": model,
                "error": str(error),
            },
        )

        raise RuntimeError(
            "Failed to check loaded Ollama models"
        ) from error

    requested_model = model.strip()

    for item in data.get("models", []):
        loaded_model = item.get("name", "").strip()

        if (
            loaded_model == requested_model
            or (
                ":" not in requested_model
                and loaded_model == f"{requested_model}:latest"
            )
        ):
            write_log(
                level="INFO",
                component="ollama/model",
                action="load",
                message="Model is already loaded",
                details={
                    "model": model,
                },
            )

            return

    # Model is not loaded. Load it into memory.
    payload = {
        "model": model,
        "prompt": "",
        "stream": False,
        "keep_alive": keep_alive,
    }

    request = urllib.request.Request(
        "http://127.0.0.1:11434/api/generate",
        data=json.dumps(
            payload
        ).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(
            request,
            timeout=300,
        ) as response:
            raw = response.read()

    except (
        urllib.error.URLError,
        TimeoutError,
        OSError,
    ) as error:
        write_log(
            level="ERROR",
            component="ollama/model",
            action="load",
            message="Failed to load model",
            details={
                "model": model,
                "error": str(error),
            },
        )

        raise RuntimeError(
            f"Failed to load model: {model}"
        ) from error

    try:
        result = json.loads(
            raw.decode("utf-8")
        )

    except json.JSONDecodeError as error:
        raise RuntimeError(
            "Ollama returned invalid JSON"
        ) from error

    if "error" in result:
        write_log(
            level="ERROR",
            component="ollama/model",
            action="load",
            message="Failed to load model",
            details={
                "model": model,
                "error": result["error"],
            },
        )

        raise RuntimeError(
            result["error"]
        )

    write_log(
        level="INFO",
        component="ollama/model",
        action="load",
        message="Model loaded successfully",
        details={
            "model": model,
            "keep_alive": keep_alive,
        },
    )

    return result


def list_running_models():
    """
    List currently running models.
    """

    result = run_command(
        ["ollama", "ps"]
    )

    return result.stdout.strip()


def configure_model(
    source_model: str,
    target_model: str,
    parameters: dict,
):
    """
    Create a new configured model from an existing model.

    The source model is not modified.
    """

    if not source_model.strip():
        raise ValueError(
            "Source model name is required"
        )

    if not target_model.strip():
        raise ValueError(
            "Target model name is required"
        )

    if not parameters:
        raise ValueError(
            "At least one configuration parameter is required"
        )

    if not isinstance(parameters, dict):
        raise TypeError(
            "Parameters must be a dictionary"
        )

    modelfile_parameters = []

    for key, value in parameters.items():
        if value is None:
            continue

        modelfile_parameters.append(
            f"PARAMETER {key} {value}"
        )

    if not modelfile_parameters:
        raise ValueError(
            "No valid configuration parameters found"
        )

    with tempfile.TemporaryDirectory() as temp_dir:
        modelfile = Path(temp_dir) / "Modelfile"

        content = (
            f"FROM {source_model}\n"
            + "\n".join(modelfile_parameters)
            + "\n"
        )

        modelfile.write_text(
            content,
            encoding="utf-8",
        )

        result = run_command(
            [
                "ollama",
                "create",
                target_model,
                "-f",
                str(modelfile),
            ]
        )

    if result.returncode != 0:
        write_log(
            level="ERROR",
            component="ollama/model",
            action="configure",
            message="Failed to create configured model",
            details={
                "source_model": source_model,
                "target_model": target_model,
                "error": result.stderr.strip(),
            },
        )

        raise RuntimeError(
            result.stderr.strip()
            or f"Failed to configure model: {target_model}"
        )

    write_log(
        level="INFO",
        component="ollama/model",
        action="configure",
        message="Configured model created successfully",
        details={
            "source_model": source_model,
            "target_model": target_model,
            "parameters": parameters,
        },
    )

    return result.stdout.strip()