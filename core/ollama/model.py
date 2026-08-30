"""Ollama model lifecycle management, configuration, and execution."""

import json
from pathlib import Path
import subprocess
import tempfile
import urllib.error
import urllib.request

from core.logging import write_log

COMPONENT = "ollama/model"


def _run_command(command: list[str]) -> subprocess.CompletedProcess:
    """Execute an Ollama CLI subprocess command.

    Args:
        command: List of command arguments.

    Returns:
        subprocess.CompletedProcess: Result with returncode, stdout, stderr.
    """
    return subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )


def list_models() -> str:
    """List installed Ollama models via CLI.

    Returns:
        str: Output text from 'ollama list'.
    """
    result = _run_command(["ollama", "list"])
    return result.stdout.strip()


def show_model_info(model: str) -> str:
    """Show metadata and layer information for an installed model.

    Args:
        model: Model name or tag.

    Returns:
        str: Detailed model information output from 'ollama show'.

    Raises:
        RuntimeError: If the model cannot be found or CLI command fails.
    """
    result = _run_command(["ollama", "show", model])

    if result.returncode != 0:
        raise RuntimeError(
            result.stderr.strip()
            or f"Failed to get information for model: {model}"
        )

    return result.stdout.strip()


def add_model(
    model_name: str,
    model_path: str,
) -> str:
    """Import and register a local GGUF/model file into Ollama.

    Args:
        model_name: Desired model identifier name in Ollama.
        model_path: Local filesystem path to the model file.

    Returns:
        str: Ollama CLI output upon successful creation.

    Raises:
        FileNotFoundError: If the specified model file does not exist.
        RuntimeError: If Ollama fails to create the model.
    """
    model_file = Path(model_path).expanduser().resolve()

    if not model_file.is_file():
        raise FileNotFoundError(f"Model file not found: {model_file}")

    with tempfile.TemporaryDirectory() as temp_dir:
        modelfile = Path(temp_dir) / "Modelfile"
        modelfile.write_text(
            f"FROM {model_file}\n",
            encoding="utf-8",
        )

        result = _run_command([
            "ollama",
            "create",
            model_name,
            "-f",
            str(modelfile),
        ])

    if result.returncode != 0:
        write_log(
            level="ERROR",
            component=COMPONENT,
            action="add",
            message="Failed to add model",
            details={
                "model": model_name,
                "error": result.stderr.strip(),
            },
        )
        raise RuntimeError(
            result.stderr.strip() or f"Failed to add model: {model_name}"
        )

    write_log(
        level="INFO",
        component=COMPONENT,
        action="add",
        message="Model added successfully",
        details={
            "model": model_name,
            "path": str(model_file),
        },
    )

    return result.stdout.strip()


def remove_model(model: str) -> str:
    """Delete a model from local Ollama storage.

    Args:
        model: Model name to remove.

    Returns:
        str: Output text from 'ollama rm'.

    Raises:
        RuntimeError: If removal fails.
    """
    result = _run_command([
        "ollama",
        "rm",
        model,
    ])

    if result.returncode != 0:
        write_log(
            level="ERROR",
            component=COMPONENT,
            action="remove",
            message="Failed to remove model",
            details={
                "model": model,
                "error": result.stderr.strip(),
            },
        )
        raise RuntimeError(
            result.stderr.strip() or f"Failed to remove model: {model}"
        )

    write_log(
        level="INFO",
        component=COMPONENT,
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
) -> str:
    """Run a prompt directly against an Ollama model via CLI.

    Args:
        model: Target model name.
        prompt: Input text prompt.

    Returns:
        str: Generated output text.

    Raises:
        RuntimeError: If execution fails.
    """
    result = _run_command([
        "ollama",
        "run",
        model,
        prompt,
    ])

    if result.returncode != 0:
        write_log(
            level="ERROR",
            component=COMPONENT,
            action="run",
            message="Failed to run model",
            details={
                "model": model,
                "error": result.stderr.strip(),
            },
        )
        raise RuntimeError(
            result.stderr.strip() or f"Failed to run model: {model}"
        )

    write_log(
        level="INFO",
        component=COMPONENT,
        action="run",
        message="Model executed successfully",
        details={
            "model": model,
            "prompt_length": len(prompt),
        },
    )

    return result.stdout.strip()


def stop_model(model: str) -> str:
    """Unload or stop a running model from memory.

    Args:
        model: Running model name to stop.

    Returns:
        str: Output text from 'ollama stop'.

    Raises:
        RuntimeError: If stopping the model fails.
    """
    result = _run_command([
        "ollama",
        "stop",
        model,
    ])

    if result.returncode != 0:
        write_log(
            level="ERROR",
            component=COMPONENT,
            action="stop",
            message="Failed to stop model",
            details={
                "model": model,
                "error": result.stderr.strip(),
            },
        )
        raise RuntimeError(
            result.stderr.strip() or f"Failed to stop model: {model}"
        )

    write_log(
        level="INFO",
        component=COMPONENT,
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
) -> dict | None:
    """Load a model into VRAM/system memory if not already active.

    Args:
        model: Model name to load.
        keep_alive: Duration string to keep the model loaded (e.g., '10m',
            '1h'). Defaults to '10m'.

    Returns:
        dict | None: Generation API response dict if loaded, or None if the
            model was already loaded.

    Raises:
        ValueError: If model name is empty.
        RuntimeError: If checking status or loading the model fails.
    """
    if not model.strip():
        raise ValueError("Model name is required")

    # Check whether the model is already loaded.
    try:
        with urllib.request.urlopen(
            "http://127.0.0.1:11434/api/ps",
            timeout=5,
        ) as response:
            data = json.loads(response.read().decode("utf-8"))
    except (
        urllib.error.URLError,
        TimeoutError,
        OSError,
        json.JSONDecodeError,
    ) as error:
        write_log(
            level="ERROR",
            component=COMPONENT,
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

    # Ollama reports loaded models with an explicit tag, so a bare name such as
    # "llama3" must also match the ":latest" form it was loaded under.
    for item in data.get("models", []):
        loaded_model = item.get("name", "").strip()

        if loaded_model == requested_model or (
            ":" not in requested_model
            and loaded_model == f"{requested_model}:latest"
        ):
            write_log(
                level="INFO",
                component=COMPONENT,
                action="load",
                message="Model is already loaded",
                details={
                    "model": model,
                },
            )
            return None

    # Model is not loaded. Load it into memory.
    payload = {
        "model": model,
        "prompt": "",
        "stream": False,
        "keep_alive": keep_alive,
    }

    request = urllib.request.Request(
        "http://127.0.0.1:11434/api/generate",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            raw = response.read()
    except (urllib.error.URLError, TimeoutError, OSError) as error:
        write_log(
            level="ERROR",
            component=COMPONENT,
            action="load",
            message="Failed to load model",
            details={
                "model": model,
                "error": str(error),
            },
        )
        raise RuntimeError(f"Failed to load model: {model}") from error

    try:
        result = json.loads(raw.decode("utf-8"))
    except json.JSONDecodeError as error:
        raise RuntimeError("Ollama returned invalid JSON") from error

    if "error" in result:
        write_log(
            level="ERROR",
            component=COMPONENT,
            action="load",
            message="Failed to load model",
            details={
                "model": model,
                "error": result["error"],
            },
        )
        raise RuntimeError(result["error"])

    write_log(
        level="INFO",
        component=COMPONENT,
        action="load",
        message="Model loaded successfully",
        details={
            "model": model,
            "keep_alive": keep_alive,
        },
    )

    return result


def list_running_models() -> str:
    """List currently active running models via CLI.

    Returns:
        str: Output text from 'ollama ps'.
    """
    result = _run_command(["ollama", "ps"])
    return result.stdout.strip()


def configure_model(
    source_model: str,
    target_model: str,
    parameters: dict,
) -> str:
    """Create a new configured model from an existing source model.

    The source model is not modified.

    Args:
        source_model: Existing model name to base configuration on.
        target_model: New target model identifier name to create.
        parameters: Dictionary of Ollama Modelfile PARAMETER key-values.

    Returns:
        str: Output text from 'ollama create'.

    Raises:
        ValueError: If model names or parameters are empty/missing.
        TypeError: If parameters is not a dictionary.
        RuntimeError: If Ollama fails to create the configured model.
    """
    if not source_model.strip():
        raise ValueError("Source model name is required")

    if not target_model.strip():
        raise ValueError("Target model name is required")

    if not parameters:
        raise ValueError("At least one configuration parameter is required")

    if not isinstance(parameters, dict):
        raise TypeError("Parameters must be a dictionary")

    modelfile_parameters = []

    for key, value in parameters.items():
        if value is None:
            continue
        modelfile_parameters.append(f"PARAMETER {key} {value}")

    if not modelfile_parameters:
        raise ValueError("No valid configuration parameters found")

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

        result = _run_command([
            "ollama",
            "create",
            target_model,
            "-f",
            str(modelfile),
        ])

    if result.returncode != 0:
        write_log(
            level="ERROR",
            component=COMPONENT,
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
        component=COMPONENT,
        action="configure",
        message="Configured model created successfully",
        details={
            "source_model": source_model,
            "target_model": target_model,
            "parameters": parameters,
        },
    )

    return result.stdout.strip()
