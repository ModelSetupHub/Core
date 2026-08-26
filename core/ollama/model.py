import subprocess
import tempfile
from pathlib import Path

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


def list_running_models():
    """
    List currently running models.
    """

    result = run_command(
        ["ollama", "ps"]
    )

    return result.stdout.strip()


def configure_model(
    model: str,
    temperature: float | None = None,
    context_length: int | None = None,
):
    """
    Configure model parameters.
    """

    parameters = []

    if temperature is not None:
        parameters.append(
            f"PARAMETER temperature {temperature}"
        )

    if context_length is not None:
        parameters.append(
            f"PARAMETER num_ctx {context_length}"
        )

    if not parameters:
        raise ValueError(
            "At least one configuration parameter is required"
        )

    with tempfile.TemporaryDirectory() as temp_dir:
        modelfile = Path(temp_dir) / "Modelfile"

        content = (
            f"FROM {model}\n"
            + "\n".join(parameters)
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
                model,
                "-f",
                str(modelfile),
            ]
        )

    if result.returncode != 0:
        write_log(
            level="ERROR",
            component="ollama/model",
            action="configure",
            message="Failed to configure model",
            details={
                "model": model,
                "error": result.stderr.strip(),
            },
        )

        raise RuntimeError(
            result.stderr.strip()
            or f"Failed to configure model: {model}"
        )

    write_log(
        level="INFO",
        component="ollama/model",
        action="configure",
        message="Model configured successfully",
        details={
            "model": model,
            "temperature": temperature,
            "context_length": context_length,
        },
    )

    return result.stdout.strip()