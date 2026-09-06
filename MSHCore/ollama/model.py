"""Ollama model lifecycle management, configuration, and execution."""

from collections.abc import Callable
import json
from pathlib import Path
import re
import subprocess
import tempfile
import threading
import urllib.error
import urllib.request

from MSHCore.cancellation import (
    POLL_INTERVAL,
    CancellationToken,
    OperationCancelled,
    log_cancelled,
)
from MSHCore.logging import write_log

COMPONENT = "ollama/model"


def _run_create(
    command: list[str],
    cancellation: CancellationToken,
) -> tuple[int, str, str]:
    """Run 'ollama create' as a watched subprocess.

    A buffered capture would leave no point where a cancellation could act,
    so the process is spawned, its output is read as it is produced, and a
    watcher thread kills it the moment the token is set.

    Args:
        command: The command line to run.
        cancellation: Token that kills the process when set.

    Returns:
        tuple[int, str, str]: The process's return code, its standard output
            and its standard error, both stripped.

    Raises:
        OperationCancelled: If the token is set during the import; the
            process is killed first, and waits for it to exit.
    """
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        stdin=subprocess.DEVNULL,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    def kill_on_cancel() -> None:
        """Kill the process as soon as the token is set."""
        while True:
            if cancellation.wait(POLL_INTERVAL):
                try:
                    process.kill()
                except Exception:  # pragma: no cover - already exited
                    pass
                return

            if process.poll() is not None:
                return

    watcher = threading.Thread(
        target=kill_on_cancel,
        name="ollama-add-cancel",
        daemon=True,
    )
    watcher.start()

    # The phases only ever move forward. Output arrives on two pipes and
    # ollama interleaves 'creating layer' with 'writing layer' for every
    # layer it builds, so later lines regularly name earlier phases — taking
    # them at face value is what made a finished phase flicker back to
    # running. A line that names the current phase or an earlier one is
    # dropped: the panel only hears about forward crossings.
    phase_index = -1

    def drain(stream, sink: list[str]) -> None:
        """Read one output stream to the end, appending each line."""
        buffer = ""

        try:
            while True:
                chunk = stream.read(4096)

                if not chunk:
                    break

                buffer += chunk
                *lines, buffer = re.split(r"[\r\n]+", buffer)

                for line in lines:
                    stripped = line.strip()

                    if stripped:
                        sink.append(stripped)

            stripped = buffer.strip()

            if stripped:
                sink.append(stripped)
        finally:
            stream.close()

    stdout_lines: list[str] = []
    stderr_lines: list[str] = []

    readers = [
        threading.Thread(target=drain, args=(process.stdout, stdout_lines)),
        threading.Thread(target=drain, args=(process.stderr, stderr_lines)),
    ]

    for reader in readers:
        reader.start()

    return_code = process.wait()

    for reader in readers:
        reader.join(timeout=5)

    watcher.join(timeout=1)

    cancellation.raise_if_cancelled()

    return (
        return_code,
        "\n".join(stdout_lines).strip(),
        "\n".join(stderr_lines).strip(),
    )


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
        # This process may be an MCP server whose stdin carries the JSON-RPC
        # stream. `ollama run` in particular reads stdin for an interactive
        # session, so a child inheriting it would consume the protocol's bytes.
        stdin=subprocess.DEVNULL,
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
    cancellation: CancellationToken | None = None,
) -> str:
    """Import and register a local GGUF/model file into Ollama.

    The import runs 'ollama create' and blocks until it finishes. There is
    deliberately no progress reporting: the caller shows a title and a
    status badge, and the copy is a single disk-bound operation whose
    completion the return value settles.

    Args:
        model_name: Desired model identifier name in Ollama.
        model_path: Local filesystem path to the model file.
        cancellation: Optional token that kills the import mid-flight. The
            process is killed, Ollama's partial state is discarded on its
            side, and nothing is left under the model's name.

    Returns:
        str: Ollama CLI output upon successful creation.

    Raises:
        FileNotFoundError: If the specified model file does not exist.
        RuntimeError: If Ollama fails to create the model.
        OperationCancelled: If the token is cancelled during the import.
    """
    model_file = Path(model_path).expanduser().resolve()

    if not model_file.is_file():
        raise FileNotFoundError(f"Model file not found: {model_file}")

    # One token for the whole import, so a caller without one still gets
    # uncancellable-by-default behaviour with a single code path.
    token = cancellation if cancellation is not None else CancellationToken()

    with tempfile.TemporaryDirectory() as temp_dir:
        modelfile = Path(temp_dir) / "Modelfile"
        modelfile.write_text(
            f"FROM {model_file}\n",
            encoding="utf-8",
        )

        try:
            return_code, stdout, stderr = _run_create(
                [
                    "ollama",
                    "create",
                    model_name,
                    "-f",
                    str(modelfile),
                ],
                cancellation=token,
            )
        except OperationCancelled:
            # Killing the process is the whole undo: Ollama discards an
            # interrupted import, so nothing is left under the model's name
            # to clean up. The log entry is the only lasting trace.
            log_cancelled(
                component=COMPONENT,
                action="add",
                message="Model import cancelled",
                details={
                    "model": model_name,
                    "path": str(model_file),
                    "reason": token.reason,
                },
            )
            raise

    if return_code != 0:
        write_log(
            level="ERROR",
            component=COMPONENT,
            action="add",
            message="Failed to add model",
            details={
                "model": model_name,
                "error": stderr,
            },
        )
        raise RuntimeError(stderr or f"Failed to add model: {model_name}")

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

    return stdout


def delete_model_file(model_path: str) -> str:
    """Delete a local GGUF model weights file from disk.

    Only files ending in '.gguf' (case-insensitive) are accepted, so a
    mistyped path cannot remove anything else on disk. The deletion is
    permanent: there is no recycle bin and no undo. Deleting the file does
    not touch Ollama itself — a model already registered from it keeps
    working, since Ollama copied the weights into its own store; use
    remove_model to delete that copy.

    Args:
        model_path: Local filesystem path to the model file.

    Returns:
        str: The deleted file's absolute path.

    Raises:
        ValueError: If the path does not end in '.gguf'.
        FileNotFoundError: If the file does not exist.
    """
    model_file = Path(model_path).expanduser().resolve()

    if model_file.suffix.lower() != ".gguf":
        raise ValueError(f"Not a GGUF file: {model_file}")

    if not model_file.is_file():
        raise FileNotFoundError(f"Model file not found: {model_file}")

    size_bytes = model_file.stat().st_size

    model_file.unlink()

    write_log(
        level="WARNING",
        component=COMPONENT,
        action="delete",
        message="Model file deleted",
        details={
            "path": str(model_file),
            "size_bytes": size_bytes,
        },
    )

    return str(model_file)


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
