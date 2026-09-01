# MSHCore

MSHCore backend toolkit for preparing and running local AI environments on
Windows. It discovers hardware, drives the Ollama runtime and its models,
benchmarks model configurations, manages Python virtual environments, and
downloads large model files with resume support. Every operation is recorded to
a single structured execution log.

The package exposes plain functions and classes; there is no CLI entry point
and no HTTP server.

## Requirements

- Python 3.10 or newer
- `psutil` (used by the hardware scanner)
- Windows: several features are Windows-only, as noted per module below
- Ollama installed and on `PATH` for the runtime, model, and benchmark modules

`psutil` is not declared in `pyproject.toml`, so install it separately:

```
pip install psutil
```

## Installation

Install the package from the repository root:

```
pip install .
```

For local development, install it in editable mode:

```
pip install -e .
```

## Project Structure

```text
MSHCore/
├── system/
│   ├── hardware.py          # OS, CPU, memory, NVIDIA GPU, storage probes
│   └── scanner.py           # Aggregates the probes into one profile
├── ollama/
│   ├── runtime.py           # Service lifecycle, installer, Ollama log reading
│   ├── model.py             # Model add/remove/run/load/stop/configure
│   └── experiment.py        # Benchmarking and configuration comparison
├── python/
│   ├── environment.py       # Virtual environment lifecycle
│   ├── tools.py             # pip operations and script management
│   └── installer.py         # Python installer and registry discovery
├── download_manager/
│   ├── downloader.py        # Single-file resumable HTTP/HTTPS downloader
│   ├── sources.py           # Allowed download domains
│   └── manager.py           # Sequential queue, progress, cancellation
├── cancellation.py          # Cancellation tokens for downloads and benchmarks
└── logging.py               # Structured execution log
data/
└── executions.log           # Created on the first logged event
```

## Modules

### System (`MSHCore.system`)

`scanner.scan_system()` returns the full machine profile: OS identification,
CPU model and clocks, CPU feature flags, total and available memory, physical
RAM modules, NVIDIA GPUs with VRAM and CUDA version, and per-drive storage
capacity. The individual probes in `hardware.py` can also be called directly.

Detection coverage is not uniform across platforms. RAM module details and the
Windows edition/build come from PowerShell and WMI, so they return empty or
`"Unknown"` elsewhere. CPU instruction-set flags are read from `/proc/cpuinfo`
on Linux; on Windows only a coarse `x86-64` or `ARM64` marker is reported. GPU
detection uses `nvidia-smi`, so AMD and Intel GPUs are not reported.
`get_memory_channels()` always returns `"Unknown"`.

### Ollama (`MSHCore.ollama`)

`runtime` controls the daemon: `get_status()`, `start()`, `stop()`, and
`install(installer_path)` for running a downloaded Ollama installer. It also
reads Ollama's own log files through `list_ollama_logs()` and
`read_ollama_logs(file_name, start_line, end_line)`, which search
`%LOCALAPPDATA%\Ollama` and `~/.ollama/logs`.

`model` wraps the Ollama CLI and HTTP API: `list_models()`,
`show_model_info()`, `add_model()` (imports a local GGUF file),
`remove_model()`, `run_model()`, `stop_model()`, `load_model()` (preloads into
memory), `list_running_models()`, and `configure_model()` (derives a new model
with different `PARAMETER` values, leaving the source model untouched).

`experiment` benchmarks models. `run_test()` runs a list of prompts against one
temporary configuration and reports per-prompt durations and token rates plus a
summary; model loading happens before the timer and is excluded from the
results. `compare_tests()` runs the same prompts against several
configurations.

### Python (`MSHCore.python`)

`environment` creates and removes virtual environments and resolves an
environment's interpreter path. `tools` installs, uninstalls, and lists
packages with pip, and creates, reads, edits, deletes, and runs `.py` scripts.
Both accept an optional `environment` path; without it the running interpreter
is used.

`installer` runs a Windows Python installer in quiet mode and reports the
Python versions found in the registry, plus the running interpreter.

### Download Manager (`MSHCore.download_manager`)

`DownloadManager` processes a queue one file at a time on a background thread,
tracking bytes and transfer speed per item and retrying failures. Downloads are
resumable: partial data is written to a `.part` file and continued with an HTTP
range request. Pause, resume, skip, and cancel are supported, and on Windows an
interactive terminal session also accepts the `p`, `s`, and `c`/`q` keys for
pause/resume, skip, and cancel.

Downloads are restricted to a fixed domain whitelist — `ollama.com`,
`huggingface.co`, and `python.org` (with their `www.` forms). Any other host
raises `PermissionError`.

### Cancellation (`MSHCore.cancellation`)

`CancellationToken` is a thread-safe flag passed into a long-running operation.
Cancellation is cooperative: the operation stops at its next safe point, undoes
its own side effects, and raises `OperationCancelled`. It applies to benchmarks
and downloads only.

### Logging (`MSHCore.logging`)

Every module writes events to `data/executions.log`. Each entry is one line of
pipe-separated fields — timestamp, level, component, action, message — with a
JSON object holding the event details. Read them back with
`read_logs(level, component, action, line_count)`, and inspect the file's size
and entry count with `get_log_file_info()`.

## Usage

### Hardware scan

```python
from MSHCore.system.scanner import scan_system

profile = scan_system()
print(profile["cpu"]["model"])
print(profile["memory"]["total_gb"], "GB")
print(profile["gpu"]["count"], "GPU(s)")
```

### Ollama runtime and models

```python
from MSHCore.ollama import model, runtime

if not runtime.get_status()["running"]:
    runtime.start()

print(model.list_models())
print(model.run_model("llama3", "Explain quantum computing in two sentences."))
```

### Benchmarking

```python
from MSHCore.ollama import experiment

result = experiment.run_test(
    model="llama3",
    prompts=["Hello, world!", "Summarize the solar system."],
    config={"temperature": 0.7, "num_ctx": 4096},
    name="baseline",
)

print(result["summary"]["average_output_tokens_per_second"])
```

Comparing several configurations over the same prompts:

```python
comparison = experiment.compare_tests(
    model="llama3",
    prompts=["Summarize the solar system."],
    configurations=[
        {"name": "cold", "options": {"temperature": 0.0}},
        {"name": "warm", "options": {"temperature": 0.9}},
    ],
)
```

### Downloads

```python
from MSHCore.download_manager import DownloadManager

manager = DownloadManager(download_directory="data/downloads")
manager.add("https://python.org/ftp/python/3.12.0/python-3.12.0-amd64.exe")
manager.start()
manager.wait()

print(manager.get_status()["downloads"])
```

Pausing keeps the queue and the partial data, and resuming continues the active
file from where it stopped:

```python
manager.pause()
manager.resume()
manager.skip()                                  # abandon the current file
```

Cancelling ends the session. By default it also deletes what the session
produced — partial files and completed ones alike, since the queue is one unit
of work that did not finish. Files that existed before the session started are
left alone.

```python
manager.cancel(reason="Not needed after all")   # removes the session's files
manager.cancel(cleanup=False)                   # keeps them
manager.close()                                 # same as cancel(cleanup=False)
```

A cancelled or closed manager cannot be reused: `add()` and `start()` then
raise `SessionCancelled`. Create a new manager instead.

### Python environments

```python
from MSHCore.python import environment, tools

env = environment.create_environment("envs/ai_env")
tools.install_packages(["numpy"], environment=env)
print(tools.list_packages(environment=env))
```

### Cancelling a benchmark

```python
import threading

from MSHCore.cancellation import CancellationToken, OperationCancelled
from MSHCore.ollama import experiment

token = CancellationToken()
threading.Timer(30, lambda: token.cancel("Taking too long")).start()

try:
    experiment.run_test(
        model="llama3",
        prompts=["Summarize the solar system."] * 10,
        cancellation=token,
    )
except OperationCancelled as error:
    # Partial results are discarded and the model has been unloaded.
    print("Cancelled:", error)
```

### Reading the execution log

```python
from MSHCore.logging import get_log_file_info, read_logs

print(get_log_file_info())
for entry in read_logs(level="ERROR", line_count=20):
    print(entry["timestamp"], entry["component"], entry["message"])
```

## Configuration

There are no configuration files or environment variables. Behaviour is set
through function arguments and a few module-level constants:

| Setting | Location | Default |
| --- | --- | --- |
| Execution log path | `MSHCore/logging.py` | `data/executions.log` |
| Allowed download domains | `download_manager/sources.py` | Ollama, Hugging Face, python.org |
| Download chunk size | `Downloader.__init__` | 1 MB |
| Download retries | `Downloader`, `DownloadManager` | 3 |
| Download connect timeout | `Downloader.__init__` | 15 s |
| Download read timeout | `Downloader.__init__` | 30 s |
| Download directory | `DownloadManager.__init__` | `data/downloads` |
| Ollama start/stop timeouts | `MSHCore/ollama/runtime.py` | 15 s / 10 s |
| Model keep-alive | `model.load_model` | `10m` |

`ALLOWED_DOMAINS` in `MSHCore/download_manager/sources.py` is the single source of
truth: both validation points — `DownloadManager.add` when a file is queued and
`Downloader.download` when the transfer starts — read it from there.
