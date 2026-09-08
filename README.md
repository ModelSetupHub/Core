# MSHCore

**The engine behind ModelSetupHub — a Python library for setting up, running and measuring local AI models.**

English · [فارسی](README.fa.md)

MSHCore does the actual work of getting an AI model running on your own machine. It reads the hardware you have, manages
the Ollama runtime and your local models, measures how fast different generation settings really are, prepares Python
environments, and downloads large model files without falling over halfway. Everything it does is recorded in a single
execution log.

This repository is a library and nothing else: you import it and call functions. There is no command-line tool, no
server, and no user interface here — those live in the two repositories built on top of it.

## Where this fits

ModelSetupHub is three repositories over one engine. This is the engine.

| Repository | What it is | Reach for it when |
| --- | --- | --- |
| **Core** (this one) | The Python library that does the work | You are writing your own scripts or automation |
| [WebApp](https://github.com/ModelSetupHub/WebApp) | A local web dashboard over the same functions | You want a graphical interface and full manual control |
| [MCPServer](https://github.com/ModelSetupHub/MCPServer) | An MCP server that hands the same tools to an AI agent | You would rather describe the goal in plain language |

Both front ends include this repository as a git submodule, so a fix made here reaches both of them.

## What it does

**Hardware discovery.** One call returns a full profile of the machine: operating system, processor model and clock
speeds, installed memory modules, NVIDIA GPUs with their VRAM and CUDA version, and every drive with its capacity and
free space. This is the figure that decides which models you can realistically load.

**Ollama runtime and models.** Check whether Ollama is installed and running, start and stop the local service, install
Ollama itself from an installer, and read Ollama's own log files when something fails and the reason is buried in them.
On the model side: list what is installed and what is currently held in memory, inspect a model, run a prompt, preload
a model, unload it, register a local GGUF file, delete a model or its weights file, and create a configured copy of a
model with different parameters — the original is never modified.

**Benchmarking.** The part most tooling leaves out. The same prompts run through one model or several under one
configuration or several — four modes cover every combination, including a tournament where each model races its own
parameter set. Every mode returns per-prompt timings, time-to-first-token, token counts and generation rates, plus VRAM
and GPU readings, averaged over repetitions with the run-to-run noise reported beside the means. Each comparison ends
with a significance verdict that says whether the gap between the top two is real or could be coincidence, and finished
runs are saved into a local benchmark history that can be listed, re-read and deleted. The model is loaded before timing
starts, so load time never contaminates the numbers.

**Python environments.** Create and remove virtual environments, install and list packages with pip, and write, read,
edit, delete and run Python scripts — either in the current interpreter or inside a specific environment. There is also
a helper that installs Python from an official installer on Windows and reports which versions are already present.

**Downloads.** A queue that fetches large files one at a time on a background thread, tracking bytes and transfer speed
per file. Transfers survive interruption: partial data is kept and continued rather than restarted, and the queue can be
paused, resumed, skipped past, or cancelled. An existing file is never overwritten — a colliding name is saved under a
numbered variant, so two sources publishing the same filename both arrive. Downloads are restricted to a fixed list of
trusted hosts — Ollama, Hugging Face and python.org — and anything else is refused.

Two things run underneath all of the above. Every operation appends a structured entry to one execution log, which can
be read back and filtered afterwards, so there is always a record of what ran and what it did. And the two operations
long enough to be worth interrupting — downloads and benchmarks — can be cancelled while they run: they stop at a safe
point and undo what they created rather than leaving half-finished work behind.

## Requirements

- Python 3.10 or newer
- Windows for the full feature set. Hardware detection relies on PowerShell and WMI, and GPU detection on `nvidia-smi`,
  so results are thinner on other platforms and AMD or Intel GPUs are not reported.

The one third-party dependency, `psutil`, is declared in `pyproject.toml` and installs automatically. It is also listed
in `requirements.txt` for anyone who wants the dependencies without the package.

## Installation

```bash
git clone https://github.com/ModelSetupHub/Core.git
cd Core
pip install .
```

For development, install in editable mode with `pip install -e .` instead.

## Using it

Everything is a plain function call. A hardware scan, then a speed measurement against a model served by Ollama:

```python
from MSHCore.system.scanner import scan_system
from MSHCore.benchmarks.ollama import benchmark_single

profile = scan_system()
print(profile["cpu"]["model"], profile["memory"]["total_gb"], "GB")

result = benchmark_single(
    model="llama3",
    prompts=["Summarize the solar system."],
)
test = result["tests"][0]
print(test["summary"]["average_output_tokens_per_second"], "tokens/sec")
```

Or a Python-based model, with no Ollama involved at all — fetch the weights, prepare an environment, and run it:

```python
from MSHCore.download_manager import DownloadManager
from MSHCore.python import environment, tools

manager = DownloadManager()  # defaults to %LOCALAPPDATA%\MSH\downloads
manager.add("https://huggingface.co/Ultralytics/YOLOv8/resolve/main/yolov8n.pt")
manager.start()
manager.wait()

env = environment.create_environment("envs/vision")
tools.install_packages(["ultralytics"], environment=env)
print(tools.run_script("scripts/detect.py", environment=env))
```

The working modules are grouped into six packages — `MSHCore.system`, `MSHCore.ollama`, `MSHCore.benchmarks`,
`MSHCore.python`, `MSHCore.download_manager` and `MSHCore.logging` — and every function's behaviour is documented in its
docstring.

## Good to know

- **There is no configuration file and no environment variable to set.** Behaviour is decided by the arguments you pass.
- **Three things live on disk, all under `%LOCALAPPDATA%\MSH`:** the execution log, saved benchmark runs and completed
  downloads. Everything else — a download queue, its progress, a comparison in flight — is memory only, and a cancelled
  session cannot be restarted: you create a new one.
- **Deleting a model is not recoverable.** Neither is cancelling a download queue with cleanup on, which removes the
  files that queue produced.
- **Only downloads and benchmarks can be cancelled mid-flight.** Installers and scans run to completion.

## Status and licence

Early but working, and under active development. Issues and pull requests are welcome — behaviour changes usually belong
here rather than in a front end, since both front ends only call into this library.

No licence file has been added yet. If you need one before using this in your own project, open an issue.
