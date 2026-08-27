# ModelSetupHub Core

Core backend toolkit for managing local AI environments, hardware discovery, Ollama runtime operations, benchmarking, Python environments, and resilient file downloads.


## Modules

- **System (`core.System`)**: Discovers hardware specifications including CPU details, system RAM, and GPU architectures/VRAM (NVIDIA, AMD, Intel).
- **Ollama Runtime & Models (`core.ollama`)**: Controls the Ollama background daemon, manages local model lifecycles (create, run, stop, remove), and runs parameter benchmarks with token-speed metrics.
- **Python Environment (`core.python`)**: Automates virtual environment (`venv`) lifecycle, package management via `pip`, script execution, and Windows Python installation discovery.
- **Download Manager (`core.download_manager`)**: Multi-file sequential downloader featuring pause/resume, retries, speed and ETA calculation, and cancellation support.
- **Logging (`core.logging`)**: Structured JSON logging system with component-level tagging and action tracking.


## Project Structure

```text
core/
├── system/
│   ├── hardware.py          # CPU, GPU, RAM detection
│   └── scanner.py           # System compatibility and capability scanner
├── ollama/
│   ├── runtime.py           # Service lifecycle (start/stop/status)
│   ├── model.py             # Model creation, loading, and execution
│   └── experiment.py        # Benchmarking and token throughput tests
├── python/
│   ├── environment.py       # Virtual environment management
│   ├── tools.py             # Pip operations and script execution
│   └── installer.py         # Python binary and registry discovery
├── download_manager/
│   ├── downloader.py        # Streamed HTTP/HTTPS downloader
│   └── manager.py           # Queue and progress manager
└── logging.py               # Structured event logging
```


## Requirements

- Python 3.10+
- Optional: Ollama installed for model management features
- Windows support


## Usage Examples

### Hardware Detection

```python
from core.System.hardware import get_hardware_info

hw = get_hardware_info()
print(f"CPU: {hw.get('cpu')}")
print(f"Memory: {hw.get('memory', {}).get('total_gb')} GB")
print(f"GPUs: {hw.get('gpus', [])}")
```

### Ollama Operations & Benchmarking

```python
from core.ollama import runtime, model, experiment

# Check and start Ollama runtime
if not runtime.get_status()["running"]:
    runtime.start()

# Run a model prompt
output = model.run_model("llama3", "Explain quantum computing in two sentences.")
print(output)

# Benchmark model configurations
result = experiment.run_test(
    model="llama3",
    prompts=["Hello, world!", "Summarize the solar system."],
    config={"temperature": 0.7, "num_ctx": 4096},
    name="baseline_test"
)
print("Avg Output Tokens/Sec:", result["summary"]["average_output_tokens_per_second"])
```

### Download Manager

```python
from core.download_manager.manager import DownloadManager

dm = DownloadManager(download_directory="data/downloads")
dm.add("https://example.com/model.gguf", "model.gguf")
dm.start()
dm.wait()
```

### Python Virtual Environment

```python
from core.python import environment, tools

# Create virtual environment and install packages
env_path = environment.create_environment("envs/ai_env")
tools.install_packages(["torch", "numpy"], environment=env_path)
```