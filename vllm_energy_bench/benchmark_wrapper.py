import os
import time
import datetime
import subprocess
from pathlib import Path
from zeus.monitor import ZeusMonitor

script_dir = Path(__file__).resolve().parent

"""
Start server:
python -m vllm.entrypoints.openai.api_server --model /home/cybsbbbb/llm_project/checkpoints/Meta-Llama-3.1-8B 
--gpu-memory-utilization 0.8 
--tensor-parallel-size 1 
--max-num-batched-tokens 8192 
--dtype half 
--max-model-len 8192 
--host 0.0.0.0 
--port 8000
"""


def benchmark_energy_wrapper(benchmark_config):
    print(f"\nRunning benchmark for ShareGPT with config: {benchmark_config} ...\n")

    gpu_monitor = ZeusMonitor(gpu_indices=[0])
    benchmark_script = benchmark_config["benchmark_script"]
    model_path = benchmark_config["model_path"]
    dataset_path = benchmark_config["dataset_path"]
    output_dir = benchmark_config["output_dir"]

    request_rate = benchmark_config["request_rate"]
    concurrency = benchmark_config["concurrency"]
    input_len = benchmark_config["input_len"]
    output_len = benchmark_config["output_len"]

    command = [
        "python", benchmark_script,
        "--backend", "vllm",
        "--model", model_path,
        "--dataset-name", "sharegpt",
        "--dataset-path", dataset_path,
        "--num-prompts", "500",
        "--max-concurrency", str(concurrency),
        "--sharegpt-output-len", str(output_len),
        "--fixed-input-len", str(input_len),
        "--ignore-eos",
        "--disable-tqdm",
        "--request-rate", str(request_rate),
        "--profile"
    ]

    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    log_file = os.path.join(output_dir, f"sharegpt_con{concurrency}_res{request_rate}_in{input_len}_out{output_len}_{timestamp}.log")

    if "benchmark" in gpu_monitor.measurement_states:
        gpu_monitor.end_window("benchmark")

    try:
        gpu_monitor.begin_window("benchmark")
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        energy_metrics = gpu_monitor.end_window("benchmark")
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Benchmark failed for concurrency={concurrency}, input_len={input_len}")
        print(e.stderr)
        return -1

    with open(log_file, "w") as f:
        f.write(result.stdout)
    avg_power = energy_metrics.total_energy / energy_metrics.time if energy_metrics.time > 0 else 0

    with open(log_file, "w") as f:
        f.write(f"Total Energy: {energy_metrics.total_energy} J\n")
        f.write(f"Total Time: {energy_metrics.time} s\n")
        f.write(f"Average Power: {avg_power} W\n")
        f.write(f"Average Power: {energy_metrics} W\n")

    print(f"Saved GPU monitoring data to: {log_file}")
    return 0


if __name__ == "__main__":
    concurrency_values = [4, 8, 12, 16, 20, 24, 28, 32]
    request_rate_values = [1, 2, 4, 8]
    input_len_values = [80, 120, 160, 200, 240]
    output_len_values = [80, 120, 160, 200, 240, 280, 320]
    benchmark_script = f"{script_dir}/benchmark_serving.py"
    model_path = f"{script_dir}/../checkpoints/Meta-Llama-3.1-8B"
    dataset_path = f"{script_dir}/../benchmark_profile/ShareGPT_V3_unfiltered_cleaned_split.json"

    output_dir = f"{script_dir}/benchmark_output"
    os.makedirs(output_dir, exist_ok=True)

    benchmark_config = dict()
    benchmark_config["benchmark_script"] = benchmark_script
    benchmark_config["model_path"] = model_path
    benchmark_config["dataset_path"] = dataset_path
    benchmark_config["output_dir"] = output_dir
    for request_rate in request_rate_values:
        for concurrency in concurrency_values:
            for input_len in input_len_values:
                for output_len in output_len_values:
                    benchmark_config["request_rate"] = request_rate
                    benchmark_config["concurrency"] = concurrency
                    benchmark_config["input_len"] = input_len
                    benchmark_config["output_len"] = output_len

                    # perform benchmark
                    benchmark_energy_wrapper(benchmark_config)
