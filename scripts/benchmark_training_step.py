#!/usr/bin/env python3
import argparse
import csv
import json
import os
import platform
import re
import signal
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


MODELS = {
    "257m": {"layers": 12, "hidden": 1024, "ffn": 2816, "heads": 8},
    "500m": {"layers": 18, "hidden": 1280, "ffn": 3584, "heads": 20},
}
PRECISIONS = ("bf16", "fp8_act", "full_fp8")
OPTIMIZERS = ("adam", "ademamix", "muon", "soap")
DEFAULT_BATCHES = (2, 4, 8, 16, 32, 64)
ITERATION_RE = re.compile(
    r"iteration\s+(\d+)/\s*\d+.*elapsed time per iteration \(ms\):\s*([0-9.]+)"
)
PARAMETER_RE = re.compile(r"number of parameters.*?:\s*([0-9]+)")
MEMORY_RE = re.compile(
    r"stage4 peak memory bytes: allocated=(\d+) reserved=(\d+)"
)
OOM_MARKERS = (
    "CUDA out of memory",
    "torch.OutOfMemoryError",
    "CUBLAS_STATUS_ALLOC_FAILED",
    "NVTE_ERROR_CUDA_ERROR",
)


def csv_values(value, allowed=None, cast=str):
    result = [cast(item) for item in value.split(",") if item]
    if allowed is not None:
        unknown = sorted(set(result) - set(allowed))
        if unknown:
            raise argparse.ArgumentTypeError(f"unsupported values: {unknown}")
    return result


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default="257m,500m")
    parser.add_argument("--precisions", default=",".join(PRECISIONS))
    parser.add_argument("--optimizers", default=",".join(OPTIMIZERS))
    parser.add_argument("--batches", default=",".join(map(str, DEFAULT_BATCHES)))
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--measure-steps", type=int, default=50)
    parser.add_argument("--timeout-seconds", type=int, default=3600)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(os.environ.get("BENCHMARK_OUTPUT_DIR", "/home/jovyan/hmoe-cloud/step-time")),
    )
    parser.add_argument("--rerun", action="store_true")
    args = parser.parse_args()
    args.models = csv_values(args.models, MODELS)
    args.precisions = csv_values(args.precisions, PRECISIONS)
    args.optimizers = csv_values(args.optimizers, OPTIMIZERS)
    args.batches = csv_values(args.batches, DEFAULT_BATCHES, int)
    if args.warmup_steps < 1 or args.measure_steps < 1:
        parser.error("warmup and measurement lengths must be positive")
    return args


def hardware_metadata():
    command = [
        "nvidia-smi",
        "--query-gpu=name,memory.total,driver_version",
        "--format=csv,noheader,nounits",
    ]
    try:
        gpu = subprocess.check_output(command, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        gpu = "unavailable"
    try:
        import torch

        torch_version = torch.__version__
        cuda_version = torch.version.cuda
    except ImportError:
        torch_version = "unavailable"
        cuda_version = "unavailable"
    return {
        "gpu": gpu,
        "hostname": platform.node(),
        "torch": torch_version,
        "cuda": cuda_version,
    }


def build_command(root, model_name, precision, optimizer, batch, warmup, measured):
    model = MODELS[model_name]
    total_steps = warmup + measured
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc-per-node",
        "1",
        "stage4/pretrain_gpt.py",
        "--optimizer-state-precision",
        "fp8" if precision == "full_fp8" else "fp32",
        "--num-layers",
        str(model["layers"]),
        "--hidden-size",
        str(model["hidden"]),
        "--ffn-hidden-size",
        str(model["ffn"]),
        "--num-attention-heads",
        str(model["heads"]),
        "--seq-length",
        "1024",
        "--max-position-embeddings",
        "1024",
        "--position-embedding-type",
        "rope",
        "--rotary-percent",
        "1.0",
        "--swiglu",
        "--normalization",
        "RMSNorm",
        "--norm-epsilon",
        "1e-5",
        "--disable-bias-linear",
        "--hidden-dropout",
        "0.0",
        "--attention-dropout",
        "0.0",
        "--make-vocab-size-divisible-by",
        "128",
        "--untie-embeddings-and-output-weights",
        "--tensor-model-parallel-size",
        "1",
        "--pipeline-model-parallel-size",
        "1",
        "--bf16",
        "--transformer-impl",
        "transformer_engine",
        "--optimizer",
        optimizer,
        "--adam-beta1",
        "0.9",
        "--adam-beta2",
        "0.999" if optimizer == "ademamix" else "0.99",
        "--adam-eps",
        "1e-8",
        "--muon-momentum",
        "0.95",
        "--muon-scale-mode",
        "spectral",
        "--muon-extra-scale-factor",
        "0.2",
        "--muon-coefficient-type",
        "quintic",
        "--muon-num-ns-steps",
        "5",
        "--muon-fp32-matmul-prec",
        "medium",
        "--lr",
        "1e-3",
        "--min-lr",
        "1e-3",
        "--lr-decay-style",
        "constant",
        "--lr-decay-iters",
        str(total_steps),
        "--weight-decay",
        "0.1",
        "--clip-grad",
        "1.0",
        "--micro-batch-size",
        str(batch),
        "--global-batch-size",
        str(batch),
        "--train-iters",
        str(total_steps),
        "--mock-data",
        "--num-workers",
        "0",
        "--tokenizer-type",
        "NullTokenizer",
        "--vocab-size",
        "50257",
        "--null-tokenizer-eod-id",
        "50256",
        "--null-tokenizer-pad-id",
        "-1",
        "--seed",
        "1234",
        "--init-method-std",
        "0.02",
        "--eval-interval",
        "1000000",
        "--eval-iters",
        "0",
        "--log-interval",
        "1",
    ]
    if precision != "bf16":
        command.extend(
            [
                "--fp8-format",
                "hybrid",
                "--fp8-recipe",
                "delayed",
                "--fp8-amax-history-len",
                "1",
                "--fp8-amax-compute-algo",
                "most_recent",
            ]
        )
    return command


def write_summary(output_dir, results, metadata):
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "metadata": metadata,
        "results": results,
    }
    (output_dir / "results.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    fields = [
        "model",
        "precision",
        "optimizer",
        "batch_size",
        "status",
        "mean_step_ms",
        "median_step_ms",
        "stdev_step_ms",
        "min_step_ms",
        "max_step_ms",
        "samples",
        "parameter_count",
        "peak_allocated_bytes",
        "peak_reserved_bytes",
        "return_code",
        "log",
        "message",
    ]
    with (output_dir / "results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for result in results:
            writer.writerow({field: result.get(field) for field in fields})


def result_key(result):
    return (
        result["model"],
        result["precision"],
        result["optimizer"],
        result["batch_size"],
    )


def load_previous(output_dir):
    path = output_dir / "results.json"
    if not path.exists():
        return [], {}
    payload = json.loads(path.read_text())
    return payload.get("results", []), payload.get("metadata", {})


def run_one(root, output_dir, args, model, precision, optimizer, batch):
    stem = f"{model}_{precision}_{optimizer}_bs{batch}"
    log_path = output_dir / "logs" / f"{stem}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = build_command(
        root,
        model,
        precision,
        optimizer,
        batch,
        args.warmup_steps,
        args.measure_steps,
    )
    env = os.environ.copy()
    source_paths = [
        str(root / "third_party" / "Megatron-LM"),
        str(root / "third_party" / "emerging-optimizers"),
        str(root),
    ]
    env["PYTHONPATH"] = os.pathsep.join(source_paths)
    env["PYTHONUNBUFFERED"] = "1"
    env["STAGE4_REPORT_PEAK_MEMORY"] = "1"
    env.setdefault("NVTE_FP8_BLOCK_SCALING_FP32_SCALES", "1")

    started = time.monotonic()
    process = subprocess.Popen(
        command,
        cwd=root,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    try:
        output, _ = process.communicate(timeout=args.timeout_seconds)
        return_code = process.returncode
        timed_out = False
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            output, _ = process.communicate(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            output, _ = process.communicate()
        return_code = process.returncode
        timed_out = True
    log_path.write_text("COMMAND: " + " ".join(command) + "\n\n" + output)

    times = [
        float(match.group(2))
        for match in ITERATION_RE.finditer(output)
        if int(match.group(1)) > args.warmup_steps
    ][: args.measure_steps]
    parameter_matches = PARAMETER_RE.findall(output)
    memory_matches = MEMORY_RE.findall(output)
    base = {
        "model": model,
        "precision": precision,
        "optimizer": optimizer,
        "batch_size": batch,
        "return_code": return_code,
        "wall_time_seconds": round(time.monotonic() - started, 3),
        "parameter_count": int(parameter_matches[-1]) if parameter_matches else None,
        "peak_allocated_bytes": int(memory_matches[-1][0]) if memory_matches else None,
        "peak_reserved_bytes": int(memory_matches[-1][1]) if memory_matches else None,
        "log": str(log_path),
    }
    if any(marker.lower() in output.lower() for marker in OOM_MARKERS):
        return {**base, "status": "oom", "samples": len(times), "message": "CUDA OOM"}
    if timed_out:
        return {**base, "status": "timeout", "samples": len(times), "message": "timeout"}
    if return_code != 0:
        return {**base, "status": "error", "samples": len(times), "message": "non-zero exit"}
    if len(times) != args.measure_steps:
        return {
            **base,
            "status": "error",
            "samples": len(times),
            "message": f"expected {args.measure_steps} timed steps",
        }
    return {
        **base,
        "status": "ok",
        "mean_step_ms": round(statistics.mean(times), 4),
        "median_step_ms": round(statistics.median(times), 4),
        "stdev_step_ms": round(statistics.stdev(times), 4) if len(times) > 1 else 0.0,
        "min_step_ms": min(times),
        "max_step_ms": max(times),
        "samples": len(times),
        "raw_step_ms": times,
        "message": "",
    }


def main():
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    previous, previous_metadata = load_previous(args.output_dir)
    results_by_key = {result_key(result): result for result in previous}
    metadata = {**previous_metadata, **hardware_metadata()}
    metadata.update(
        {
            "warmup_steps": args.warmup_steps,
            "measure_steps": args.measure_steps,
            "sequence_length": 1024,
            "git_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=root, text=True
            ).strip(),
        }
    )

    for model in args.models:
        for precision in args.precisions:
            for optimizer in args.optimizers:
                stop_after_oom = False
                for batch in args.batches:
                    key = (model, precision, optimizer, batch)
                    if stop_after_oom:
                        result = {
                            "model": model,
                            "precision": precision,
                            "optimizer": optimizer,
                            "batch_size": batch,
                            "status": "skipped_after_oom",
                            "samples": 0,
                            "message": "larger than the first OOM batch",
                        }
                    elif key in results_by_key and not args.rerun:
                        print(f"SKIP {key}: already recorded", flush=True)
                        if results_by_key[key]["status"] == "oom":
                            stop_after_oom = True
                        continue
                    else:
                        print(f"RUN model={model} precision={precision} optimizer={optimizer} batch={batch}", flush=True)
                        result = run_one(
                            root, args.output_dir, args, model, precision, optimizer, batch
                        )
                        print(
                            f"RESULT model={model} precision={precision} optimizer={optimizer} "
                            f"batch={batch} status={result['status']} "
                            f"mean_step_ms={result.get('mean_step_ms')}",
                            flush=True,
                        )
                        if result["status"] == "oom":
                            stop_after_oom = True
                    results_by_key[key] = result
                    write_summary(
                        args.output_dir,
                        sorted(results_by_key.values(), key=result_key),
                        metadata,
                    )

    results = sorted(results_by_key.values(), key=result_key)
    write_summary(args.output_dir, results, metadata)
    failures = [result for result in results if result["status"] in {"error", "timeout"}]
    print(f"SUMMARY total={len(results)} failures={len(failures)} output={args.output_dir}", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
