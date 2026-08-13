from pathlib import Path

from scripts.benchmark_training_step import build_command


def test_muon_syrk_flag_is_forwarded_only_when_enabled():
    common = {
        "root": Path("."),
        "model_name": "500m",
        "precision": "fp8_act",
        "optimizer": "muon",
        "global_batch": 128,
        "micro_batch": 32,
        "data_parallel_size": 1,
        "warmup": 2,
        "measured": 2,
    }

    baseline = build_command(**common, muon_use_syrk=False)
    optimized = build_command(**common, muon_use_syrk=True)

    assert "--muon-use-syrk" not in baseline
    assert optimized.count("--muon-use-syrk") == 1


def test_muon_syrk_flag_is_not_forwarded_to_other_optimizers():
    command = build_command(
        root=Path("."),
        model_name="500m",
        precision="fp8_act",
        optimizer="adam",
        global_batch=128,
        micro_batch=32,
        data_parallel_size=1,
        warmup=2,
        measured=2,
        muon_use_syrk=True,
    )

    assert "--muon-use-syrk" not in command
