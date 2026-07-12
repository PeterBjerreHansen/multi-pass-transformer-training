from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import dataclass

from tasks.bbh import permutation, state_machine, tracking
from tasks.trace import othello, random_graph_walk


@dataclass(frozen=True)
class ExperimentPreset:
    description: str
    values: dict[str, object]


def _shared(
    *,
    task: str,
    smoke: bool,
    inference_mode: str,
    token_selection: str,
) -> dict[str, object]:
    n_pass = 3 if smoke else 4
    return {
        "task": task,
        "architecture": "transformer",
        "model_size": "tiny" if smoke else "small",
        "n_layer": 1 if smoke else None,
        "n_head": 1 if smoke else None,
        "n_embd": 16 if smoke else None,
        "n_pass": n_pass,
        "pass_loss_weights": [0.0] * (n_pass - 1) + [1.0] if smoke else [0.0, 0.0, 1.0, 1.0],
        "memory_update_gate": "off",
        "memory_gate_bias": -1.0,
        "inference_mode": inference_mode,
        "token_selection": token_selection,
        "batch_size": 1 if smoke else 64,
        "train_steps": 1 if smoke else 50_000,
        "eval_interval": 1 if smoke else 1_000,
        "eval_batches": 1 if smoke else 4,
        "lr": 1e-4 if smoke else 3e-4,
        "weight_decay": 0.0,
        "seed": 1337,
        "run_dir": None,
        "resume_from": None,
        "device": None,
        "block_size": None,
    }


TRACE_PRESETS: dict[str, ExperimentPreset] = {}

rgw_main = _shared(
    task="random_graph_walk",
    smoke=False,
    inference_mode="append_recurrent",
    token_selection="sample",
)
rgw_main.update(
    num_states=random_graph_walk.DEFAULT_NUM_STATES,
    label_pool_size=random_graph_walk.DEFAULT_LABEL_POOL_SIZE,
    max_level=32,
)
TRACE_PRESETS["random_graph_walk_main"] = ExperimentPreset(
    "Main random-graph-walk trace setup.",
    rgw_main,
)

rgw_smoke = _shared(
    task="random_graph_walk",
    smoke=True,
    inference_mode="append_recurrent",
    token_selection="argmax",
)
rgw_smoke.update(num_states=4, label_pool_size=4, max_level=2)
TRACE_PRESETS["random_graph_walk_smoke"] = ExperimentPreset(
    "Tiny random-graph-walk smoke setup.",
    rgw_smoke,
)

othello_main = _shared(
    task="othello",
    smoke=False,
    inference_mode="append_recurrent",
    token_selection="sample",
)
othello_main.update(
    batch_size=128,
    train_steps=500_000,
    eval_interval=5_000,
    eval_batches=1,
    othello_data_dir=othello.DEFAULT_DATA_DIR,
    othello_train_games=othello.DEFAULT_TRAIN_GAMES,
    othello_val_games=othello.DEFAULT_VAL_GAMES,
    othello_dataset_seed=othello.DEFAULT_DATASET_SEED,
    othello_prepend_opening=othello.DEFAULT_PREPEND_OPENING,
)
TRACE_PRESETS["othello_main"] = ExperimentPreset("Main Othello trace setup.", othello_main)

othello_smoke = _shared(
    task="othello",
    smoke=True,
    inference_mode="append_recurrent",
    token_selection="argmax",
)
othello_smoke.update(
    othello_data_dir="data/othello_smoke",
    othello_train_games=16,
    othello_val_games=8,
    othello_dataset_seed=9,
    othello_prepend_opening=False,
)
TRACE_PRESETS["othello_smoke"] = ExperimentPreset("Tiny Othello smoke setup.", othello_smoke)


BBH_PRESETS: dict[str, ExperimentPreset] = {}


def _add_bbh_pair(task: str, main_values: dict[str, object], smoke_values: dict[str, object]) -> None:
    main = _shared(task=task, smoke=False, inference_mode="recompute", token_selection="argmax")
    main.update(
        max_level=64,
        curriculum_start_level=main_values.pop("curriculum_start_level"),
        curriculum_threshold=0.95,
        review_easier_every=2,
        **main_values,
    )
    smoke = _shared(task=task, smoke=True, inference_mode="recompute", token_selection="argmax")
    smoke.update(
        max_level=2,
        curriculum_start_level=smoke_values.pop("curriculum_start_level"),
        curriculum_threshold=0.95,
        review_easier_every=2,
        **smoke_values,
    )
    BBH_PRESETS[f"{task}_main"] = ExperimentPreset(f"Main {task} curriculum setup.", main)
    BBH_PRESETS[f"{task}_smoke"] = ExperimentPreset(f"Tiny {task} smoke setup.", smoke)


_add_bbh_pair(
    "pointer_chasing",
    {"num_nodes": 32, "curriculum_start_level": 0},
    {"num_nodes": 4, "curriculum_start_level": 0},
)
_add_bbh_pair(
    "tracking",
    {"num_objects": tracking.DEFAULT_NUM_OBJECTS, "curriculum_start_level": 1},
    {"num_objects": 4, "curriculum_start_level": 1},
)
_add_bbh_pair(
    "permutation",
    {"num_objects": permutation.DEFAULT_NUM_OBJECTS, "curriculum_start_level": 1},
    {"num_objects": 4, "curriculum_start_level": 1},
)
_add_bbh_pair(
    "state_machine",
    {
        "num_states": state_machine.DEFAULT_NUM_STATES,
        "alphabet_size": state_machine.DEFAULT_ALPHABET_SIZE,
        "curriculum_start_level": 0,
    },
    {"num_states": 4, "alphabet_size": 2, "curriculum_start_level": 0},
)


def resolve_preset_args(
    raw_args: argparse.Namespace,
    presets: dict[str, ExperimentPreset],
    *,
    default_preset: str,
    parser: argparse.ArgumentParser,
) -> argparse.Namespace:
    overrides = vars(raw_args).copy()
    preset_name = str(overrides.pop("preset", default_preset))
    if preset_name not in presets:
        parser.error(f"unknown preset: {preset_name}")
    values = deepcopy(presets[preset_name].values)
    values.update(overrides)
    values["preset"] = preset_name
    return argparse.Namespace(**values)


def preset_help_text(presets: dict[str, ExperimentPreset]) -> str:
    return " ".join(f"{name}: {preset.description}" for name, preset in sorted(presets.items()))
