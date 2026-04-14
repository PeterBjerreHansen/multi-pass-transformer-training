import random
import json
import subprocess
import sys
from pathlib import Path

import torch

from tasks.clrs_text import (
    BOS_ID,
    EOS_ID,
    ClrsTextExample,
    build_clrs_text_batch,
    decode_answer_suffix,
    decode_ids,
    encode_example,
    encode_text,
    filter_examples,
    split_trace_and_final,
)
from train_clrs_text import apply_presets
from argparse import Namespace


ROOT_DIR = Path(__file__).resolve().parents[1]


def test_byte_tokenizer_round_trips_clrs_text():
    text = "binary_search: key: [0.1 0.2], target: 0.2 | (1, 1)"
    ids = [BOS_ID] + encode_text(text) + [EOS_ID]
    assert decode_ids(ids) == text


def test_clrs_text_batch_masks_prompt_and_keeps_answer_metric():
    examples = [
        ClrsTextExample(
            question="minimum: key: [0.3 0.1], initial_trace: 0 trace | min:",
            answer="1 | 1",
            algo_name="minimum",
            length=2,
        )
    ]
    batch = build_clrs_text_batch(
        examples,
        batch_size=2,
        block_size=128,
        rng=random.Random(5),
    )

    assert batch.idx.shape == batch.targets.shape
    assert batch.idx.size(0) == 2
    for row in range(batch.idx.size(0)):
        prompt_len = int(batch.prompt_lengths[row].item())
        output_len = int(batch.output_lengths[row].item())
        assert torch.equal(batch.targets[row, : prompt_len - 1], torch.full((prompt_len - 1,), -1))
        assert not batch.metric_mask[row, : prompt_len - 1].any()
        assert batch.metric_mask[row, prompt_len - 1 : prompt_len - 1 + output_len].all()
        answer = decode_answer_suffix(batch.targets[row, prompt_len - 1 : prompt_len - 1 + output_len].tolist())
        assert answer == "1 | 1"


def test_clrs_text_filtering_and_answer_splitting():
    examples = [
        ClrsTextExample("q1", "trace | final", "minimum", 4),
        ClrsTextExample("q2", "answer_only", "binary_search", None),
        ClrsTextExample("q3", "bad", "kmp_matcher", 32),
    ]

    filtered = filter_examples(
        examples,
        algorithms=["minimum", "binary_search"],
        lengths=[4],
        block_size=64,
    )
    assert [example.algo_name for example in filtered] == ["minimum", "binary_search"]

    trace, final = split_trace_and_final("a | b | c")
    assert trace == "a | b"
    assert final == "c"

    encoded = encode_example(filtered[0])
    assert len(encoded.idx) <= 64


def test_clrs_presets_fill_missing_values_and_preserve_overrides():
    args = Namespace(
        model_size="medium",
        task_preset="easy",
        n_layer=None,
        n_head=2,
        n_embd=None,
        block_size=None,
        batch_size=None,
        train_steps=123,
        eval_interval=None,
        eval_batches=None,
        algorithms=None,
        max_train_examples=None,
        max_eval_examples=None,
    )

    apply_presets(args)

    assert args.n_layer == 6
    assert args.n_head == 2
    assert args.n_embd == 192
    assert args.algorithms == "minimum,binary_search"
    assert args.block_size == 768
    assert args.train_steps == 123


def test_clrs_text_training_entrypoint_runs_one_jsonl_step(tmp_path):
    jsonl_path = tmp_path / "clrs_tiny.jsonl"
    rows = [
        {
            "question": "minimum: key: [0.3 0.1], initial_trace: 0 trace | min:",
            "answer": "1 | 1",
            "algo_name": "minimum",
            "length": 2,
        },
        {
            "question": "binary_search: key: [0.1 0.2], target: 0.2, initial_trace: (0, 1) trace | (low, high):",
            "answer": "(1, 1) | (1, 1)",
            "algo_name": "binary_search",
            "length": 2,
        },
    ]
    jsonl_path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            "train_clrs_text.py",
            "--data-source",
            "jsonl",
            "--train-jsonl",
            str(jsonl_path),
            "--eval-jsonl",
            str(jsonl_path),
            "--architecture",
            "transformer",
            "--device",
            "cpu",
            "--n-layer",
            "1",
            "--n-head",
            "1",
            "--n-embd",
            "8",
            "--block-size",
            "256",
            "--batch-size",
            "1",
            "--train-steps",
            "1",
            "--eval-interval",
            "1",
            "--eval-batches",
            "1",
            "--max-groups",
            "1",
            "--max-train-examples",
            "2",
            "--max-eval-examples",
            "2",
        ],
        cwd=ROOT_DIR,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "task: clrs_text" in result.stdout
    assert "eval recompute" in result.stdout
