"""Othello trace task: generate a legal move sequence from the standard opening.

Dataset traces are generated deterministically and cached as compact NumPy arrays.
The cache is an implementation detail; callers interact through the ordinary task
batch API used by the other symbolic tasks.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import json
import os
from pathlib import Path
import random
from typing import Dict, List, Sequence, Tuple

import numpy as np

from tasks.common import (
    BOS_TOKEN,
    EOS_TOKEN,
    PAD_TOKEN,
    SEP_TOKEN,
    SymbolicBatch,
    build_batch_from_sequences,
    build_vocab,
    make_sequence,
)
from tasks.trace.common import format_legal_generation_metrics, trace_generation_metrics


BOARD_SIZE = 8
NUM_SQUARES = BOARD_SIZE * BOARD_SIZE
MAX_MOVES = 60
BLACK = 1
WHITE = -1
EMPTY = 0
OPENING_PREFIX = (28, 27, 35, 36)
MOVE_TOKEN_OFFSET = 4  # pad, bos, sep, eos
PAD_MOVE = NUM_SQUARES
DEFAULT_DATA_DIR = "data/othello"
DEFAULT_TRAIN_GAMES = 500_000
DEFAULT_VAL_GAMES = 512
DEFAULT_DATASET_SEED = 1_337
DEFAULT_PREPEND_OPENING = False
DATASET_VERSION = 4
PARALLEL_GENERATION_THRESHOLD = 2_048
MAX_DATASET_WORKERS = 8


def _build_rays() -> tuple[tuple[tuple[int, ...], ...], ...]:
    directions = (
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 1),
        (1, -1),
        (1, 0),
        (1, 1),
    )
    all_rays = []
    for square in range(NUM_SQUARES):
        row, col = divmod(square, BOARD_SIZE)
        square_rays = []
        for dr, dc in directions:
            ray = []
            r, c = row + dr, col + dc
            while 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE:
                ray.append(r * BOARD_SIZE + c)
                r += dr
                c += dc
            square_rays.append(tuple(ray))
        all_rays.append(tuple(square_rays))
    return tuple(all_rays)


RAYS = _build_rays()


def move_token(square_index: int) -> str:
    if not 0 <= square_index < NUM_SQUARES:
        raise ValueError(f"square_index must be in [0, {NUM_SQUARES})")
    return f"m{square_index}"


def required_block_size(
    *,
    othello_prepend_opening: bool = DEFAULT_PREPEND_OPENING,
    othello_train_games: int = DEFAULT_TRAIN_GAMES,
    othello_val_games: int = DEFAULT_VAL_GAMES,
    **_unused,
) -> int:
    _validate_dataset_sizes(othello_train_games, othello_val_games)
    prompt_tokens = len(OPENING_PREFIX) if othello_prepend_opening else 0
    return 2 + prompt_tokens + MAX_MOVES


def build_othello_vocab(
    *,
    othello_train_games: int = DEFAULT_TRAIN_GAMES,
    othello_val_games: int = DEFAULT_VAL_GAMES,
    **_unused,
) -> Tuple[List[str], Dict[str, int], Dict[int, str]]:
    _validate_dataset_sizes(othello_train_games, othello_val_games)
    tokens = [PAD_TOKEN, BOS_TOKEN, SEP_TOKEN, EOS_TOKEN]
    tokens.extend(move_token(square) for square in range(NUM_SQUARES))
    return build_vocab(tokens)


def _initial_board_flat() -> np.ndarray:
    board = np.zeros(NUM_SQUARES, dtype=np.int8)
    board[27] = WHITE
    board[28] = BLACK
    board[35] = BLACK
    board[36] = WHITE
    return board


def _flips_for_move(board: np.ndarray, square: int, player: int) -> list[int]:
    if int(board[square]) != EMPTY:
        return []
    opponent = -player
    flips: list[int] = []
    for ray in RAYS[square]:
        ray_flips: list[int] = []
        for target in ray:
            value = int(board[target])
            if value == opponent:
                ray_flips.append(target)
                continue
            if value == player and ray_flips:
                flips.extend(ray_flips)
            break
    return flips


def _legal_move_indices(board: np.ndarray, player: int) -> list[int]:
    return [square for square in range(NUM_SQUARES) if _flips_for_move(board, square, player)]


def _active_player_and_legal_move_indices(
    board: np.ndarray,
    player: int,
) -> tuple[int | None, list[int]]:
    legal = _legal_move_indices(board, player)
    if legal:
        return player, legal
    legal = _legal_move_indices(board, -player)
    if legal:
        return -player, legal
    return None, []


def _apply_move_flat(board: np.ndarray, square: int, player: int) -> np.ndarray:
    flips = _flips_for_move(board, square, player)
    if not flips:
        raise ValueError("move is not legal")
    result = board.copy()
    result[square] = player
    result[np.asarray(flips, dtype=np.int64)] = player
    return result


def random_game_trace64(
    seed: int | None = None,
    rng: np.random.Generator | None = None,
) -> list[int]:
    rng = rng or np.random.default_rng(seed)
    board = _initial_board_flat()
    player = BLACK
    trace: list[int] = []
    while True:
        active_player, legal_moves = _active_player_and_legal_move_indices(board, player)
        if active_player is None:
            break
        move = int(legal_moves[int(rng.integers(len(legal_moves)))])
        board = _apply_move_flat(board, move, active_player)
        trace.append(move)
        player = -active_player
    return trace


def generate_trace_dataset_arrays(n_games: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate a deterministic dataset independent of worker count."""
    if n_games < 1:
        raise ValueError("n_games must be at least 1")
    game_seeds = np.random.SeedSequence(seed).generate_state(n_games, dtype=np.uint64)
    workers = min(n_games, MAX_DATASET_WORKERS, os.cpu_count() or 1)
    if n_games >= PARALLEL_GENERATION_THRESHOLD and workers > 1:
        return _generate_trace_dataset_arrays_parallel(game_seeds, workers)
    return _generate_trace_dataset_arrays_from_seeds(game_seeds)


def _generate_trace_dataset_arrays_parallel(
    game_seeds: np.ndarray,
    workers: int,
) -> tuple[np.ndarray, np.ndarray]:
    seed_chunks = [chunk for chunk in np.array_split(game_seeds, workers) if len(chunk)]
    serializable_chunks = [chunk.tolist() for chunk in seed_chunks]
    try:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            chunks = list(executor.map(_generate_trace_dataset_chunk, serializable_chunks))
    except (OSError, PermissionError):
        return _generate_trace_dataset_arrays_from_seeds(game_seeds)
    return (
        np.concatenate([traces for traces, _lengths in chunks], axis=0),
        np.concatenate([lengths for _traces, lengths in chunks], axis=0),
    )


def _generate_trace_dataset_chunk(game_seeds: list[int]) -> tuple[np.ndarray, np.ndarray]:
    return _generate_trace_dataset_arrays_from_seeds(np.asarray(game_seeds, dtype=np.uint64))


def _generate_trace_dataset_arrays_from_seeds(
    game_seeds: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    n_games = int(len(game_seeds))
    traces = np.full((n_games, MAX_MOVES), PAD_MOVE, dtype=np.uint8)
    lengths = np.zeros(n_games, dtype=np.uint8)
    for row, game_seed in enumerate(game_seeds):
        trace = random_game_trace64(seed=int(game_seed))
        lengths[row] = len(trace)
        traces[row, : len(trace)] = np.asarray(trace, dtype=np.uint8)
    return traces, lengths


def _metadata_payload(train_games: int, val_games: int, seed: int) -> dict[str, int]:
    return {
        "dataset_version": DATASET_VERSION,
        "train_games": int(train_games),
        "val_games": int(val_games),
        "dataset_seed": int(seed),
        "max_moves": MAX_MOVES,
    }


def ensure_othello_datasets(
    *,
    othello_data_dir: str,
    othello_train_games: int,
    othello_val_games: int,
    othello_dataset_seed: int,
) -> None:
    _validate_dataset_sizes(othello_train_games, othello_val_games)
    data_dir = Path(othello_data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = data_dir / "metadata.json"
    expected = _metadata_payload(othello_train_games, othello_val_games, othello_dataset_seed)
    if metadata_path.exists():
        try:
            current = json.loads(metadata_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            current = None
        if current == expected and all(
            (data_dir / f"{split}_{kind}.npy").exists()
            for split in ("train", "val")
            for kind in ("traces", "lengths")
        ):
            return

    for split, n_games, seed in (
        ("train", othello_train_games, othello_dataset_seed),
        ("val", othello_val_games, othello_dataset_seed + 1),
    ):
        traces, lengths = generate_trace_dataset_arrays(n_games, seed)
        _atomic_save_array(data_dir / f"{split}_traces.npy", traces)
        _atomic_save_array(data_dir / f"{split}_lengths.npy", lengths)
    metadata_temp = metadata_path.with_name(f".{metadata_path.name}.{os.getpid()}.tmp")
    metadata_temp.write_text(json.dumps(expected, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    metadata_temp.replace(metadata_path)
    _DATASET_CACHE.clear()


def _atomic_save_array(path: Path, array: np.ndarray) -> None:
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temp.open("wb") as handle:
        np.save(handle, array, allow_pickle=False)
    temp.replace(path)


@dataclass(frozen=True)
class OthelloTraceDataset:
    traces: np.ndarray
    lengths: np.ndarray

    def __len__(self) -> int:
        return int(self.traces.shape[0])

    def sample_trace(self, rng: random.Random) -> list[int]:
        row = rng.randrange(len(self))
        length = int(self.lengths[row])
        return [int(value) for value in self.traces[row, :length]]


_DATASET_CACHE: dict[tuple[str, str, int, int, int], OthelloTraceDataset] = {}


def load_othello_dataset(
    *,
    split: str,
    othello_data_dir: str,
    othello_train_games: int,
    othello_val_games: int,
    othello_dataset_seed: int,
) -> OthelloTraceDataset:
    if split not in {"train", "val"}:
        raise ValueError("split must be 'train' or 'val'")
    key = (
        str(Path(othello_data_dir).resolve()),
        split,
        int(othello_train_games),
        int(othello_val_games),
        int(othello_dataset_seed),
    )
    cached = _DATASET_CACHE.get(key)
    if cached is not None:
        return cached
    ensure_othello_datasets(
        othello_data_dir=othello_data_dir,
        othello_train_games=othello_train_games,
        othello_val_games=othello_val_games,
        othello_dataset_seed=othello_dataset_seed,
    )
    data_dir = Path(othello_data_dir)
    dataset = OthelloTraceDataset(
        traces=np.load(data_dir / f"{split}_traces.npy", mmap_mode="r", allow_pickle=False),
        lengths=np.load(data_dir / f"{split}_lengths.npy", mmap_mode="r", allow_pickle=False),
    )
    _DATASET_CACHE[key] = dataset
    return dataset


def sample_othello_example(
    stoi: Dict[str, int],
    rng: random.Random,
    *,
    split: str = "train",
    othello_data_dir: str = DEFAULT_DATA_DIR,
    othello_train_games: int = DEFAULT_TRAIN_GAMES,
    othello_val_games: int = DEFAULT_VAL_GAMES,
    othello_dataset_seed: int = DEFAULT_DATASET_SEED,
    othello_prepend_opening: bool = DEFAULT_PREPEND_OPENING,
) -> tuple[list[int], list[int], list[int]]:
    dataset = load_othello_dataset(
        split=split,
        othello_data_dir=othello_data_dir,
        othello_train_games=othello_train_games,
        othello_val_games=othello_val_games,
        othello_dataset_seed=othello_dataset_seed,
    )
    trace = dataset.sample_trace(rng)
    prompt = [stoi[move_token(square)] for square in OPENING_PREFIX] if othello_prepend_opening else []
    answer = [stoi[move_token(square)] for square in trace]
    answer.extend([stoi[PAD_TOKEN]] * (MAX_MOVES - len(trace)))
    return prompt, answer, trace


def build_othello_batch(
    batch_size: int,
    stoi: Dict[str, int],
    device=None,
    rng: random.Random | None = None,
    *,
    split: str = "train",
    othello_data_dir: str = DEFAULT_DATA_DIR,
    othello_train_games: int = DEFAULT_TRAIN_GAMES,
    othello_val_games: int = DEFAULT_VAL_GAMES,
    othello_dataset_seed: int = DEFAULT_DATASET_SEED,
    othello_prepend_opening: bool = DEFAULT_PREPEND_OPENING,
) -> SymbolicBatch:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    rng = rng or random.Random()
    rows = []
    for _ in range(batch_size):
        prompt, answer, _trace = sample_othello_example(
            stoi,
            rng,
            split=split,
            othello_data_dir=othello_data_dir,
            othello_train_games=othello_train_games,
            othello_val_games=othello_val_games,
            othello_dataset_seed=othello_dataset_seed,
            othello_prepend_opening=othello_prepend_opening,
        )
        rows.append(make_sequence(prompt, answer, stoi))
    return build_batch_from_sequences(rows, pad_id=stoi[PAD_TOKEN], device=device)


def othello_generation_metrics(
    model,
    batch,
    args,
    *,
    inference_mode: str | None = None,
    **_unused,
) -> dict[str, float]:
    return trace_generation_metrics(
        model,
        batch,
        args,
        legality_check=lambda _prompt_tokens, generated_tokens: legal_prefix_length(generated_tokens),
        inference_mode=inference_mode,
    )


def format_othello_eval_metrics(metrics: dict[str, float]) -> str:
    return format_legal_generation_metrics(metrics)


def legal_move_token_ids_after_prefix(
    prefix_move_token_ids: Sequence[int],
) -> tuple[int, ...]:
    """Return the legal next-move token IDs after replaying a move prefix."""
    board, player = _replay_token_prefix(prefix_move_token_ids)
    _active_player, legal_moves = _active_player_and_legal_move_indices(board, player)
    return tuple(move + MOVE_TOKEN_OFFSET for move in legal_moves)


def legal_prefix_length(
    move_token_ids: Sequence[int],
    *,
    prefix_move_token_ids: Sequence[int] = (),
) -> tuple[int, bool]:
    """Return the number of valid generated positions and full-sequence legality.

    ``prefix_move_token_ids`` is replayed before generated moves are checked.
    This supports continuation evaluation from arbitrary points in a game while
    preserving the training serialization, where ``<sep>`` precedes all moves.

    Once the game is terminal, padding tokens are valid and count as consumed
    positions.  This makes a fully legal fixed-width trace score 1.0 regardless
    of how many actual moves its game contains.
    """
    board, player = _replay_token_prefix(prefix_move_token_ids)
    valid_positions = 0
    for token_id in move_token_ids:
        active_player, legal_moves = _active_player_and_legal_move_indices(board, player)
        if active_player is None:
            if int(token_id) != 0:
                return valid_positions, False
            valid_positions += 1
            continue
        move = token_id_to_square(token_id)
        if move is None or move not in legal_moves:
            return valid_positions, False
        board = _apply_move_flat(board, move, active_player)
        valid_positions += 1
        player = -active_player
    return valid_positions, True


def _replay_token_prefix(move_token_ids: Sequence[int]) -> tuple[np.ndarray, int]:
    board = _initial_board_flat()
    player = BLACK
    for token_id in move_token_ids:
        active_player, legal_moves = _active_player_and_legal_move_indices(board, player)
        if active_player is None:
            raise ValueError("Othello prefix continues after the game is terminal")
        move = token_id_to_square(token_id)
        if move is None or move not in legal_moves:
            raise ValueError("Othello prefix contains an illegal move")
        board = _apply_move_flat(board, move, active_player)
        player = -active_player
    return board, player


def token_id_to_square(token_id: int) -> int | None:
    move_index = int(token_id) - MOVE_TOKEN_OFFSET
    return move_index if 0 <= move_index < NUM_SQUARES else None


def _validate_dataset_sizes(train_games: int, val_games: int) -> None:
    if train_games < 1:
        raise ValueError("othello_train_games must be at least 1")
    if val_games < 1:
        raise ValueError("othello_val_games must be at least 1")


__all__ = [
    "DEFAULT_DATASET_SEED",
    "DEFAULT_DATA_DIR",
    "DEFAULT_PREPEND_OPENING",
    "DEFAULT_TRAIN_GAMES",
    "DEFAULT_VAL_GAMES",
    "MAX_MOVES",
    "build_othello_batch",
    "build_othello_vocab",
    "ensure_othello_datasets",
    "format_othello_eval_metrics",
    "legal_move_token_ids_after_prefix",
    "legal_prefix_length",
    "load_othello_dataset",
    "move_token",
    "othello_generation_metrics",
    "random_game_trace64",
    "required_block_size",
    "sample_othello_example",
]
