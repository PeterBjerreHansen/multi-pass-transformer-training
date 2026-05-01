"""Othello trace task: generate a random legal move trace from the fixed opening board.

Example sequence:
<bos> <sep> m19 m18 m26 m34 ... <eos>
"""

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import random

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
DATASET_VERSION = 3
DATASET_CHUNK_SIZE = 1_000
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
    rays = []
    for square in range(NUM_SQUARES):
        row, col = divmod(square, BOARD_SIZE)
        square_rays = []
        for dr, dc in directions:
            ray = []
            next_row = row + dr
            next_col = col + dc
            while 0 <= next_row < BOARD_SIZE and 0 <= next_col < BOARD_SIZE:
                ray.append(next_row * BOARD_SIZE + next_col)
                next_row += dr
                next_col += dc
            square_rays.append(tuple(ray))
        rays.append(tuple(square_rays))
    return tuple(rays)


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
    answer_len = MAX_MOVES
    return 2 + prompt_tokens + answer_len


def build_othello_vocab(
    *,
    othello_train_games: int = DEFAULT_TRAIN_GAMES,
    othello_val_games: int = DEFAULT_VAL_GAMES,
    **_unused,
) -> Tuple[List[str], Dict[str, int], Dict[int, str]]:
    _validate_dataset_sizes(othello_train_games, othello_val_games)
    tokens = [
        PAD_TOKEN,
        BOS_TOKEN,
        SEP_TOKEN,
        EOS_TOKEN,
    ]
    tokens.extend(move_token(square) for square in range(NUM_SQUARES))
    return build_vocab(tokens)


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
    rng = rng or random.Random()
    # Dataset generation/loading is intentionally NumPy-only CPU work. The
    # device argument only controls where the finished training tensors live.
    rows = []
    for _ in range(batch_size):
        prompt, answer, _ = sample_othello_example(
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
) -> dict[str, float | None]:
    return trace_generation_metrics(
        model,
        batch,
        args,
        legality_check=lambda _prompt_tokens, generated_tokens: legal_prefix_length(generated_tokens),
        inference_mode=inference_mode,
    )


def format_othello_eval_metrics(metrics: dict[str, float]) -> str:
    return format_legal_generation_metrics(metrics)


def legal_prefix_length(move_token_ids: Sequence[int]) -> tuple[int, bool]:
    board = _initial_board_flat()
    player = BLACK
    legal_steps = 0

    for token_id in move_token_ids:
        active_player, legal_moves = _active_player_and_legal_move_indices(board, player)
        if active_player is None:
            if int(token_id) == 0:
                continue
            return legal_steps, False
        move = token_id_to_square(token_id)
        if move is None:
            return legal_steps, False
        if move not in legal_moves:
            return legal_steps, False
        board = _apply_move_flat(board, move, active_player)
        legal_steps += 1
        player = -active_player

    return legal_steps, True


def token_id_to_square(token_id: int) -> int | None:
    move_index = int(token_id) - MOVE_TOKEN_OFFSET
    if 0 <= move_index < NUM_SQUARES:
        return move_index
    return None


def random_game_trace64(seed: int | None = None, rng: np.random.Generator | None = None) -> list[int]:
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
    if n_games < 1:
        raise ValueError("n_games must be at least 1")
    workers = min(n_games, MAX_DATASET_WORKERS, os.cpu_count() or 1)
    if n_games >= PARALLEL_GENERATION_THRESHOLD and workers > 1:
        return _generate_trace_dataset_arrays_parallel(n_games, seed, workers)

    return _generate_trace_dataset_arrays_sequential(n_games, seed)


def _generate_trace_dataset_arrays_parallel(n_games: int, seed: int, workers: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    chunk_sizes = [n_games // workers] * workers
    for idx in range(n_games % workers):
        chunk_sizes[idx] += 1
    chunk_seeds = rng.integers(0, 2**63 - 1, size=workers, dtype=np.int64).tolist()

    try:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            chunks = list(executor.map(_generate_trace_dataset_chunk, zip(chunk_sizes, chunk_seeds)))
    except (OSError, PermissionError):
        return _generate_trace_dataset_arrays_sequential(n_games, seed)

    traces = np.concatenate([chunk_traces for chunk_traces, _chunk_lengths in chunks], axis=0)
    lengths = np.concatenate([chunk_lengths for _chunk_traces, chunk_lengths in chunks], axis=0)

    return traces, lengths


def _generate_trace_dataset_chunk(args: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    n_games, seed = args
    return _generate_trace_dataset_arrays_sequential(n_games, seed)


def _generate_trace_dataset_arrays_sequential(n_games: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    traces = np.full((n_games, MAX_MOVES), PAD_MOVE, dtype=np.uint8)
    lengths = np.zeros(n_games, dtype=np.uint8)
    rng = np.random.default_rng(seed)

    for row in range(n_games):
        trace = random_game_trace64(rng=rng)
        lengths[row] = len(trace)
        traces[row, : len(trace)] = np.asarray(trace, dtype=np.uint8)

    return traces, lengths


def ensure_othello_datasets(
    *,
    othello_data_dir: str,
    othello_train_games: int,
    othello_val_games: int,
    othello_dataset_seed: int,
) -> None:
    if othello_train_games < 1:
        raise ValueError("othello_train_games must be at least 1")
    if othello_val_games < 1:
        raise ValueError("othello_val_games must be at least 1")

    data_dir = Path(othello_data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = data_dir / "metadata.json"
    required_metadata = {
        "dataset_version": DATASET_VERSION,
        "train_games": int(othello_train_games),
        "val_games": int(othello_val_games),
        "dataset_seed": int(othello_dataset_seed),
        "max_moves": MAX_MOVES,
        "storage_format": "dense_split_arrays",
    }

    if metadata_path.exists():
        current_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if current_metadata == required_metadata and _split_arrays_exist(data_dir, "train") and _split_arrays_exist(data_dir, "val"):
            return
        legacy_metadata = dict(required_metadata)
        legacy_metadata["dataset_version"] = 2
        legacy_metadata["chunk_size"] = DATASET_CHUNK_SIZE
        del legacy_metadata["storage_format"]
        if (
            current_metadata == legacy_metadata
            and _legacy_split_chunks_exist(data_dir, "train")
            and _legacy_split_chunks_exist(data_dir, "val")
        ):
            print("othello data: migrating legacy chunked dataset to dense split arrays", flush=True)
            _migrate_legacy_split_dataset(data_dir=data_dir, split="train", n_games=othello_train_games)
            _migrate_legacy_split_dataset(data_dir=data_dir, split="val", n_games=othello_val_games)
            metadata_path.write_text(json.dumps(required_metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            _DATASET_CACHE.clear()
            return

    print(f"othello data: generating {othello_train_games} train and {othello_val_games} val traces", flush=True)
    _generate_split_dataset(
        data_dir=data_dir,
        split="train",
        n_games=othello_train_games,
        seed=othello_dataset_seed,
    )
    _generate_split_dataset(
        data_dir=data_dir,
        split="val",
        n_games=othello_val_games,
        seed=othello_dataset_seed + 1,
    )

    metadata_path.write_text(json.dumps(required_metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _DATASET_CACHE.clear()


@dataclass
class OthelloTraceDataset:
    traces_path: Path
    lengths_path: Path
    traces: np.ndarray | None = None
    lengths: np.ndarray | None = None
    total_rows: int = 0

    def __post_init__(self):
        if self.traces is None:
            self.traces = np.load(self.traces_path)
        if self.lengths is None:
            self.lengths = np.load(self.lengths_path)
        if self.traces.shape[0] != self.lengths.shape[0]:
            raise ValueError("Othello traces and lengths must contain the same number of rows")
        if self.total_rows == 0:
            self.total_rows = int(self.traces.shape[0])

    def locate_row(self, global_row: int) -> int:
        if not 0 <= global_row < self.total_rows:
            raise IndexError(f"global_row must be in [0, {self.total_rows})")
        return int(global_row)

    def sample_trace(self, rng: random.Random) -> list[int]:
        row = self.locate_row(rng.randrange(self.total_rows))
        trace_len = int(self.lengths[row])
        return [int(move) for move in self.traces[row, :trace_len].tolist()]

def _split_array_paths(data_dir: Path, split: str) -> tuple[Path, Path]:
    return data_dir / f"{split}_traces.npy", data_dir / f"{split}_lengths.npy"


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
    cache_key = (
        split,
        str(Path(othello_data_dir).resolve()),
        int(othello_train_games),
        int(othello_val_games),
        int(othello_dataset_seed),
    )
    if cache_key in _DATASET_CACHE:
        return _DATASET_CACHE[cache_key]

    ensure_othello_datasets(
        othello_data_dir=othello_data_dir,
        othello_train_games=othello_train_games,
        othello_val_games=othello_val_games,
        othello_dataset_seed=othello_dataset_seed,
    )

    data_dir = Path(othello_data_dir)
    traces_path, lengths_path = _split_array_paths(data_dir, split)
    if not traces_path.exists() or not lengths_path.exists():
        raise ValueError(f"missing Othello {split} arrays in {data_dir}")
    dataset = OthelloTraceDataset(traces_path=traces_path, lengths_path=lengths_path)
    _DATASET_CACHE[cache_key] = dataset
    return dataset


def _split_arrays_exist(data_dir: Path, split: str) -> bool:
    traces_path, lengths_path = _split_array_paths(data_dir, split)
    return traces_path.exists() and lengths_path.exists()


def _generate_split_dataset(*, data_dir: Path, split: str, n_games: int, seed: int) -> None:
    traces, lengths = generate_trace_dataset_arrays(n_games, seed)
    _save_split_arrays(data_dir, split, traces, lengths)
    _remove_legacy_split_dir(data_dir / split)
    _log_split_progress(split, n_games, n_games, suffix="saved dense arrays")


def _save_split_arrays(data_dir: Path, split: str, traces: np.ndarray, lengths: np.ndarray) -> None:
    traces_path, lengths_path = _split_array_paths(data_dir, split)
    np.save(traces_path, traces)
    np.save(lengths_path, lengths)


def _legacy_split_chunks_exist(data_dir: Path, split: str) -> bool:
    split_dir = data_dir / split
    trace_paths = sorted(split_dir.glob("traces_*.npy"))
    length_paths = sorted(split_dir.glob("lengths_*.npy"))
    return bool(trace_paths) and len(trace_paths) == len(length_paths)


def _migrate_legacy_split_dataset(*, data_dir: Path, split: str, n_games: int) -> None:
    split_dir = data_dir / split
    trace_paths = sorted(split_dir.glob("traces_*.npy"))
    length_paths = sorted(split_dir.glob("lengths_*.npy"))
    if not trace_paths or len(trace_paths) != len(length_paths):
        raise ValueError(f"missing or incomplete legacy Othello {split} chunk files in {split_dir}")

    traces = np.full((n_games, MAX_MOVES), PAD_MOVE, dtype=np.uint8)
    lengths = np.zeros(n_games, dtype=np.uint8)
    offset = 0
    total_chunks = len(trace_paths)
    log_interval = max(1, total_chunks // 20)
    for chunk_idx, (trace_path, length_path) in enumerate(zip(trace_paths, length_paths), start=1):
        chunk_traces = np.load(trace_path)
        chunk_lengths = np.load(length_path)
        chunk_rows = int(chunk_traces.shape[0])
        traces[offset:offset + chunk_rows] = chunk_traces
        lengths[offset:offset + chunk_rows] = chunk_lengths
        offset += chunk_rows
        if chunk_idx == total_chunks or chunk_idx % log_interval == 0:
            _log_split_progress(split, offset, n_games, suffix=f"migrated chunk {chunk_idx}/{total_chunks}")
    if offset != n_games:
        raise ValueError(f"legacy Othello {split} chunks contained {offset} rows, expected {n_games}")

    _save_split_arrays(data_dir, split, traces, lengths)
    _remove_legacy_split_dir(split_dir)


def _remove_legacy_split_dir(split_dir: Path) -> None:
    if not split_dir.exists():
        return
    for pattern in ("traces_*.npy", "lengths_*.npy"):
        for path in split_dir.glob(pattern):
            path.unlink()
    try:
        split_dir.rmdir()
    except OSError:
        pass


def _log_split_progress(split: str, traces_done: int, total_traces: int, *, suffix: str) -> None:
    pct = 100.0 * traces_done / float(total_traces)
    print(f"othello data: {split} {traces_done}/{total_traces} traces ({pct:.1f}%, {suffix})", flush=True)
def initial_board() -> np.ndarray:
    return _initial_board_flat().reshape(BOARD_SIZE, BOARD_SIZE)


def active_player_and_legal_moves(board: np.ndarray, player: int) -> tuple[int | None, np.ndarray]:
    flat_board = _as_flat_board(board)
    legal_self = _legal_moves_flat(flat_board, player)
    if legal_self.any():
        return player, _reshape_like_board(legal_self, board)
    legal_opp = _legal_moves_flat(flat_board, -player)
    if legal_opp.any():
        return -player, _reshape_like_board(legal_opp, board)
    return None, np.zeros_like(board, dtype=bool)


def legal_moves(board: np.ndarray, player: int) -> np.ndarray:
    return _reshape_like_board(_legal_moves_flat(_as_flat_board(board), player), board)


def apply_move(board: np.ndarray, move_coordinate: tuple[int, int], player: int) -> np.ndarray:
    row, col = move_coordinate
    move = row * BOARD_SIZE + col
    return _reshape_like_board(_apply_move_flat(_as_flat_board(board), move, player), board)


def _initial_board_flat() -> np.ndarray:
    board = np.zeros(NUM_SQUARES, dtype=np.int8)
    board[28] = BLACK
    board[27] = WHITE
    board[35] = BLACK
    board[36] = WHITE
    return board


def _as_flat_board(board: np.ndarray) -> np.ndarray:
    board_array = np.asarray(board)
    if board_array.size != NUM_SQUARES:
        raise ValueError(f"board must contain {NUM_SQUARES} squares")
    return board_array.reshape(NUM_SQUARES)


def _reshape_like_board(flat: np.ndarray, board: np.ndarray) -> np.ndarray:
    if np.asarray(board).ndim == 1:
        return flat.copy()
    return flat.reshape(BOARD_SIZE, BOARD_SIZE)


def _active_player_and_legal_moves_flat(board: np.ndarray, player: int) -> tuple[int | None, np.ndarray]:
    active_player, legal_moves = _active_player_and_legal_move_indices(board, player)
    if active_player is not None:
        return active_player, _legal_move_mask(legal_moves)
    return None, np.zeros(NUM_SQUARES, dtype=bool)


def _legal_moves_flat(board: np.ndarray, player: int) -> np.ndarray:
    return _legal_move_mask(_legal_move_indices(board, player))


def _active_player_and_legal_move_indices(board: np.ndarray, player: int) -> tuple[int | None, list[int]]:
    legal_self = _legal_move_indices(board, player)
    if legal_self:
        return player, legal_self
    legal_opp = _legal_move_indices(board, -player)
    if legal_opp:
        return -player, legal_opp
    return None, []


def _legal_move_mask(legal_moves: Sequence[int]) -> np.ndarray:
    mask = np.zeros(NUM_SQUARES, dtype=bool)
    mask[list(legal_moves)] = True
    return mask


def _legal_move_indices(board: np.ndarray, player: int) -> list[int]:
    opp = -player
    legal = []

    for square in range(NUM_SQUARES):
        if board[square] != EMPTY:
            continue
        for ray in RAYS[square]:
            if len(ray) < 2 or board[ray[0]] != opp:
                continue
            for pos in ray[1:]:
                value = board[pos]
                if value == opp:
                    continue
                if value == player:
                    legal.append(square)
                break
            if legal and legal[-1] == square:
                break

    return legal


def _apply_move_flat(board: np.ndarray, move: int, player: int) -> np.ndarray:
    new_board = board.copy()
    new_board[move] = player
    opp = -player

    for ray in RAYS[move]:
        captured = []
        for pos in ray:
            value = new_board[pos]
            if value == opp:
                captured.append(pos)
                continue
            if value == player and captured:
                new_board[captured] = player
            break

    return new_board


def _validate_num_steps(num_steps: int) -> int:
    if not 1 <= num_steps <= MAX_MOVES:
        raise ValueError(f"num_steps must be in [1, {MAX_MOVES}]")
    return num_steps


def _validate_dataset_sizes(othello_train_games: int, othello_val_games: int) -> None:
    if othello_train_games < 1:
        raise ValueError("othello_train_games must be at least 1")
    if othello_val_games < 1:
        raise ValueError("othello_val_games must be at least 1")


__all__ = [
    "BLACK",
    "BOARD_SIZE",
    "DEFAULT_DATASET_SEED",
    "DEFAULT_DATA_DIR",
    "DEFAULT_TRAIN_GAMES",
    "DEFAULT_VAL_GAMES",
    "MAX_MOVES",
    "OPENING_PREFIX",
    "apply_move",
    "build_othello_batch",
    "build_othello_vocab",
    "ensure_othello_datasets",
    "format_othello_eval_metrics",
    "legal_moves",
    "legal_prefix_length",
    "load_othello_dataset",
    "move_token",
    "othello_generation_metrics",
    "random_game_trace64",
    "required_block_size",
    "sample_othello_example",
]
