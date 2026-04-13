from dataclasses import dataclass
from string import ascii_lowercase
from typing import Dict, List, Sequence, Tuple

import random
import torch


PAD_TOKEN = "<pad>"
BOS_TOKEN = "<bos>"
SEP_TOKEN = "<sep>"
STATE_TOKEN = "<state>"
OUTPUT_TOKEN = "<output>"
EOS_TOKEN = "<eos>"
NEWLINE_TOKEN = "<nl>"
PROMPT_TOKEN = ">>>"
PRINT_TOKEN = "print"
EQUAL_TOKEN = "="
COMMA_TOKEN = ","
PLUS_TOKEN = "+"
MINUS_TOKEN = "-"
STAR_TOKEN = "*"

ADD_OP = "add"
SUB_OP = "sub"
MUL_OP = "mul"

SUPPORTED_OPERATIONS = (ADD_OP, SUB_OP, MUL_OP)
SUPPORTED_OPERAND_PATTERNS = ("num_num", "var_num", "num_var", "var_var")


@dataclass
class ReplBatch:
    idx: torch.Tensor
    targets: torch.Tensor
    metric_mask: torch.Tensor
    prompt_lengths: torch.Tensor
    output_lengths: torch.Tensor
    num_vars: torch.Tensor


@dataclass
class ReplVerificationResult:
    ok: bool
    python_source: str
    expected_suffix_lines: List[str]
    observed_suffix_lines: List[str]
    trace_lines: List[str]


def variable_names(num_vars: int) -> List[str]:
    if num_vars < 2:
        raise ValueError("num_vars must be at least 2")
    if num_vars > len(ascii_lowercase):
        raise ValueError(f"num_vars must be <= {len(ascii_lowercase)}")
    return list(ascii_lowercase[:num_vars])


def value_names(value_mod: int) -> List[str]:
    if value_mod < 2:
        raise ValueError("value_mod must be at least 2")
    return [str(i) for i in range(value_mod)]


def parse_operations(operations: Sequence[str] | str) -> List[str]:
    if isinstance(operations, str):
        items = [item.strip() for item in operations.split(",") if item.strip()]
    else:
        items = [item.strip() for item in operations if item.strip()]
    if not items:
        raise ValueError("operations must contain at least one operator")
    for item in items:
        if item not in SUPPORTED_OPERATIONS:
            raise ValueError(
                f"Unsupported operation '{item}'. Supported operations: {', '.join(SUPPORTED_OPERATIONS)}"
            )
    return items


def parse_operand_patterns(patterns: Sequence[str] | str | None) -> List[str]:
    if patterns is None:
        return list(SUPPORTED_OPERAND_PATTERNS)
    if isinstance(patterns, str):
        items = [item.strip() for item in patterns.split(",") if item.strip()]
    else:
        items = [item.strip() for item in patterns if item.strip()]
    if not items:
        raise ValueError("operand_patterns must contain at least one pattern")
    for item in items:
        if item not in SUPPORTED_OPERAND_PATTERNS:
            raise ValueError(
                f"Unsupported operand pattern '{item}'. Supported patterns: {', '.join(SUPPORTED_OPERAND_PATTERNS)}"
            )
    return items


def required_block_size(max_num_vars: int, program_length: int) -> int:
    # Worst-case full sequence:
    # BOS
    # max_num_vars init lines: >>> a = 0 <nl>                             -> 5 * max_num_vars
    # program_length assignment lines: >>> a = b + 1 <nl>                  -> 7 * program_length
    # print after every assignment: >>> print a <nl>                       -> 4 * program_length
    # SEP                                                                   -> 1
    # state line: <state> a = 0 , ... <nl>                                 -> 1 + 4*max_num_vars
    # one output line per print: <output> a 0 <nl>                         -> 4 * program_length
    # EOS                                                                   -> 1
    return 9 * max_num_vars + 15 * program_length + 4


def build_repl_vocab(value_mod: int, max_num_vars: int) -> Tuple[List[str], Dict[str, int], Dict[int, str]]:
    vars_ = variable_names(max_num_vars)
    values = value_names(value_mod)
    tokens = [
        PAD_TOKEN,
        BOS_TOKEN,
        SEP_TOKEN,
        STATE_TOKEN,
        OUTPUT_TOKEN,
        EOS_TOKEN,
        NEWLINE_TOKEN,
        PROMPT_TOKEN,
        PRINT_TOKEN,
        EQUAL_TOKEN,
        COMMA_TOKEN,
        PLUS_TOKEN,
        MINUS_TOKEN,
        STAR_TOKEN,
    ]
    tokens.extend(vars_)
    tokens.extend(values)
    stoi = {token: i for i, token in enumerate(tokens)}
    itos = {i: token for token, i in stoi.items()}
    return tokens, stoi, itos


def decode_ids(ids: Sequence[int], itos: Dict[int, str]) -> List[str]:
    return [itos[int(i)] for i in ids]


def token_ids_to_trace_lines(ids: Sequence[int], itos: Dict[int, str]) -> List[str]:
    tokens = decode_ids(ids, itos)
    lines: List[str] = []
    current: List[str] = []
    for token in tokens:
        if token in {PAD_TOKEN, BOS_TOKEN}:
            continue
        if token == EOS_TOKEN:
            if current:
                lines.append(_tokens_to_line(current))
            break
        if token == NEWLINE_TOKEN:
            lines.append(_tokens_to_line(current))
            current = []
            continue
        if token == SEP_TOKEN:
            if current:
                lines.append(_tokens_to_line(current))
                current = []
            lines.append(SEP_TOKEN)
            continue
        current.append(token)
    if current:
        lines.append(_tokens_to_line(current))
    return lines


def build_repl_batch(
    batch_size: int,
    max_num_vars: int,
    program_length: int,
    stoi: Dict[str, int],
    operations: Sequence[str] | str,
    operand_patterns: Sequence[str] | str | None = None,
    value_mod: int = 10,
    randomize_num_vars: bool = False,
    print_prob: float = 0.25,
    loss_on: str = "suffix",
    device=None,
    rng: random.Random | None = None,
) -> ReplBatch:
    if loss_on not in {"suffix", "full"}:
        raise ValueError("loss_on must be either 'suffix' or 'full'")
    if not 0.0 <= print_prob <= 1.0:
        raise ValueError("print_prob must be in [0, 1]")

    parsed_ops = parse_operations(operations)
    parsed_patterns = parse_operand_patterns(operand_patterns)
    rng = rng or random.Random()

    idx_rows: List[List[int]] = []
    target_rows: List[List[int]] = []
    metric_rows: List[List[bool]] = []
    prompt_lengths: List[int] = []
    output_lengths: List[int] = []
    num_vars_rows: List[int] = []

    for _ in range(batch_size):
        num_vars = rng.randint(2, max_num_vars) if randomize_num_vars else max_num_vars
        full, prompt_len, output_len = _sample_repl_example(
            num_vars=num_vars,
            program_length=program_length,
            stoi=stoi,
            operations=parsed_ops,
            operand_patterns=parsed_patterns,
            value_mod=value_mod,
            print_prob=print_prob,
            rng=rng,
        )
        idx_row = full[:-1]
        target_row = full[1:]
        metric_row = [False] * len(target_row)

        if loss_on == "suffix":
            for pos in range(prompt_len - 1):
                target_row[pos] = -1

        for pos in range(prompt_len - 1, prompt_len - 1 + output_len):
            metric_row[pos] = True

        idx_rows.append(idx_row)
        target_rows.append(target_row)
        metric_rows.append(metric_row)
        prompt_lengths.append(prompt_len)
        output_lengths.append(output_len)
        num_vars_rows.append(num_vars)

    pad_id = stoi[PAD_TOKEN]
    max_len = max(len(row) for row in idx_rows)
    padded_idx_rows = [row + [pad_id] * (max_len - len(row)) for row in idx_rows]
    padded_target_rows = [row + [-1] * (max_len - len(row)) for row in target_rows]
    padded_metric_rows = [row + [False] * (max_len - len(row)) for row in metric_rows]

    return ReplBatch(
        idx=torch.tensor(padded_idx_rows, dtype=torch.long, device=device),
        targets=torch.tensor(padded_target_rows, dtype=torch.long, device=device),
        metric_mask=torch.tensor(padded_metric_rows, dtype=torch.bool, device=device),
        prompt_lengths=torch.tensor(prompt_lengths, dtype=torch.long, device=device),
        output_lengths=torch.tensor(output_lengths, dtype=torch.long, device=device),
        num_vars=torch.tensor(num_vars_rows, dtype=torch.long, device=device),
    )


def sample_repl_trace_ids(
    max_num_vars: int,
    program_length: int,
    stoi: Dict[str, int],
    operations: Sequence[str] | str,
    operand_patterns: Sequence[str] | str | None = None,
    value_mod: int = 10,
    randomize_num_vars: bool = False,
    print_prob: float = 0.25,
    rng: random.Random | None = None,
) -> List[int]:
    rng = rng or random.Random()
    parsed_ops = parse_operations(operations)
    parsed_patterns = parse_operand_patterns(operand_patterns)
    num_vars = rng.randint(2, max_num_vars) if randomize_num_vars else max_num_vars
    full, _, _ = _sample_repl_example(
        num_vars=num_vars,
        program_length=program_length,
        stoi=stoi,
        operations=parsed_ops,
        operand_patterns=parsed_patterns,
        value_mod=value_mod,
        print_prob=print_prob,
        rng=rng,
    )
    return full


def verify_repl_trace_ids(ids: Sequence[int], itos: Dict[int, str], value_mod: int) -> ReplVerificationResult:
    trace_lines = token_ids_to_trace_lines(ids, itos)
    python_lines: List[str] = []
    expected_suffix_lines: List[str] = []
    seen_sep = False

    for line in trace_lines:
        if line == SEP_TOKEN:
            seen_sep = True
            continue
        if not seen_sep:
            python_lines.append(_prompt_line_to_python(line, value_mod))
        else:
            expected_suffix_lines.append(line)

    namespace: Dict[str, object] = {"__builtins__": {}}
    pending_outputs: List[str] = []

    def capture_print(*args):
        pending_outputs.append(" ".join(str(arg) for arg in args))

    namespace["print"] = capture_print
    for python_line in python_lines:
        exec(compile(python_line, "<repl-task>", "exec"), namespace, namespace)

    observed_suffix_lines: List[str] = []
    observed_suffix_lines.append(_serialize_state_line(namespace))
    for output in pending_outputs:
        observed_suffix_lines.append(f"{OUTPUT_TOKEN} {output}")

    return ReplVerificationResult(
        ok=expected_suffix_lines == observed_suffix_lines,
        python_source="\n".join(python_lines),
        expected_suffix_lines=expected_suffix_lines,
        observed_suffix_lines=observed_suffix_lines,
        trace_lines=trace_lines,
    )


def _sample_repl_example(
    num_vars: int,
    program_length: int,
    stoi: Dict[str, int],
    operations: Sequence[str],
    operand_patterns: Sequence[str],
    value_mod: int,
    print_prob: float,
    rng: random.Random,
) -> Tuple[List[int], int, int]:
    vars_ = variable_names(num_vars)
    state = {var_name: rng.randrange(value_mod) for var_name in vars_}
    full: List[int] = [stoi[BOS_TOKEN]]
    printed_outputs: List[Tuple[str, int]] = []

    for var_name in vars_:
        full.extend(_encode_init_line(var_name, state[var_name], stoi))

    for _ in range(program_length):
        full.extend(
            _sample_assignment_line(
                vars_=vars_,
                state=state,
                stoi=stoi,
                operations=operations,
                operand_patterns=operand_patterns,
                value_mod=value_mod,
                rng=rng,
            )
        )
        if rng.random() < print_prob:
            printed_var = rng.choice(vars_)
            printed_outputs.append((printed_var, state[printed_var]))
            full.extend(_encode_print_line(printed_var, stoi))

    if not printed_outputs:
        printed_var = rng.choice(vars_)
        printed_outputs.append((printed_var, state[printed_var]))
        full.extend(_encode_print_line(printed_var, stoi))

    prompt_len = len(full) + 1  # +1 for upcoming SEP token
    full.append(stoi[SEP_TOKEN])

    state_line = _encode_state_line(vars_, state, stoi)
    output_lines: List[int] = []
    for printed_var, printed_value in printed_outputs:
        output_lines.extend(_encode_output_line(printed_var, printed_value, stoi))
    full.extend(state_line)
    full.extend(output_lines)
    full.append(stoi[EOS_TOKEN])

    output_len = len(state_line) + len(output_lines) + 1  # include EOS target
    return full, prompt_len, output_len


def _sample_assignment_line(
    vars_: Sequence[str],
    state: Dict[str, int],
    stoi: Dict[str, int],
    operations: Sequence[str],
    operand_patterns: Sequence[str],
    value_mod: int,
    rng: random.Random,
) -> List[int]:
    target = rng.choice(vars_)
    op_name = rng.choice(list(operations))
    pattern = rng.choice(list(operand_patterns))

    operator_token = _op_to_token(op_name)
    lhs_value, lhs_token = _sample_operand(pattern.split("_")[0], vars_, state, value_mod, rng)
    rhs_value, rhs_token = _sample_operand(pattern.split("_")[1], vars_, state, value_mod, rng)

    if operator_token == PLUS_TOKEN:
        result = (lhs_value + rhs_value) % value_mod
    elif operator_token == MINUS_TOKEN:
        result = (lhs_value - rhs_value) % value_mod
    elif operator_token == STAR_TOKEN:
        result = (lhs_value * rhs_value) % value_mod
    else:
        raise ValueError(f"Unsupported operator token: {operator_token}")

    state[target] = result
    return [
        stoi[PROMPT_TOKEN],
        stoi[target],
        stoi[EQUAL_TOKEN],
        stoi[lhs_token],
        stoi[operator_token],
        stoi[rhs_token],
        stoi[NEWLINE_TOKEN],
    ]


def _sample_operand(
    kind: str,
    vars_: Sequence[str],
    state: Dict[str, int],
    value_mod: int,
    rng: random.Random,
) -> Tuple[int, str]:
    if kind == "num":
        value = rng.randrange(value_mod)
        return value, str(value)
    if kind == "var":
        var_name = rng.choice(vars_)
        return state[var_name], var_name
    raise ValueError(f"Unsupported operand kind: {kind}")


def _encode_init_line(var_name: str, value: int, stoi: Dict[str, int]) -> List[int]:
    return [
        stoi[PROMPT_TOKEN],
        stoi[var_name],
        stoi[EQUAL_TOKEN],
        stoi[str(value)],
        stoi[NEWLINE_TOKEN],
    ]


def _encode_print_line(var_name: str, stoi: Dict[str, int]) -> List[int]:
    return [
        stoi[PROMPT_TOKEN],
        stoi[PRINT_TOKEN],
        stoi[var_name],
        stoi[NEWLINE_TOKEN],
    ]


def _encode_state_line(vars_: Sequence[str], state: Dict[str, int], stoi: Dict[str, int]) -> List[int]:
    tokens: List[int] = [stoi[STATE_TOKEN]]
    for index, var_name in enumerate(vars_):
        if index > 0:
            tokens.append(stoi[COMMA_TOKEN])
        tokens.extend([stoi[var_name], stoi[str(state[var_name])]])
    tokens.append(stoi[NEWLINE_TOKEN])
    return tokens


def _encode_output_line(var_name: str, value: int, stoi: Dict[str, int]) -> List[int]:
    return [
        stoi[OUTPUT_TOKEN],
        stoi[var_name],
        stoi[str(value)],
        stoi[NEWLINE_TOKEN],
    ]


def _prompt_line_to_python(line: str, value_mod: int) -> str:
    if not line.startswith(PROMPT_TOKEN):
        raise ValueError(f"Expected prompt line, got: {line}")
    content = line[len(PROMPT_TOKEN) :].strip()
    if content.startswith(PRINT_TOKEN):
        _, var_name = content.split(maxsplit=1)
        return f'print("{var_name}", {var_name})'
    lhs, rhs = content.split(EQUAL_TOKEN, maxsplit=1)
    return f"{lhs.strip()} = ({rhs.strip()}) % {value_mod}"


def _serialize_state_line(namespace: Dict[str, object]) -> str:
    vars_ = sorted(key for key in namespace.keys() if len(key) == 1 and key.isalpha())
    assignments = [f"{var_name} {namespace[var_name]}" for var_name in vars_]
    return f"{STATE_TOKEN} " + " , ".join(assignments)


def _tokens_to_line(tokens: Sequence[str]) -> str:
    if not tokens:
        return ""
    head = tokens[0]
    if head == PROMPT_TOKEN:
        if tokens[1] == PRINT_TOKEN:
            return f"{PROMPT_TOKEN} {PRINT_TOKEN} {tokens[2]}"
        return f"{PROMPT_TOKEN} {tokens[1]} = {tokens[3]} {tokens[4]} {tokens[5]}" if len(tokens) == 6 else f"{PROMPT_TOKEN} {tokens[1]} = {tokens[3]}"
    if head == STATE_TOKEN:
        parts: List[str] = []
        index = 1
        while index < len(tokens):
            var_name = tokens[index]
            value = tokens[index + 1]
            parts.append(f"{var_name} {value}")
            index += 2
            if index < len(tokens) and tokens[index] == COMMA_TOKEN:
                index += 1
        return f"{STATE_TOKEN} " + " , ".join(parts)
    if head == OUTPUT_TOKEN:
        return f"{OUTPUT_TOKEN} {tokens[1]} {tokens[2]}"
    raise ValueError(f"Unsupported token line: {tokens}")


def _op_to_token(op_name: str) -> str:
    if op_name == ADD_OP:
        return PLUS_TOKEN
    if op_name == SUB_OP:
        return MINUS_TOKEN
    if op_name == MUL_OP:
        return STAR_TOKEN
    raise ValueError(f"Unsupported operation: {op_name}")
