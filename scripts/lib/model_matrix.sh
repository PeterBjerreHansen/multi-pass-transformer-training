#!/usr/bin/env bash

# Keep launcher validation explicit and fail before starting any long run.
# The Python CLI remains the authoritative model factory; this list protects
# matrix launchers from malformed shell tokens such as "memory_state#".
MPT_ARCHITECTURES=(
  transformer
  memory_tape
  joint_memory_tape
  memory_concat
  memory_add
  memory_state
  memory_update
)

validate_architecture_matrix() {
  if [[ "$#" -eq 0 ]]; then
    echo "architecture matrix must contain at least one model" >&2
    return 2
  fi

  local architecture candidate found
  for architecture in "$@"; do
    found=0
    for candidate in "${MPT_ARCHITECTURES[@]}"; do
      if [[ "${architecture}" == "${candidate}" ]]; then
        found=1
        break
      fi
    done
    if [[ "${found}" -ne 1 ]]; then
      echo "invalid architecture in matrix: ${architecture}" >&2
      echo "valid architectures: ${MPT_ARCHITECTURES[*]}" >&2
      return 2
    fi
  done
}
