#!/usr/bin/env bash

# Validate the complete matrix before starting a long run. Read the supported
# names from the Python model factory so shell and Python cannot drift apart.
read -r -a MPT_ARCHITECTURES <<< "$(
  python -c 'from model_factory import ARCHITECTURES; print(" ".join(ARCHITECTURES))'
)"

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
