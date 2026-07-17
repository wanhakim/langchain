# UV Unit Test Report

- **Date:** 2026-07-13
- **Repository:** LangChain monorepo
- **Environment model:** One isolated `uv`-managed `.venv` per package

## Executive Summary

All 21 package suites are software and mock based. No unit test was detected
that dispatches tensor or model computation to Intel XPU hardware. The test
script does not install the Intel XPU PyTorch wheel; this run therefore
validates isolated software environments, not XPU kernel execution.

| Metric | Count | Remarks |
|---|---:|---|
| Total unit tests (UT) | **5,989** | Normalized count across all 21 package suites |
| Pure software UT | **5,989** | No XPU-specific unit test was detected |
| Passing software UT | **5,805** | Passed in isolated per-package `uv` environments |
| Failed UT | **5** | Pytest assertion/snapshot failures |
| Blocked UT | **179** | 171 raw skipped/xfail outcomes plus 8 tests expanded from one module-level skip |
| XPU-runnable UT | **0** | No unit test dispatches computation to XPU |
| Passing XPU-runnable UT | **0** | The XPU-runnable denominator is zero |
| Passing rate over XPU-runnable UT | **n/a** | Undefined: `0 / 0` |
| Environment execution rate | **99.9%** | `5,805 / (5,805 + 5)`; blocked tests excluded |
| Pass rate including blocked UT | **96.9%** | `5,805 / 5,989` |

The useful signal for this run is the software environment execution rate. It
measures whether isolated package tests pass with their declared `uv` test
dependencies; it does not measure XPU execution.

## Environment and Workflow

| Item | Value |
|---|---|
| Package manager | `uv` 0.11.26 or repository-installed `uv` |
| Python environment | Separate `.venv` per package |
| Dependency setup | `uv sync --group test` |
| Test execution | `uv run --group test pytest` |
| XPU PyTorch installation | Not performed by the automated UV workflow |
| Unit-test scope | `tests/unit_tests/` only |
| XPU-specific unit tests detected | 0 |

Each package environment is independent. Removing `.venv` and running
`uv sync --group test` recreates the package environment without manually
creating or activating a virtual environment.

## Reproducible Commands

Remove each package `.venv`, runs `uv sync --group test`, executes
the package unit tests, and writes machine-readable and console artifacts.

The common per-package command is:

```bash
cd langchain/libs/xxx
rm -rf .venv
uv sync --group test

env -u LANGCHAIN_TRACING_V2 -u LANGCHAIN_API_KEY -u LANGSMITH_API_KEY \
    -u LANGSMITH_TRACING -u LANGCHAIN_PROJECT \
  uv run --group test pytest tests/unit_tests/ \
    -n 16 --dist worksteal \
    --benchmark-disable \
    --disable-socket --allow-unix-socket \
    --junitxml=/tmp/ut_results/xml/<package>.xml
```

The optional flags are used when the corresponding pytest plugins are
installed. If xdist or socket support is unavailable, rerun serially with the
supported subset of the command. Avoid `-n auto` on this host because it can
start too many workers.

## Per-Package Results

`Executed` means passed plus failed. `Blocked` includes the package's
normalized skipped/xfail outcomes. `XPU device` is zero for every package.

| Package | Total | Passed | Failed | Blocked, including skips/xfails | Executed | XPU device | Env passing rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| `langchain-core` | 2,149 | 2,138 | 0 | 11 | 2,138 | 0 | 100.0% |
| `langchain-text-splitters` | 150 | 140 | 0 | 10 | 140 | 0 | 100.0% |
| `langchain-tests` | 208 | 166 | 0 | 42 | 166 | 0 | 100.0% |
| `langchain-model-profiles` | 47 | 47 | 0 | 0 | 47 | 0 | 100.0% |
| `langchain` | 1,032 | 1,020 | 0 | 12 | 1,020 | 0 | 100.0% |
| `langchain-classic` | 745 | 652 | 0 | 93 | 652 | 0 | 100.0% |
| `langchain-anthropic` | 205 | 205 | 0 | 0 | 205 | 0 | 100.0% |
| `langchain-chroma` | 29 | 29 | 0 | 0 | 29 | 0 | 100.0% |
| `langchain-deepseek` | 30 | 29 | 0 | 1 | 29 | 0 | 100.0% |
| `langchain-exa` | 2 | 2 | 0 | 0 | 2 | 0 | 100.0% |
| `langchain-fireworks` | 129 | 129 | 0 | 0 | 129 | 0 | 100.0% |
| `langchain-groq` | 70 | 69 | 0 | 1 | 69 | 0 | 100.0% |
| `langchain-huggingface` | 40 | 39 | 0 | 1 | 39 | 0 | 100.0% |
| `langchain-mistralai` | 85 | 83 | 1 | 1 | 84 | 0 | 98.8% |
| `langchain-nomic` | 3 | 3 | 0 | 0 | 3 | 0 | 100.0% |
| `langchain-ollama` | 98 | 96 | 0 | 2 | 96 | 0 | 100.0% |
| `langchain-openai` | 523 | 515 | 4 | 4 | 519 | 0 | 99.2% |
| `langchain-openrouter` | 251 | 251 | 0 | 0 | 251 | 0 | 100.0% |
| `langchain-perplexity` | 153 | 152 | 0 | 1 | 152 | 0 | 100.0% |
| `langchain-qdrant` | 2 | 2 | 0 | 0 | 2 | 0 | 100.0% |
| `langchain-xai` | 38 | 38 | 0 | 0 | 38 | 0 | 100.0% |
| **Total** | **5,989** | **5,805** | **5** | **179** | **5,810** | **0** | **99.9%** |

## Failures and Blocked Outcomes

| Package or scope | Failed | Blocked | Details |
|---|---:|---:|---|
| `langchain-mistralai` | 1 | 1 | Failed `tests.unit_tests.test_embeddings.test_mistral_init`; one skipped/xfail outcome |
| `langchain-openai` | 4 | 4 | Failed token-ID, client utility, and two embedding tests; four skipped/xfail outcomes |
| Other packages | 0 | 174 | Skipped/xfail outcomes, including the normalized module-level skip expansion in `langchain-classic` |

The UV run did not reproduce the shared-conda OpenAI collection blocker. In
the isolated OpenAI environment, the response-stream file collected and the
package normalized to 523 tests.

## Comparison With Shared Conda Run

The two reports use different environment layouts and test outcomes, so their
pass/fail counts should not be treated as a controlled benchmark. The UV run
has two more normalized tests because of OpenAI collection behavior and
isolated dependency resolution.

| Metric | Per-package UV | Shared conda report |
|---|---:|---:|
| Total UT | 5,989 | 5,987 |
| Passed | 5,805 | 5,735 |
| Failed | 5 | 46 |
| Blocked, including skips | 179 | 206 |
| XPU-runnable | 0 | 0 |

The UV workflow does not install `torch==2.12.0+xpu` automatically. The
optional runbook check for the Hugging Face environment can install it after
`uv sync`, then verify `torch.xpu.is_available()` and the device count. This
does not change the conclusion that the unit tests themselves are software or
mock based.

## Why XPU-Runnable UT Is Zero

The per-package report found no unit test that assigns or uses XPU hardware.
The tests exercise software behavior, mocks, API request plumbing, and local
fixtures.

Actual accelerator execution belongs in integration or hardware-specific
tests, which are outside this report's `tests/unit_tests/` scope.

## Known Issues

1. Remove only the target package's `.venv` when a clean package environment is
   needed; `uv sync --group test` recreates it.
2. Avoid `-n auto` because this host reports many CPUs and can start too many
   xdist workers. Use a fixed count such as `-n 16` or run serially.
3. `langchain-core` can be run in its own `.venv`, which avoids unrelated
   Hugging Face, `transformers`, `sklearn`, and `pandas` dependencies from the
   shared conda environment.
4. Check for local test edits before trusting core results. The runbook calls
   out a local short-circuit of the `blockbuster` fixture in
   `libs/core/tests/unit_tests/conftest.py`.
5. The automated UV script does not install Intel XPU PyTorch. Install it only
   after package sync if an XPU-bearing package environment is required, then
   verify the installation before testing.

## Automated Notes

The full automated script that drives all 21 packages is `run_per_package_uv_unit_tests.sh`:

```bash
#!/usr/bin/env bash
# Clean all package .venv directories, sync each package's test group with uv,
# and run unit tests while writing XML/log/status artifacts.

set -u

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
OUT="${OUT:-/tmp/per_package_uv_ut_results}"
NPROC="${NPROC:-16}"
RUN_TIMEOUT="${RUN_TIMEOUT:-0}"

mkdir -p "$OUT/xml" "$OUT/console" "$OUT/sync_logs"
STATUS_FILE="$OUT/status.tsv"
PACKAGE_FILE="$OUT/packages.tsv"
: > "$STATUS_FILE"
: > "$PACKAGE_FILE"

PACKAGES=(
  "core:libs/core:langchain-core"
  "text-splitters:libs/text-splitters:langchain-text-splitters"
  "standard-tests:libs/standard-tests:langchain-tests"
  "model-profiles:libs/model-profiles:langchain-model-profiles"
  "langchain:libs/langchain_v1:langchain"
  "langchain-classic:libs/langchain:langchain-classic"
  "anthropic:libs/partners/anthropic:langchain-anthropic"
  "chroma:libs/partners/chroma:langchain-chroma"
  "deepseek:libs/partners/deepseek:langchain-deepseek"
  "exa:libs/partners/exa:langchain-exa"
  "fireworks:libs/partners/fireworks:langchain-fireworks"
  "groq:libs/partners/groq:langchain-groq"
  "huggingface:libs/partners/huggingface:langchain-huggingface"
  "mistralai:libs/partners/mistralai:langchain-mistralai"
  "nomic:libs/partners/nomic:langchain-nomic"
  "ollama:libs/partners/ollama:langchain-ollama"
  "openai:libs/partners/openai:langchain-openai"
  "openrouter:libs/partners/openrouter:langchain-openrouter"
  "perplexity:libs/partners/perplexity:langchain-perplexity"
  "qdrant:libs/partners/qdrant:langchain-qdrant"
  "xai:libs/partners/xai:langchain-xai"
)

if ! command -v uv >/dev/null 2>&1; then
  echo "ERROR: uv is not on PATH." >&2
  exit 127
fi

if [[ "$RUN_TIMEOUT" == "0" ]]; then
  TIMEOUT_PREFIX=()
else
  TIMEOUT_PREFIX=(timeout "${RUN_TIMEOUT}s")
fi

echo "Root: $ROOT"
echo "Output: $OUT"
echo "Workers: $NPROC"
echo

echo "== Removing package .venv directories =="
for entry in "${PACKAGES[@]}"; do
  IFS=: read -r label relpath package_name <<< "$entry"
  printf '%s\t%s\t%s\n' "$label" "$relpath" "$package_name" >> "$PACKAGE_FILE"
  dir="$ROOT/$relpath"
  if [[ -d "$dir/.venv" ]]; then
    echo "rm -rf $relpath/.venv"
    rm -rf "$dir/.venv"
  else
    echo "no .venv: $relpath"
  fi
done

echo
START_ALL=$SECONDS
for entry in "${PACKAGES[@]}"; do
  IFS=: read -r label relpath package_name <<< "$entry"
  dir="$ROOT/$relpath"
  sync_log="$OUT/sync_logs/${label}.log"
  console_log="$OUT/console/${label}.log"
  xml_file="$OUT/xml/${label}.xml"

  echo "=================================================="
  echo "== $package_name ($relpath) =="

  if [[ ! -d "$dir" ]]; then
    echo "MISSING package dir: $dir"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$label" "$relpath" "$package_name" "missing_dir" "127" "127" "0" "$xml_file" "$console_log" >> "$STATUS_FILE"
    continue
  fi

  if [[ ! -d "$dir/tests/unit_tests" ]]; then
    echo "SKIP no tests/unit_tests directory"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$label" "$relpath" "$package_name" "no_tests_dir" "0" "0" "0" "$xml_file" "$console_log" >> "$STATUS_FILE"
    continue
  fi

  cd "$dir"

  echo "uv sync --group test"
  sync_start=$SECONDS
  uv sync --group test > "$sync_log" 2>&1
  sync_rc=$?
  sync_dt=$((SECONDS - sync_start))
  if [[ "$sync_rc" -ne 0 ]]; then
    echo "SYNC FAILED rc=$sync_rc (${sync_dt}s); see $sync_log"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$label" "$relpath" "$package_name" "sync_failed" "$sync_rc" "127" "$sync_dt" "$xml_file" "$console_log" >> "$STATUS_FILE"
    continue
  fi
  echo "sync ok (${sync_dt}s)"

    PACKAGE_XDIST_ARGS=()
    if [[ "$NPROC" != "0" && "$NPROC" != "1" ]] && uv run --group test python -c "import xdist" >/dev/null 2>&1; then
        PACKAGE_XDIST_ARGS=(-n "$NPROC" --dist worksteal)
    fi

    PACKAGE_BENCHMARK_ARGS=()
    if uv run --group test python -c "import pytest_benchmark" >/dev/null 2>&1; then
        PACKAGE_BENCHMARK_ARGS=(--benchmark-disable)
    fi

    PACKAGE_SOCKET_ARGS=()
    if uv run --group test python -c "import pytest_socket" >/dev/null 2>&1; then
        PACKAGE_SOCKET_ARGS=(--disable-socket --allow-unix-socket)
    fi

    echo "pytest args: ${PACKAGE_XDIST_ARGS[*]:-(serial)} ${PACKAGE_BENCHMARK_ARGS[*]:-(no benchmark flag)} ${PACKAGE_SOCKET_ARGS[*]:-(no socket flag)}"

  echo "pytest tests/unit_tests/"
  test_start=$SECONDS
  env -u LANGCHAIN_TRACING_V2 -u LANGCHAIN_API_KEY -u LANGSMITH_API_KEY \
      -u LANGSMITH_TRACING -u LANGCHAIN_PROJECT \
    "${TIMEOUT_PREFIX[@]}" uv run --group test pytest tests/unit_tests/ \
            "${PACKAGE_XDIST_ARGS[@]}" \
      -p no:cacheprovider \
            "${PACKAGE_BENCHMARK_ARGS[@]}" \
            "${PACKAGE_SOCKET_ARGS[@]}" \
      --junitxml="$xml_file" \
      > "$console_log" 2>&1
  test_rc=$?
  test_dt=$((SECONDS - test_start))
  summary=$(grep -E "= .* (passed|failed|error|errors|skipped|xfailed|xpassed).* in " "$console_log" | tail -1 || true)
  echo "test rc=$test_rc (${test_dt}s) ${summary}"

  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$label" "$relpath" "$package_name" "done" "$sync_rc" "$test_rc" "$test_dt" "$xml_file" "$console_log" >> "$STATUS_FILE"
done

echo "=================================================="
echo "All done in $((SECONDS - START_ALL))s"
echo "Artifacts: $OUT"
echo "Status TSV: $STATUS_FILE"
```
