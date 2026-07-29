.PHONY: all format lint type test tests integration_tests serve_xpu serve_xpu_down e2e_test_xpu help extended_tests

# Default target executed when no arguments are given to make.
all: help

.EXPORT_ALL_VARIABLES:
UV_FROZEN = true

# Define a variable for the test file path.
TEST_FILE ?= tests/unit_tests/
PYTEST_EXTRA ?=
integration_test: TEST_FILE = tests/integration_tests/

# Test model configuration — all optional; override via env or command line.
# URL defaults point at the local `make serve` ports, so `make integration_test`
# works out of the box after `make serve`.
VLLM_TEST_MODEL ?= Qwen/Qwen2.5-0.5B-Instruct
VLLM_TEST_BASE_URL ?= http://localhost:8000/v1
VLLM_TEST_EMBED_MODEL ?= BAAI/bge-small-en-v1.5
VLLM_TEST_EMBED_BASE_URL ?= http://localhost:8001/v1


# unit tests are run with the --disable-socket flag to prevent network calls
test tests:
	uv run --group test pytest $(PYTEST_EXTRA) --disable-socket --allow-unix-socket $(TEST_FILE)

test_watch:
	uv run --group test ptw --snapshot-update --now . -- -vv $(TEST_FILE)


# Start both XPU vLLM servers (chat + embed) via Docker Compose, then wait until
# both /health endpoints respond. Model loading takes several minutes; this polls
# every 10s for up to 10 minutes before giving up.
serve_xpu:
	docker compose -f tests/integration_tests/docker-compose.xpu.yml up -d
	@echo "Waiting for chat server (port 8000)..."
	@for i in $$(seq 1 60); do \
		curl -sf --noproxy '*' http://localhost:8000/health > /dev/null 2>&1 && echo "  chat server ready" && break; \
		echo "  attempt $$i/60 — not ready yet, retrying in 10s..."; \
		sleep 10; \
		if [ $$i -eq 60 ]; then echo "ERROR: chat server did not become ready in time"; exit 1; fi; \
	done
	@echo "Waiting for embed server (port 8001)..."
	@for i in $$(seq 1 60); do \
		curl -sf --noproxy '*' http://localhost:8001/health > /dev/null 2>&1 && echo "  embed server ready" && break; \
		echo "  attempt $$i/60 — not ready yet, retrying in 10s..."; \
		sleep 10; \
		if [ $$i -eq 60 ]; then echo "ERROR: embed server did not become ready in time"; exit 1; fi; \
	done
	@echo "Both XPU servers ready."

# Stop and remove the XPU vLLM containers.
serve_xpu_down:
	docker compose -f tests/integration_tests/docker-compose.xpu.yml down

# Run the full XPU end-to-end cycle: start servers, run integration tests, tear down.
e2e_test_xpu:
	$(MAKE) serve_xpu && $(MAKE) integration_test; $(MAKE) serve_xpu_down

# integration tests are run without the --disable-socket flag to allow network calls
integration_test:
	uv run --group test --group test_integration pytest -v --tb=short -n auto --benchmark-disable $(TEST_FILE)

# CI integration tests - disabled until vLLM service is configured in CI

######################
# LINTING AND FORMATTING
######################

# Define a variable for Python and notebook files.
PYTHON_FILES=.
lint format: PYTHON_FILES=.
lint_diff format_diff: PYTHON_FILES=$(shell git diff --relative=libs/partners/vllm --name-only --diff-filter=d master | grep -E '\.py$$|\.ipynb$$')
lint_package: PYTHON_FILES=langchain_vllm
lint_tests: PYTHON_FILES=tests
UV_RUN_LINT = uv run --all-groups
lint_package lint_tests: UV_RUN_LINT = uv run --group lint

lint lint_diff lint_package lint_tests:
	./scripts/lint_imports.sh
	[ "$(PYTHON_FILES)" = "" ] || $(UV_RUN_LINT) ruff check $(PYTHON_FILES)
	[ "$(PYTHON_FILES)" = "" ] || $(UV_RUN_LINT) ruff format $(PYTHON_FILES) --diff
	$(MAKE) type

type:
	uv run --all-groups ty check langchain_vllm

format format_diff:
	[ "$(PYTHON_FILES)" = "" ] || $(UV_RUN_LINT) ruff format $(PYTHON_FILES)
	[ "$(PYTHON_FILES)" = "" ] || $(UV_RUN_LINT) ruff check --fix $(PYTHON_FILES)

check_imports: $(shell find langchain_vllm -name '*.py')
	$(UV_RUN_LINT) python ./scripts/check_imports.py $^

check_version:
	uv run python ./scripts/check_version.py

######################
# HELP
######################

help:
	@echo '----'
	@echo 'check_imports				- check imports'
	@echo 'check_version                - validate version consistency'
	@echo 'format                       - run code formatters'
	@echo 'lint                         - run linters'
	@echo 'type                         - run type checking'
	@echo 'test                         - run unit tests'
	@echo 'tests                        - run unit tests'
	@echo 'test TEST_FILE=<test_file>   - run all tests in file'
	@echo 'serve_xpu                    - start chat + embed vLLM servers (Intel XPU) via Docker Compose'
	@echo 'serve_xpu_down               - stop and remove the XPU vLLM containers'
	@echo 'e2e_test_xpu                 - serve_xpu + integration_test + serve_xpu_down in one command'
	@echo 'integration_test             - run integration tests (set VLLM_TEST_BASE_URL / VLLM_TEST_EMBED_BASE_URL)'
