# Intel XPU Enablement in LangChain

## Objective

Enable Intel XPU execution in LangChain

---

## Test Coverage Review (21 Packages)

| Result | Count |
| --- | --- |
| Passing | 5805 |
| Failing | 5 |
| Skipped / Blocked | 179 |

Failing and blocked tests are pre-existing framework issues unrelated to XPU enablement. Outside this workstream.

---

## LangChain Package Execution Model

| Package Category | Examples | How model runs |
| --- | --- | --- |
| Remote API | OpenAI, Anthropic, Groq | HTTP client → cloud server |
| Local server | Ollama, TGI | HTTP client → local daemon |
| **In-process (torch)** | **`langchain-huggingface`** | **Loads model directly via `torch`** |

`langchain-huggingface` is where LangChain's own code makes the device decision — where XPU support can be enabled.

---

## XPU Support Enhancement — Device Validation in `HuggingFacePipeline`

`HuggingFacePipeline` validates the requested device index against the CUDA device count — which is always 0 on an XPU machine, preventing XPU devices from being recognized even when XPU hardware is present.

**Before** — CUDA only. On an XPU machine, device count is 0, so any device request is rejected:

```
device_count = get CUDA device count        # returns 0 on XPU machine
if requested device >= device_count:
    raise ValueError                        # not XPU-aware
```

**After** — Falls back to XPU when no CUDA devices are present:

```
device_count = get CUDA device count
if device_count == 0 and XPU is available:
    device_count = get XPU device count     # now reflects actual hardware
if requested device >= device_count:
    raise ValueError                        # passes on XPU machine
```

The fallback approach was chosen to ensure full backward compatibility with existing CUDA deployments.

---

## Delivery Status

| Deliverable | Status |
| --- | --- |
| Implementation | Ready |
| Unit tests | Ready |
| XPU integration test | Ready |
| GitHub Issue | Created — pending maintainer discussion |
| Pull request | Ready to open once Issue is agreed |

---

## Next Step

Obtain maintainer agreement on the GitHub Issue → open PR for review.
