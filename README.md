# NInfer-5070Ti

**Qwen3.8-27B (3-bit GSQ) on a single 16 GB RTX 5070 Ti (Blackwell, `sm_120a`): 100K context, vision and speculative decoding (MTP3 / DFlash2).**

This is a fork of [roofkid/ninfer-4080](https://github.com/roofkid/ninfer-4080), itself an `sm_89` port of NInfer-4090 derived from [Neroued/ninfer](https://github.com/Neroued/ninfer) (the RTX 5090 / `sm_120a` engine). The engine, the `gsq3` weights profile and the Q3 kernels are inherited unchanged. This fork makes the build and the runtime accept `sm_120a` and documents what was measured on a 5070 Ti. The original 4080 README lives in the upstream repository.

Everything below was measured on one machine. Read [Known limits](#known-limits) before relying on a number.

## What changed

| File | Change |
|---|---|
| `CMakeLists.txt` | `CMAKE_CUDA_ARCHITECTURES` may be `89` or `120a` |
| `src/targets/qwen3_6/impl/runtime/layouts_impl.h` | runtime gate accepts compute capability 8.9 or 12.0 |
| `src/ops/gdn_gating_proj/bf16/bf16_gdn_gating_proj_residency.h` | minimum SM count is 76 (`NINFER_SM86`, Ada) or 70 (RTX 5070 Ti) |
| `tests/CMakeLists.txt` | the e8 codec test follows the configured architecture |
| `scripts/build-5070ti.sh` | native build with a CUDA 13.x toolchain |
| `tools/bench/make_corpus_131072.py` | builds the 131,072-token benchmark corpus |

## Requirements

- RTX 5070 Ti 16 GB (`sm_120a`), driver with CUDA 13 support.
- **nvcc 13.x** (tested with 13.4.92), gcc-14 / g++-14, CMake 3.28+, Ninja.
- nvcc 12.8 does **not** work: with `120a` several kernels use static shared memory above 48 KiB (up to 99 KiB, which the code checks with a `static_assert`), and the 12.8 linker rejects it (`uses too much shared data (0x.... bytes, 0xc000 max)`). 13.4 links and runs.
- The Docker image and the `run-ninfer-4080*` launchers of the 4080 fork are `sm_89` only and do not run on this card. No Docker image is provided here.

## Build

    scripts/build-5070ti.sh /path/to/cuda-13.x

If your system CUDA runtime is older than 13, put the CUDA 13 libraries on `LD_LIBRARY_PATH` when running the binaries:

    export LD_LIBRARY_PATH=/path/to/cuda-13.x/targets/x86_64-linux/lib:$LD_LIBRARY_PATH

## Model

`roofkid/Qwen3.8-27B-GSQ3-NInfer`, file `qwen3_8_27b_gsq3.ninfer` (13.3 GB). Download with `scripts/download-qwen38-gsq3.sh` (the script verifies the SHA-256, `c6f27073393e5bcc629489420470d71f52a27553bfc5c360fef07a25b3b550d7`; it places the file under `scripts/models/`).

## Serve (100K context, vision, MTP3)

    ./build/apps/ninfer-serve scripts/models/qwen3_8_27b_gsq3.ninfer --host 127.0.0.1 --port 8080 \
      --max-context 102400 --kv-capacity 102400 --kv-dtype rk4v4-e8 --max-concurrency 1 \
      --max-pending-requests 16 --prefill-chunk 2688 --host-kv-mib 4096 \
      --spec mtp --draft-tokens 3 --lm-head-draft --vision --preserve-thinking

For DFlash2 replace the spec flags with `--spec dflash2 --draft-tokens 7 --lm-head-draft`. Useful extras: `--api-key KEY`, `--default-max-tokens N`, `--default-thinking-budget N`.

OpenAI-style `/v1/chat/completions` works, including one `tools` call (returned HTTP 200 with a `tool_calls` entry).

## Measured results

RTX 5070 Ti 16 GB, driver 580.178.04, nvcc 13.4.92, Ryzen 5 5600G, PCIe 3.0 x16, Ubuntu 26.04.

### Correctness

| Check | Result |
|---|---|
| `ctest` | 133/133 pass (8 real-model tests skipped) |
| Quick perplexity, 1M-token corpus, INT8 KV | 4.596691 (the original artifact record is 4.596095) |
| Retrieval at 98K + vision (`tools.smoke.serve_retrieval`, 3 independent runs) | all pass |

### Throughput against the published 4080 table

Same method as the table in the [4080 README](https://github.com/roofkid/ninfer-4080#readme): `ninfer_bench`, 131,072-token tiled corpus, `rk4v4-e8` KV, `--prefill-chunk 1024`, one warmup, one measured repetition. The 4080 column is the figure published by the original author, **not** measured on the same machine. Decode uses a tiled corpus, so acceptance (and decode) is higher than on real text.

| Depth | Prefill tok/s (4080 → 5070 Ti) | MTP3 decode tok/s (4080 → 5070 Ti) | DFlash2 K=7 decode tok/s (4080 → 5070 Ti) |
|---:|---:|---:|---:|
| 8K | 2,719.9 → 2,699.4 | 151.2 → 173.4 | 166.7 → 172.3 (47 % acceptance, outlier) |
| 32K | 2,424.9 → 2,409.0 | 141.7 → 168.7 | 262.3 → 300.4 |
| 98K | 1,895.1 → 1,902.4 (98,304 tokens) | 122.3 → 144.1 | 212.7 → 234.5 |

### Real prompts

| Test | Result |
|---|---|
| Decode at 98K, code request, 256 tokens, MTP3 | 111.9 tok/s, acceptance 73.1 % (identical in 3 runs) |
| Decode with an image prompt, MTP3 | 134.0 tok/s, acceptance 86.7 % |
| Prefill at 98K through the server | about 1,930 tok/s |
| Four short prompts (code and prose), 512 output tokens, MTP3 | 111.6 – 132.2 tok/s, acceptance 49 – 65 % |
| Same four prompts, DFlash2 K=7 | 112.1 – 169.9 tok/s, acceptance 25 – 46 % |

### Memory (engine log, vision enabled)

| Profile | Weights | Free after startup |
|---|---:|---:|
| 102,400 tokens, MTP3 | 11.2 GiB | 1.48 GiB |
| 102,400 tokens, DFlash2 | 12.0 GiB | 0.54 GiB |
| 118,784 tokens, MTP3 | 11.2 GiB | 1.20 GiB |
| 131,072 tokens, MTP3 | 11.2 GiB | 0.98 GiB |

The 118,784 and 131,072 profiles only started and listened; no generation was run with them.

Raw reports and logs are under [docs/results/rtx5070ti/](docs/results/rtx5070ti/).

## Known limits

- Only `sm_120a` was built and tested. The `sm_89` path was not rebuilt after the per-architecture SM floor change.
- The cooperative-launch CTAs-per-SM table was measured on `sm_89` and was not re-measured on `sm_120a`. No `cudaErrorCooperativeLaunchTooLarge` occurred in any test.
- Perplexity was measured with INT8 KV, like the original anchor. The shipped `rk4v4-e8` KV profile was not re-measured for perplexity on this card.
- Throughput numbers are one repetition each and are compared against the author's published 4080 figures, not a 4080 in the same machine.
- The model is 3-bit weights with a 4-bit KV cache; occasional small errors in generated text were observed in interactive use.
- DFlash2 with vision leaves 0.54 GiB free after startup and was not stress-tested; the author validated it with vision up to 65,536 tokens on the 4080.
- 8 real-model `ctest` cases are skipped. Kernels remain tuned for Ada; no re-tuning was done for Blackwell.
- Tool calling was checked with a single request.

## Reporting results

Measurements from other 16 GB Blackwell cards are welcome. Please include GPU, driver, nvcc version, the exact command, and the raw output.

## Credits

[Neroued/ninfer](https://github.com/Neroued/ninfer), [roofkid/ninfer-4080](https://github.com/roofkid/ninfer-4080), [sergiuszm/ninfer-4090](https://github.com/sergiuszm/ninfer-4090), [Don-Chad/ninfer-3090](https://github.com/Don-Chad/ninfer-3090), [UDPSendToFailed/ninfer-4090](https://github.com/UDPSendToFailed/ninfer-4090), [ISTA-DASLab/Qwen3.8-27B-3Bit-GSQ](https://huggingface.co/ISTA-DASLab/Qwen3.8-27B-3Bit-GSQ), [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B), [z-lab/Qwen3.8-27B-DFlash2](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2).

## License

See [LICENSE](LICENSE) (inherited from the upstream repositories).
