# NInfer RTX 5070 Ti (sm_120a)

Fork de [roofkid/ninfer-4080](https://github.com/roofkid/ninfer-4080) que compila y corre en RTX 5070 Ti 16 GB (Blackwell, `sm_120a`, 70 SMs) con Qwen3.8-27B GSQ3 y 100K tokens de KV cache.

## Cambios
- `CMakeLists.txt`: permite `CMAKE_CUDA_ARCHITECTURES` = `89` o `120a`.
- `layouts_impl.h`: el gate de runtime acepta compute capability 8.9 o 12.0.
- `bf16_gdn_gating_proj_residency.h`: mínimo de SMs 76 (Ada, `NINFER_SM86`) o 70 (5070 Ti).
- `tests/CMakeLists.txt`: el test e8 usa la arquitectura configurada.
- `scripts/build-5070ti.sh`: build con CUDA 13.x.

## Build
Requiere nvcc 13.x, gcc-14/g++-14 y Ninja.

    scripts/build-5070ti.sh /ruta/a/cuda-13.x

## Modelo
`roofkid/Qwen3.8-27B-GSQ3-NInfer`, `qwen3_8_27b_gsq3.ninfer` (13.3 GB). Descarga con `scripts/download-qwen38-gsq3.sh` (verifica SHA-256).

## Servidor (100K de KV)

    ./build/apps/ninfer-serve scripts/models/qwen3_8_27b_gsq3.ninfer --host 127.0.0.1 --port 8080 \
      --max-context 102400 --kv-capacity 102400 --kv-dtype rk4v4-e8 --max-concurrency 1 \
      --max-pending-requests 16 --prefill-chunk 2688 --host-kv-mib 4096 \
      --spec mtp --draft-tokens 3 --lm-head-draft --vision --preserve-thinking

## Resultados medidos
RTX 5070 Ti, driver 580.178.04, nvcc 13.4, Ryzen 5 5600G, PCIe 3.0 x16.

| Prueba | Resultado |
|---|---|
| ctest | 133/133 pasan |
| Retrieval 98K + visión (3 corridas) | todas pasan |
| Decode a 98K, MTP3, 256 tok | 111.9 tok/s, aceptación 73.1% |
| Decode con visión, 16 tok | 134.0 tok/s, aceptación 86.7% |
| Prefill a 98K | ~1.95k tok/s |

Notas: requests cortos (12-256 tokens). Perplexity aún sin medir. Rendimiento sin re-afinar para Blackwell.
