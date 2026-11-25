# Validation

The benchmark validates itself on every run: each (size, block) result is compared with the CPU output and reported as PASS/FAIL. The process exits with code 2 if any configuration fails.

## Manual checks (run on a CUDA machine)

| # | Command | Expected |
| --- | --- | --- |
| 1 | `./build/cuda_vector_benchmark` | All rows `VALIDATION: PASS`, exit code 0, CSV populated |
| 2 | `./build/cuda_vector_benchmark --sizes 1,7,1000 --blocks 32,256` | PASS. Sizes not divisible by block size exercise the bounds check |
| 3 | `./build/cuda_vector_benchmark --blocks 2048` | "block size must be <= 1024", exit code 1 |
| 4 | `./build/cuda_vector_benchmark --sizes abc` | "invalid value for --sizes", exit code 1 |
| 5 | `./build/cuda_vector_benchmark --sizes 10000 --iters 3 --output /tmp/t.csv` | CSV written to `/tmp/t.csv` |
| 6 | Run on a machine with no GPU / `CUDA_VISIBLE_DEVICES=""` | Clean error message, exit code 1, no crash |
| 7 | `compute-sanitizer ./build/cuda_vector_benchmark --sizes 100000 --iters 2` | No memory errors, no leaks |
| 8 | `python visualization/plot_results.py` | Four PNGs written to `visualization/` |

Record the actual outcomes (GPU model, driver, CUDA version) in the README once you have run them.
