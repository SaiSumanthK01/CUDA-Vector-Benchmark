#!/usr/bin/env bash
# One command: build -> benchmark -> charts -> report -> final zip.
set -e
cd "$(dirname "$0")"

ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '.')
[ -n "$ARCH" ] && ARCH_FLAG="-DCMAKE_CUDA_ARCHITECTURES=$ARCH" || ARCH_FLAG=""

cmake -S . -B build $ARCH_FLAG
cmake --build build -j
./build/cuda_vector_benchmark
pip install -q -r requirements.txt
python3 visualization/plot_results.py
python3 visualization/make_report.py

cd ..
rm -f cuda-vector-benchmark-final.zip
zip -qr cuda-vector-benchmark-final.zip "$(basename "$OLDPWD")" -x "*/build/*" "*/__pycache__/*"
echo "DONE. Download: $(pwd)/cuda-vector-benchmark-final.zip"
