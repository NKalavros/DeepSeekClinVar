# DeepSeek ClinVar

Using a distilled DeepSeek, ollama and DSPy to create an automated task of structured knowledge extraction against existing knowledgebases.

Given a database of articles and relationships, can it answer questions about the relationship of two proteins/a phenotype and a variant, etc.

## Installation

In a conda environment with cuda and pytorch (using 12.8 here), and preferably mamba for speed.

```
mamba install -c nvidia -c conda-forge llama-cpp-python llama.cpp cmake cuda-toolkit=12.8*
```

### Model options


#### Download a local 1.58-bit 671B quantized DeepSeek (130 GB) - High VRAM

Based on unsloth.ai instructions.

```
echo '
import os
os.chdir("/gpfs/scratch/nk4167")
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
from huggingface_hub import snapshot_download
snapshot_download(
  repo_id = "unsloth/DeepSeek-R1-GGUF",
  local_dir = "DeepSeek-R1-GGUF",
  allow_patterns = ["*UD-IQ1_S*"],
)' > download_deepseek.py
python download_deepseek.py
```

#### Build llama-cpp

```

git clone https://github.com/ggerganov/llama.cpp
export CUDA_HOME=$CONDA_PREFIX
export CUDAToolkit_ROOT=$CONDA_PREFIX
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH$:CONDA_PREFIX/lib"
export CUDA_TOOLKIT_ROOT_DIR=$CONDA_PREFIX

# Add find_package(CUDAToolkit REQUIRED) and set CMP0146 policy to NEW
echo '
cmake_policy(SET CMP0146 NEW)
find_package(CUDAToolkit REQUIRED)
include_directories(${CUDAToolkit_INCLUDE_DIRS})
' >> llama.cpp/CMakeLists.txt

export CMAKE_PREFIX_PATH=$CONDA_PREFIX:$CMAKE_PREFIX_PATH
export CUDAToolkit_ROOT=$CONDA_PREFIX

cmake llama.cpp -B llama.cpp/build \
    -DBUILD_SHARED_LIBS=OFF -DGGML_CUDA=ON -DLLAMA_CURL=ON \
    -DCMAKE_CUDA_COMPILER=$CONDA_PREFIX/bin/nvcc \
    -DCUDAToolkit_ROOT=$CUDAToolkit_ROOT \
    -DCMAKE_PREFIX_PATH=$CUDAToolkit_ROOT
cmake --build llama.cpp/build --config Release -j --clean-first --target llama-quantize llama-cli llama-gguf-split
cp llama.cpp/build/bin/llama-* llama.cpp
```

#### Run the example

```
./llama.cpp/llama-cli \
    --model DeepSeek-R1-GGUF/DeepSeek-R1-UD-IQ1_S/DeepSeek-R1-UD-IQ1_S-00001-of-00003.gguf \
    --cache-type-k q4_0 \
    --threads $SLURM_CPUS_PER_TASK \
    --prio 2 \
    --temp 0.6 \
    --ctx-size 8192 \
    --seed 3407 \
    --n-gpu-layers 10 \
    -no-cnv \
    --prompt "<｜User｜>Create a Flappy Bird game in Python.<｜Assistant｜>"
```

This was too slow for me so I'm abandoning it.


### Pull a distilled Deepseek-R1

#### Download an ollama binary (so you don't need root)

```
cd /gpfs/scratch/nk4167
mkdir ollama
cd ollama
wget https://github.com/ollama/ollama/releases/download/v0.5.7/ollama-linux-amd64.tgz
tar -xvf ollama-linux-amd64.tgz
cd ..
```

#### Generate the server

```
ollama/bin/ollama serve &
```

#### Pull and serve a model (e.g. 32B - Needs 2 V100 GPUs)

```
ollama/bin/ollama pull deepseek-r1:32b
ollama/bin/ollama run deepseek-r1:32b
```

#### 70B - (Needs 4 V100 GPUs)

```
ollama/bin/ollama pull deepseek-r1:70b
ollama/bin/ollama run deepseek-r1:70b
```

### Setting up DSPy

```
pip install dspy
pip install pymupdf4llm
pip install huggingface_hub hf_transfer
```