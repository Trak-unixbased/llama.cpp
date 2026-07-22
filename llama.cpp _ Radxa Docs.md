---
    tags: []
    source: https://docs.radxa.com/en/orion/o6/app-development/artificial-intelligence/llama-cpp
    created: 2026-07-22T20:31:38.918Z
---

The core goal of llama.cpp is to enable high-performance large language model (LLM) inference across a wide range of hardware, from local devices to the cloud, with minimal configuration.

What is llama.cpp?

llama.cpp is a high-performance LLM inference framework implemented in pure C/C++. It avoids heavy external dependencies and supports efficient computation on both CPUs and GPUs. With its GGUF format and quantization techniques, models that were previously too large can run smoothly on consumer devices such as PCs, Macs, and even phones.

This document will help you get started with llama.cpp quickly and complete environment setup and model execution efficiently.

Device

```
git clone https://github.com/ggml-org/llama.cpp.git

```

Device

```
sudo  apt  install cmake gcc g++ libcurl4-openssl-dev

```

Device

```
cmake -B build
cmake --build build --config Release -j$(nproc)

```

Device

```
git checkout 4aced7a
cmake -B build -DGGML_NATIVE=OFF -DGGML_CPU_ARM_ARCH=armv9-a+i8mm+dotprod -DGGML_CPU_KLEIDIAI=ON
cmake --build build --config Release -j$(nproc)

```

Hardware optimization

llama.cpp integrates the Arm KleidiAI library, which provides highly optimized matrix-multiplication kernels for hardware features such as SME, I8MM, and dot-product acceleration. You can enable this feature with the build option `GGML_CPU_KLEIDIAI=ON`.

Device

```
cmake -B build -DGGML_CPU_KLEIDIAI=ON
cmake --build build --config Release -j$(nproc)

```

Device

```
git lfs install
git clone https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B

```

Device

```
cd llama.cpp
pip3 install  -r ./requirements.txt
python3 convert_hf_to_gguf.py DeepSeek-R1-Distill-Qwen-1.5B/

```

Device

```
cd build/bin
./llama-quantize DeepSeek-R1-Distill-Qwen-1.5B-F16.gguf DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf Q4_K_M

```

```
 2  or  Q4_0    :  4.34G, +0.4685 ppl @ Llama-3-8B 3  or  Q4_1    :  4.78G, +0.4511 ppl @ Llama-3-8B 8  or  Q5_0    :  5.21G, +0.1316 ppl @ Llama-3-8B 9  or  Q5_1    :  5.65G, +0.1062 ppl @ Llama-3-8B 19  or  IQ2_XXS :  2.06 bpw quantization 20  or  IQ2_XS  :  2.31 bpw quantization 28  or  IQ2_S   :  2.5  bpw quantization 29  or  IQ2_M   :  2.7  bpw quantization 24  or  IQ1_S   :  1.56 bpw quantization 31  or  IQ1_M   :  1.75 bpw quantization 36  or  TQ1_0   :  1.69 bpw ternarization 37  or  TQ2_0   :  2.06 bpw ternarization 10  or  Q2_K    :  2.96G, +3.5199 ppl @ Llama-3-8B 21  or  Q2_K_S  :  2.96G, +3.1836 ppl @ Llama-3-8B 23  or  IQ3_XXS :  3.06 bpw quantization 26  or  IQ3_S   :  3.44 bpw quantization 27  or  IQ3_M   :  3.66 bpw quantization mix 12  or  Q3_K    : alias for Q3_K_M 22  or  IQ3_XS  :  3.3 bpw quantization 11  or  Q3_K_S  :  3.41G, +1.6321 ppl @ Llama-3-8B 12  or  Q3_K_M  :  3.74G, +0.6569 ppl @ Llama-3-8B 13  or  Q3_K_L  :  4.03G, +0.5562 ppl @ Llama-3-8B 25  or  IQ4_NL  :  4.50 bpw non-linear quantization 30  or  IQ4_XS  :  4.25 bpw non-linear quantization 15  or  Q4_K    : alias for Q4_K_M 14  or  Q4_K_S  :  4.37G, +0.2689 ppl @ Llama-3-8B 15  or  Q4_K_M  :  4.58G, +0.1754 ppl @ Llama-3-8B 17  or  Q5_K    : alias for Q5_K_M 16  or  Q5_K_S  :  5.21G, +0.1049 ppl @ Llama-3-8B 17  or  Q5_K_M  :  5.33G, +0.0569 ppl @ Llama-3-8B 18  or  Q6_K    :  6.14G, +0.0217 ppl @ Llama-3-8B 7  or  Q8_0    :  7.96G, +0.0026 ppl @ Llama-3-8B 1  or  F16     : 14.00G, +0.0020 ppl @ Mistral-7B 32  or  BF16    : 14.00G, -0.0050 ppl @ Mistral-7B 0  or  F32     : 26.00G              @ 7B COPY    : only copy tensors, no quantizing
```

Device

```
cd build/bin
./llama-cli -m DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf

```

```
> hi, who are you
<think>
</think>
Hi! I'm DeepSeek-R1, an artificial intelligence assistant created by DeepSeek. I'm at your service and would be delighted to assist you with any inquiries or tasks you may have.

```

Device

```
./llama-bench -m DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf

```

```
radxa@orion-o6:~/llama.cpp/build/bin$ ./llama-bench -m ~/DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf -t 8
| model                          |       size |     params | backend    | threads |          test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | ------------: | -------------------: |
| qwen2 1.5B Q4_K - Medium       |   1.04 GiB |     1.78 B | CPU        |       8 |         pp512 |         64.60 ± 0.27 |
| qwen2 1.5B Q4_K - Medium       |   1.04 GiB |     1.78 B | CPU        |       8 |         tg128 |         36.29 ± 0.16 |

```