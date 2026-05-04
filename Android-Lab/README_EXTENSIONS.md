# Android-Lab Extensions

This document describes the extra features added in this fork of AndroidLab. 
## What Is Added

This version adds evaluation support for several vision-language GUI agents:

- **Qwen2.5-VL**, through `Qwen2d5VLAgent`
- **Qwen3-VL**, through `Qwen3VLAgent`
- **ScaleCUA**, through `ScaleCUAAgent`

It also extends the result generation pipeline so that **GPT-5** can be used as the judge model when producing evaluation summaries.

## Main Files

The new model integrations and evaluation entry points are implemented in:

- `agent/mllm/qwen2d5vl_model.py`
- `agent/mllm/qwen3vl_model.py`
- `agent/mllm/scalecua_model.py`
- `evaluation/evaluation.py`
- `evaluation/auto_test.py`
- `templates/android_screenshot_template.py`
- `generate_result.py`

Ready-to-use configuration files are provided under `configs/`:

- `configs/qwen2.5vl_config.yaml`
- `configs/qwen3vl_config.yaml`
- `configs/scalecua_config.yaml`

## Installation

Install the base AndroidLab dependencies following the original README:

```bash
cd /path/to/Android-Lab
conda create -n Android-Lab python=3.11
conda activate Android-Lab
pip install -r requirements.txt
```

Qwen2.5-VL uses Qwen's agent utilities. If the import is unavailable in your environment, install it separately:

```bash
pip install qwen-agent
```

The added agents are called through an OpenAI-compatible Chat Completions API. This makes them usable with local model servers such as vLLM or SGLang, as long as the server exposes a compatible `/v1` endpoint.

## Configuration

Edit the corresponding YAML file before running evaluation. For example:

```yaml
agent:
    name: Qwen3VLAgent
    args:
        api_key: "EMPTY"
        api_base: "http://127.0.0.1:8000/v1"
        model_name: "qwen3-vl-8b"
        max_new_tokens: 1024
```

Update `api_key`, `api_base`, and `model_name` according to your model service. The default configs use Docker-based evaluation:

```yaml
eval:
  docker: True
  docker_args:
    image_name: android_eval:latest
    port: 6060
```

You can switch to an AVD setup or adjust the Docker image and port by editing the same config file.

## Running Evaluation

Run the new agents with the standard AndroidLab evaluation command:

```bash
# Qwen2.5-VL
python eval.py -n qwen25vl_eval -c configs/qwen2.5vl_config.yaml

# Qwen3-VL
python eval.py -n qwen3vl_eval -c configs/qwen3vl_config.yaml

# ScaleCUA
python eval.py -n scalecua_eval -c configs/scalecua_config.yaml
```

To run only selected tasks:

```bash
python eval.py -n qwen3vl_debug -c configs/qwen3vl_config.yaml --task_id calendar_1 clock_1
```

To run one or more apps:

```bash
python eval.py -n scalecua_calendar -c configs/scalecua_config.yaml --app calendar
```

Parallel evaluation is supported through the original `-p/--parallel` option:

```bash
python eval.py -n qwen25vl_parallel -c configs/qwen2.5vl_config.yaml -p 3
```

Make sure the machine has enough memory and disk space before increasing parallelism, especially when using Docker.

## Generating Results

Evaluation logs are written to:

```text
./logs/evaluation/<run_name>/
```

This fork adds `gpt-5` as a supported judge model in `generate_result.py`:

```bash
export OPENAI_API_KEY="your-openai-api-key"

python generate_result.py \
  --input_folder ./logs/evaluation/ \
  --output_folder ./logs/evaluation/ \
  --output_excel ./logs/evaluation/result_gpt5.xlsx \
  --judge_model gpt-5
```

If you use a custom OpenAI-compatible endpoint for judging, pass the API settings explicitly:

```bash
python generate_result.py \
  --input_folder ./logs/evaluation/ \
  --output_folder ./logs/evaluation/ \
  --output_excel ./logs/evaluation/result_gpt5.xlsx \
  --judge_model gpt-5 \
  --api_key your-api-key \
  --api_base http://127.0.0.1:8000/v1
```

The script produces a summary Excel file and a detailed per-app result file.

## Configuration Reference

| Model | Agent class | AutoTest class | Config |
| --- | --- | --- | --- |
| Qwen2.5-VL | `Qwen2d5VLAgent` | `PixellevelQWEN2d5VLMobileTask_AutoTest` | `configs/qwen2.5vl_config.yaml` |
| Qwen3-VL | `Qwen3VLAgent` | `PixellevelQWEN3VLMobileTask_AutoTest` | `configs/qwen3vl_config.yaml` |
| ScaleCUA | `ScaleCUAAgent` | `PixellevelMobileScaleCUATask_AutoTest` | `configs/scalecua_config.yaml` |

## Quick Smoke Test

```bash
cd /path/to/Android-Lab

# Edit configs/qwen3vl_config.yaml first.
python eval.py -n smoke_qwen3vl -c configs/qwen3vl_config.yaml --task_id calendar_1

python generate_result.py \
  --input_folder ./logs/evaluation/ \
  --output_folder ./logs/evaluation/ \
  --output_excel ./logs/evaluation/smoke_qwen3vl_gpt5.xlsx \
  --judge_model gpt-5
```
