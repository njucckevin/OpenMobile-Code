# OpenMobile: Building Open Mobile Agents with Task and Trajectory Synthesis

<p align="center">
&nbsp&nbsp📑 <a href="https://arxiv.org/pdf/2604.15093">Paper</a>&nbsp&nbsp | &nbsp&nbsp🌐 <a href="https://njucckevin.github.io/openmobile/">Homepage</a>&nbsp&nbsp | &nbsp&nbsp🤗 <a href="https://huggingface.co/datasets/cckevinn/OpenMobile-Data">Dataset</a>&nbsp&nbsp | &nbsp&nbsp🤖 <a href="https://huggingface.co/cckevinn/OpenMobile-8B">Model</a>&nbsp&nbsp | &nbsp&nbsp🤗 <a href="https://alankamisslin-mobile-agent-trajectory-viewer.hf.space/">DataViewer</a>&nbsp&nbsp
</p>

Mobile agents powered by vision-language models have demonstrated impressive capabilities in automating mobile tasks, with recent leading models achieving a marked performance leap, e.g., nearly 70\% success on AndroidWorld. However, these systems keep their training data closed and remain opaque about their task and trajectory synthesis recipes. We present **OpenMobile**, an open-source framework that synthesizes high-quality task instructions and agent trajectories, with two key components: (1) The first is a scalable task synthesis pipeline that constructs a global environment memory from exploration, then leverages it to generate diverse and grounded instructions.and (2) a policy-switching strategy for trajectory rollout. By alternating between learner and expert models, it captures essential error-recovery data often missing in standard imitation learning. Agents trained on our data achieve competitive results across three dynamic mobile agent benchmarks: notably, our fine-tuned Qwen2.5-VL and Qwen3-VL reach 51.7\% and **64.7\% on AndroidWorld**, far surpassing existing open-data approaches. Furthermore, we conduct transparent analyses on the overlap between our synthetic instructions and benchmark test sets, and verify that performance gains stem from broad functionality coverage rather than benchmark overfitting.

![OpenMobile Logo](assets/openmobile.png)

Release Plans:

- [x] OpenMobile trajectoy data
- [x] Fine-tuned checkpoints based on OpenMobile data
- [x] AndroidWorld evaluation code
- [x] Task and trajectory synthesis code
- [ ] Other code and resources

## 📋 Table of Contents
- [Project Structure](#project-structure)
- [Environment Setup](#environment-setup)
- [Evaluation](#evaluation)
- [Trajectory Synthesis](#trajectory-synthesis)
- [Training](#training)
- [Acknowledgements](#acknowledgements)
- [License](#license)
- [Citation](#citation)

<a id="project-structure"></a>
## 📂 Project Structure

The repository is organized into two main components.
* `AndroidWorld/` contains the execution-side code, including environment exploration, trajectory rollout, trajectory post-processing, and model evaluation on AndroidWorld.
* `task_synthesis/` contains the task-synthesis pipeline: it takes processed exploration results, builds screen-level context and environment memory, synthesizes the final high-level instructions.

<a id="environment-setup"></a>
## ⚙️ Environment Setup
We recommend using a single conda environment for the full OpenMobile pipeline, including both `AndroidWorld/` and `task_synthesis/`. The detailed setup instructions are documented in [`AndroidWorld/environment.md`](AndroidWorld/environment.md).


<a id="evaluation"></a>
## 📊 Evaluation

Evaluation on AndroidWorld can be run with the following steps.

1. Deploy the target model with vLLM (for example, `OpenMobile-8B`) and obtain `model_base_url` and `model_name`.

2. Start the AndroidWorld emulator / ADB environment:

```bash
EMULATOR_NAME=AndroidWorldAvd
~/Library/Android/sdk/emulator/emulator -avd $EMULATOR_NAME -port 5554 -no-snapshot -grpc 8554
```

For more details about the AndroidWorld environment setup, please also refer to the [official AndroidWorld repository](https://github.com/google-research/android_world).

3. Launch evaluation:

```bash
cd AndroidWorld
python run.py \
  --agent_name qwen3vl \
  --console_port 5554 \
  --grpc_port 8554 \
  --perform_emulator_setup=true \
  --model_base_url your_vllm_url \
  --model_name OpenMobile-8B \
  --model_api_key EMPTY \
  --checkpoint_dir runs/openmobile_8b_seed30 \
  --task_random_seed 30
```

<a id="trajectory-synthesis"></a>
## 🎮 Trajectory Synthesis

OpenMobile synthesizes trajectories in three stages: environment exploration, task-instruction synthesis, and trajectory rollout/post-processing. The scripts below are organized so that the output of each stage is the input of the next stage.

Before running this pipeline, please finish the environment setup in [`AndroidWorld/environment.md`](AndroidWorld/environment.md), start the AndroidWorld emulator, and configure an OpenAI-compatible API endpoint for strong-model calls:

```bash
export OPENAI_BASE_URL=your_openai_compatible_base_url
export OPENAI_API_KEY=your_api_key_or_EMPTY
```

The examples below assume the repository root is `/path/to/OpenMobile`.

### 1. Explore AndroidWorld Screens

First, randomly explore AndroidWorld apps to collect screen transitions. This stage records screenshots, transition trajectories, and the task initialization parameters used later for rollout.

```bash
cd /path/to/OpenMobile/AndroidWorld

python random_walk_aw.py \
  --console_port 5554 \
  --grpc_port 8554 \
  --perform_emulator_setup=true
```

The exploration results are written to:

```text
AndroidWorld/explore_results/
  screenshots/
  trajectories/
  params/
```

Then convert the raw random-walk trajectories into the state-transfer format consumed by the task-synthesis pipeline:

```bash
python process_explore.py \
  --traj_dir explore_results/trajectories \
  --out explore_results/state_transfer_explore.json
```

The key output of this step is `AndroidWorld/explore_results/state_transfer_explore.json`.

### 2. Synthesize Task Instructions

Next, build global environment memory from the exploration results and synthesize high-level task instructions with a strong multimodal model.

```bash
cd /path/to/OpenMobile/task_synthesis

python pipeline.py \
  --dataset_id androidworld_explore \
  --state_transfer ../AndroidWorld/explore_results/state_transfer_explore.json \
  --screenshots_dir ../AndroidWorld/explore_results/screenshots \
  --max_num_syn_screen 1000 \
  --max_workers 64
```

This pipeline deduplicates screens, annotates UI elements, builds context from the explored transition graph, generates task instructions, judges their quality, and writes the final rollout-ready instruction file:

```text
task_synthesis/output/androidworld_explore/
  synthesized_tasks_androidworld_explore_final.json
```

Each final item contains the base AndroidWorld task name, synthesized instruction, sample id, and the exploration `task_id` used to recover the same initialization parameters during rollout.

### 3. Rollout Synthesized Tasks

Use the synthesized instructions to collect agent trajectories in AndroidWorld. The rollout script initializes the environment with corresponding base task, but uses the synthesized instruction for agent rollout, and saves per-step screenshots and metadata.

```bash
cd /path/to/OpenMobile/AndroidWorld

python run_diy.py \
  --input_json ../task_synthesis/output/androidworld_explore/synthesized_tasks_androidworld_explore_final.json \
  --output_dir runs/androidworld_explore_rollout \
  --agent_name qwen3vl \
  --console_port 5554 \
  --grpc_port 8554 \
  --perform_emulator_setup=false \
  --use_params_init True \
  --qwen3vl_model_base_url your_vllm_url \
  --qwen3vl_model_name your_strong_model_name \
  --qwen3vl_model_api_key EMPTY
```

Rollout trajectories are stored under `AndroidWorld/runs/androidworld_explore_rollout/`, with one subdirectory per synthesized task.

#### Policy-switching Rollout

In the paper, we use **Policy-switching Rollout** to collect trajectories with error-recovery signal. Instead of using a single policy throughout the episode, the rollout starts with a weaker actor model and lets a stronger model monitor the trajectory. When the weak actor appears to deviate from the task, the strong model intervenes and continues from the current screen.

To enable this mode, deploy both a strong OpenAI-compatible model endpoint and a weak OpenAI-compatible model endpoint, then use `--agent_name qwen3vl_switching`:

```bash
cd /path/to/OpenMobile/AndroidWorld

python run_diy.py \
  --input_json ../task_synthesis/output/androidworld_explore/synthesized_tasks_androidworld_explore_final.json \
  --output_dir runs/androidworld_explore_switching_rollout \
  --agent_name qwen3vl_switching \
  --console_port 5554 \
  --grpc_port 8554 \
  --perform_emulator_setup=false \
  --qwen3vl_model_base_url your_strong_model_url \
  --qwen3vl_model_name your_strong_model_name \
  --qwen3vl_model_api_key EMPTY \
  --qwen3vl_switching_weak_model_base_url your_weak_model_url \
  --qwen3vl_switching_weak_model_name your_weak_model_name \
  --qwen3vl_switching_weak_model_api_key EMPTY
```

The strong model is used for monitoring and intervention, while the weak model is used for the initial rollout policy. The saved trajectories include policy-switching metadata such as `policy_source`, `monitor_output`, and `intervention_triggered`, which are consumed by the post-processing and conversion scripts below.

### 4. Post-process and Convert Trajectories

Merge successful rollouts into a single trajectory file:

```bash
python process_trajs.py \
  --runs_dir runs/androidworld_explore_rollout \
  --output-name data_merge_success.json
```

Optionally refine step-level text fields such as conclusions and thinking traces with a strong multimodal model:

```bash
python process_refine.py \
  --input runs/androidworld_explore_rollout/data_merge_success.json \
  --output runs/androidworld_explore_rollout/data_merge_success_conclusion.json \
  --mode conclusion

python process_refine.py \
  --input runs/androidworld_explore_rollout/data_merge_success_conclusion.json \
  --output runs/androidworld_explore_rollout/data_merge_success_conclusion_thinking.json \
  --mode thinking \
```

Finally, convert the merged/refined trajectories into the ShareGPT-style multimodal format used by LLaMA-Factory:

```bash
python convert_traj.py \
  --input runs/androidworld_explore_rollout/data_merge_success_conclusion_thinking.json \
  --output runs/androidworld_explore_rollout/openmobile_train.json \
  --refine \
  --model-format qwen25vl
```

The `--model-format` option controls the target SFT format. We currently support `qwen25vl` and `qwen3vl`, corresponding to the Qwen2.5-VL-style and Qwen3-VL-style action formats. The resulting JSON file can be used as instruction-tuning data.

<a id="training"></a>
## 💻 Training

We fine-tune OpenMobile models with [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory). Instead of maintaining a full training tutorial in this repository, we provide two reference configuration files under [`LlamaFactory/`](LlamaFactory/):

```text
LlamaFactory/
  dataset_info.json
  qwen3vl_full_sft.yaml
```

The released SFT data can be downloaded from the [OpenMobile-Data dataset](https://huggingface.co/datasets/cckevinn/OpenMobile-Data). The public training splits are `split1.json`, `split2.json`, `split3.json`, and `split4.json`; `dataset_info.json` registers these four files as `openmobile_split1` through `openmobile_split4`.

To use the reference config, place the four JSON split files in the LLaMA-Factory data directory, copy or merge `dataset_info.json` into LLaMA-Factory's dataset registry, and launch SFT with the provided YAML config:

```bash
llamafactory-cli train LlamaFactory/qwen3vl_full_sft.yaml
```

The YAML file is intended as a reproducible starting point for Qwen3-VL full-parameter SFT. Please adjust `model_name_or_path`, `dataset_dir`, batch size, DeepSpeed config, and output paths according to your local LLaMA-Factory setup and hardware.

<a id="acknowledgements"></a>
## 💐 Acknowledgements
Thanks to the following open-sourced projects:

[AndroidWorld](https://github.com/google-research/android_world)&#8194; 
[AndroidLab](https://github.com/THUDM/Android-Lab)&#8194;
[MobileWorld](https://github.com/Tongyi-MAI/MobileWorld)&#8194;
[ScaleCUA](https://github.com/OpenGVLab/ScaleCUA)&#8194; 
[OS-Genesis](https://github.com/OS-Copilot/OS-Genesis) &#8194;
[Qwen-VL](https://github.com/QwenLM/Qwen3-VL) &#8194; 
[LlamaFactory](https://github.com/hiyouga/LlamaFactory) &#8194;

<a id="license"></a>
## ⚖️ License

This project is licensed under the [Apache 2.0 License](https://www.google.com/search?q=LICENSE). Other released artifacts, third-party models, datasets, and derived resources may be subject to their own respective licenses and usage terms.

<a id="citation"></a>
## 📜 Citation

If you find this project useful, please consider citing:

```bibtex
@article{cheng2026openmobile,
  title={OpenMobile: Building Open Mobile Agents with Task and Trajectory Synthesis},
  author={Cheng, Kanzhi and Li, Zehao and Ma, Zheng and Chen, Nuo and Cao, Jialin and Sun, Qiushi and Ding, Zichen and Xu, Fangzhi and Yan, Hang and Chen, Jiajun and others},
  journal={arXiv preprint arXiv:2604.15093},
  year={2026}
}
```


