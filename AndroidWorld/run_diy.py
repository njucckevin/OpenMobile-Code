# Copyright 2024 The android_world Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
DIY runner: generate trajectories for (instruction, base_task_name) pairs.

Input JSON format (a list):
[
  {"base_task_name": "...", "instruction": "...", "sample_id": "..."},
  ...
]

Behavior:
- Start env once, start agent once (same as run.py).
- For each sample: instantiate the base task, run base_task.initialize_task(env),
  then run episode with goal=instruction (NOT task.goal), save trajectories, then
  base_task.tear_down(env).

Note:
- We intentionally do NOT require labels; evaluation (is_successful) is skipped.
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
from collections.abc import Sequence
from typing import Any, Type

from absl import app
from absl import flags
from absl import logging

# ---- Reduce noisy gRPC C++/absl logs (must be set BEFORE importing grpc/android_world) ----
# These messages often look like:
#   I0000 ... fork_posix.cc:71] Other threads are currently calling into gRPC, skipping fork() handlers
# They are usually harmless, but very noisy.
os.environ.setdefault("GRPC_VERBOSITY", "ERROR")
os.environ.setdefault("GRPC_TRACE", "none")
# gRPC uses Abseil logging in C++ in many builds; this env can help reduce INFO logs.
os.environ.setdefault("ABSL_MIN_LOG_LEVEL", "2")  # 0=INFO,1=WARNING,2=ERROR,3=FATAL
# Some environments still honor glog.
os.environ.setdefault("GLOG_minloglevel", "2")

from android_world import constants
from android_world import registry
from android_world import suite_utils
from android_world.agents import base_agent
from android_world.agents import human_agent
from android_world.agents import infer
from android_world.agents import m3a
from android_world.agents import random_agent
from android_world.agents import seeact
from android_world.agents import t3a
from android_world.agents import seeact_v
from android_world.env import env_launcher
from android_world.env import interface
from android_world.episode_runner import run_episode
from android_world.task_evals import task_eval

logging.set_verbosity(logging.WARNING)


def _find_adb_directory() -> str:
    """Returns the directory where adb is located."""
    potential_paths = [
        os.path.expanduser("~/Library/Android/sdk/platform-tools/adb"),
        os.path.expanduser("~/Android/Sdk/platform-tools/adb"),
        os.path.expanduser("~/android-sdk/platform-tools/adb"),
    ]
    for path in potential_paths:
        if os.path.isfile(path):
            return path
    raise EnvironmentError(
        "adb not found in the common Android SDK paths. Please install Android"
        " SDK and ensure adb is in one of the expected directories. If it's"
        " already installed, point to the installed location."
    )


_ADB_PATH = flags.DEFINE_string(
    "adb_path",
    _find_adb_directory(),
    "Path to adb. Set if not installed through SDK.",
)
_EMULATOR_SETUP = flags.DEFINE_boolean(
    "perform_emulator_setup",
    False,
    "Whether to perform emulator setup. This must be done once and only once"
    " before running Android World. After an emulator is setup, this flag"
    " should always be False.",
)
_DEVICE_CONSOLE_PORT = flags.DEFINE_integer(
    "console_port",
    5554,
    "The console port of the running Android device. This can usually be"
    " retrieved by looking at the output of `adb devices`. In general, the"
    " first connected device is port 5554, the second is 5556, and"
    " so on.",
)

_DEVICE_GRPC_PORT = flags.DEFINE_integer(
    "grpc_port", 8554, "The gprc_port of android device."
)

_SUITE_FAMILY = flags.DEFINE_enum(
    "suite_family",
    registry.TaskRegistry.ANDROID_WORLD_FAMILY,
    [
        # Families from the paper.
        registry.TaskRegistry.ANDROID_WORLD_FAMILY,
        registry.TaskRegistry.MINIWOB_FAMILY_SUBSET,
        # Other families for more testing.
        registry.TaskRegistry.MINIWOB_FAMILY,
        registry.TaskRegistry.ANDROID_FAMILY,
        registry.TaskRegistry.INFORMATION_RETRIEVAL_FAMILY,
    ],
    "Suite family to run. See registry.py for more information.",
)
_TASK_RANDOM_SEED = flags.DEFINE_integer(
    "task_random_seed", 30, "Random seed for task randomness."
)



# Agent specific.
_AGENT_NAME = flags.DEFINE_string("agent_name", "seeact_v", help="Agent name.")

# Qwen3VL (OpenAI-compatible server) specific.
_QWEN3VL_MODEL_BASE_URL = flags.DEFINE_string(
    "qwen3vl_model_base_url",
    "http://127.0.0.1:8000/v1",
    "Qwen3VL OpenAI-compatible base_url, e.g. http://host:port/v1",
)
_QWEN3VL_MODEL_API_KEY = flags.DEFINE_string(
    "qwen3vl_model_api_key",
    "EMPTY",
    "Qwen3VL API key for OpenAI-compatible server (if needed).",
)
_QWEN3VL_MODEL_NAME = flags.DEFINE_string(
    "qwen3vl_model_name",
    "",
    "Model name passed to /v1/chat/completions (depends on your server).",
)
_QWEN3VL_SWITCHING_WEAK_MODEL_BASE_URL = flags.DEFINE_string(
    "qwen3vl_switching_weak_model_base_url",
    "http://127.0.0.1:32011/v1",
    "Weak model OpenAI-compatible base_url for qwen3vl_switching.",
)
_QWEN3VL_SWITCHING_WEAK_MODEL_API_KEY = flags.DEFINE_string(
    "qwen3vl_switching_weak_model_api_key",
    "EMPTY",
    "Weak model API key for qwen3vl_switching.",
)
_QWEN3VL_SWITCHING_WEAK_MODEL_NAME = flags.DEFINE_string(
    "qwen3vl_switching_weak_model_name",
    "Qwen2.5-VL-7B-Instruct-baseline",
    "Weak model name for qwen3vl_switching.",
)

_FIXED_TASK_SEED = flags.DEFINE_boolean(
    "fixed_task_seed",
    True,
    "Whether to use the same task seed when running multiple task combinations"
    " (n_task_combinations > 1).",
)

_INPUT_JSON = flags.DEFINE_string(
    "input_json",
    None,
    "Path to a JSON file containing a list of {base_task_name, instruction, sample_id}.",
)
_OUTPUT_DIR = flags.DEFINE_string(
    "output_dir",
    None,
    "Output directory for trajectories.",
)

_USE_PARAMS_INIT = flags.DEFINE_boolean(
    "use_params_init",
    True,
    "If True, load params from each sample's 'params_path' (pickle) instead of "
    "calling task_type.generate_random_params().",
)

# MiniWoB is very lightweight and new screens/View Hierarchy load quickly.
_MINIWOB_TRANSITION_PAUSE = 0.2

# Additional guidelines for the MiniWob tasks.
_MINIWOB_ADDITIONAL_GUIDELINES = [
    (
        "This task is running in a mock app, you must stay in this app and"
        " DO NOT use the `navigate_home` action."
    ),
]


def _get_agent(
    env: interface.AsyncEnv,
    family: str | None = None,
) -> base_agent.EnvironmentInteractingAgent:
    """Gets agent."""
    print("Initializing agent...")
    agent = None
    if _AGENT_NAME.value == "human_agent":
        agent = human_agent.HumanAgent(env)
    elif _AGENT_NAME.value == "random_agent":
        agent = random_agent.RandomAgent(env)
    # Gemini.
    elif _AGENT_NAME.value == "m3a_gemini_gcp":
        agent = m3a.M3A(env, infer.GeminiGcpWrapper(model_name="gemini-1.5-pro-latest"))
    elif _AGENT_NAME.value == "t3a_gemini_gcp":
        agent = t3a.T3A(env, infer.GeminiGcpWrapper(model_name="gemini-1.5-pro-latest"))
    # GPT.
    elif _AGENT_NAME.value == "t3a_gpt4":
        agent = t3a.T3A(env, infer.Gpt4Wrapper("gpt-4-turbo-2024-04-09"))
    elif _AGENT_NAME.value == "m3a_gpt4v":
        agent = m3a.M3A(env, infer.Gpt4Wrapper("gpt-4-turbo-2024-04-09"))
    elif _AGENT_NAME.value == "InternVL":
        agent = seeact_v.InternVL(
            env,
            infer.Gpt4Wrapper("gpt-4o"),
            model_name="",
            model_address="http://127.0.0.1:8000/",
        )
    elif _AGENT_NAME.value == "qwenvl":
        agent = seeact_v.QwenVL(
            env,
            infer.Gpt4Wrapper("gpt-4o"),
            model_name="",
            model_address="http://127.0.0.1:8000/",
            mode="Agent",
        )
    elif _AGENT_NAME.value == "qwen3vl":
        agent = seeact_v.Qwen3VL(
            env,
            infer.Gpt4Wrapper("gpt-4o"),
            model_base_url=_QWEN3VL_MODEL_BASE_URL.value,
            model_api_key=_QWEN3VL_MODEL_API_KEY.value,
            model_name=_QWEN3VL_MODEL_NAME.value,
        )
    elif _AGENT_NAME.value == "qwen3vl_switching":
        agent = seeact_v.Qwen3VL_Switching(
            env,
            infer.Gpt4Wrapper("gpt-4o"),
            model_base_url=_QWEN3VL_MODEL_BASE_URL.value,
            model_api_key=_QWEN3VL_MODEL_API_KEY.value,
            model_name=_QWEN3VL_MODEL_NAME.value,
            weak_model_base_url=_QWEN3VL_SWITCHING_WEAK_MODEL_BASE_URL.value,
            weak_model_api_key=_QWEN3VL_SWITCHING_WEAK_MODEL_API_KEY.value,
            weak_model_name=_QWEN3VL_SWITCHING_WEAK_MODEL_NAME.value,
        )
    elif _AGENT_NAME.value == "qwen25vl":
        agent = seeact_v.Qwen25VL(
            env,
            infer.Gpt4Wrapper("gpt-4o"),
            model_base_url=_QWEN3VL_MODEL_BASE_URL.value,
            model_api_key=_QWEN3VL_MODEL_API_KEY.value,
            model_name=_QWEN3VL_MODEL_NAME.value,
        )

    if not agent:
        raise ValueError(f"Unknown agent: {_AGENT_NAME.value}")

    if (
        agent.name in ["M3A", "T3A", "SeeAct"]
        and family
        and family.startswith("miniwob")
        and hasattr(agent, "set_task_guidelines")
    ):
        agent.set_task_guidelines(_MINIWOB_ADDITIONAL_GUIDELINES)
    agent.name = _AGENT_NAME.value

    return agent


def _read_samples(path: str) -> list[dict[str, str]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("input_json must be a JSON list.")
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"Item {i} must be an object.")
        # Normalize/validate required fields.
        # NOTE: Some datasets store `sample_id` as an int. We accept int/str and
        # normalize to str so downstream code can assume a stable type.
        for k in ("base_task_name", "instruction"):
            if k not in item or not isinstance(item[k], str) or not item[k].strip():
                raise ValueError(f"Item {i} missing/invalid field: {k}")

        if "sample_id" not in item or item["sample_id"] is None:
            raise ValueError(f"Item {i} missing/invalid field: sample_id")
        if isinstance(item["sample_id"], int):
            item["sample_id"] = str(item["sample_id"])
        if not isinstance(item["sample_id"], str) or not item["sample_id"].strip():
            raise ValueError(f"Item {i} missing/invalid field: sample_id")
    return data  # type: ignore[return-value]


def _derive_instance_seed(task_random_seed: int, task_name: str, instance_id: int) -> int:
    """Match suite_utils.create_suite() seed derivation exactly."""
    unique_seed_str = f"{task_random_seed}_{task_name}_{instance_id}"
    return int(hashlib.sha256(unique_seed_str.encode()).hexdigest(), 16) % (2**32)


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _main() -> None:
    if not _INPUT_JSON.value:
        raise ValueError("--input_json is required.")

    samples = _read_samples(_INPUT_JSON.value)
    params_dir = "./explore_results/params"

    env = env_launcher.load_and_setup_env(
        console_port=_DEVICE_CONSOLE_PORT.value,
        emulator_setup=_EMULATOR_SETUP.value,
        adb_path=_ADB_PATH.value,
        grpc_port=_DEVICE_GRPC_PORT.value,
    )
    agent = _get_agent(env, _SUITE_FAMILY.value)

    if _SUITE_FAMILY.value.startswith("miniwob"):
        # MiniWoB pages change quickly, don't need to wait for screen to stabilize.
        agent.transition_pause = _MINIWOB_TRANSITION_PAUSE
    else:
        agent.transition_pause = None

    task_registry = registry.TaskRegistry().get_registry(family=_SUITE_FAMILY.value)

    base_out = os.path.join(os.path.dirname(os.path.abspath(__file__)), _OUTPUT_DIR.value)
    _ensure_dir(base_out)

    instance_counters: dict[str, int] = {}

    for idx, item in enumerate(samples):
        base_task_name = item["base_task_name"]
        instruction = item["instruction"]
        sample_id = item["sample_id"]

        if base_task_name not in task_registry:
            raise ValueError(f"Unknown base_task_name: {base_task_name}")
        task_type: Type[task_eval.TaskEval] = task_registry[base_task_name]

        # Instantiate params similar to suite_utils._instantiate_task(), and match run.py seeds.
        if _FIXED_TASK_SEED.value:
            instance_id = 0
        else:
            instance_id = instance_counters.get(base_task_name, 0)
            instance_counters[base_task_name] = instance_id + 1

        # Simple resume behavior: if save_dir exists, skip (avoid re-running buggy/incomplete samples).
        save_dir = os.path.join(base_out, sample_id + "_" + base_task_name)
        if os.path.exists(save_dir):
            print(f"[{idx+1}/{len(samples)}] SKIP (dir exists): {save_dir}")
            continue

        seed = _derive_instance_seed(_TASK_RANDOM_SEED.value, base_task_name, instance_id)
        task_type.set_device_time(env)
        import random as _random  # local to keep file minimal

        if _USE_PARAMS_INIT.value:
            print("load params")
            task_id = item["task_id"]
            params_filename = task_id+"_params.pkl"
            params_path = os.path.join(params_dir, params_filename)
            params = pickle.load(open(params_path, "rb"))
        else:
            _random.seed(seed)
            params = task_type.generate_random_params()

        # Ensure params contains a seed value.
        if constants.EpisodeConstants.SEED not in params:
            raise ValueError(f"params does not have seed: {params}")
        
        task = task_type(params)
        _ensure_dir(save_dir)

        print(f"[{idx+1}/{len(samples)}] base_task={base_task_name} sample_id={sample_id}")
        try:
            task.initialize_task(env)
            print("init complete")
            # NOTE: use synthesized instruction (not task.goal), so we do not use the test instructions.
            max_n = 30
            run_episode(
                goal=instruction,
                agent=agent,
                max_n_steps=max_n,  # reuse
                start_on_home_screen=task.start_on_home_screen,
                termination_fn=None,  # keep simple; you can add MiniWoB termination if needed
                save_dir=save_dir,
            )
        except Exception as e:  # pylint: disable=broad-exception-caught
            # We still want to keep the loop going for trajectory generation.
            with open(os.path.join(save_dir, "error.txt"), "w", encoding="utf-8") as f:
                f.write(repr(e))
            print(f"  ERROR: {e!r}")
        finally:
            try:
                task.tear_down(env)
            except Exception as e:  # pylint: disable=broad-exception-caught
                print(f"  tear_down ERROR: {e!r}")

    env.close()


def main(argv: Sequence[str]) -> None:
    del argv
    _main()


if __name__ == "__main__":
    app.run(main)
