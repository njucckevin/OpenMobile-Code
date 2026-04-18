"""
process_task.py - synthesize GUI tasks with a strong model API.

For each screen, gather context with `get_context(...)` and then ask the model
to generate tasks.
"""

import os
import json
import base64
from io import BytesIO
import random
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from PIL import Image

# Environment setup
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Import process_context. The caller should initialize it with the full
# unique-screen knowledge base via `init_context(...)` before calling `get_context(...)`.
import process_context
from utils import get_client, get_qwen_stream_client

# ============ System prompt sections ============

SYSTEM_PROMPT_BASE = """
You are a GUI explorer. Your goal is to explore a GUI environment and synthesize high-quality, high-difficulty, executable, high-level, multi-step GUI tasks/instructions.

You have already completed the exploration work. You have collected many screenshots from the current GUI environment, the transitions between them, and various functionalities within the corresponding app.

Now, you need to fully associate and imagine based on the following three sources of information to generate long-range, high-level tasks/instructions that are possible within the current app:
(1) A recalled screenshot of a specific screen
(2) Several screenshots in short-term memory that have transition relationships with this screenshot (screens that can be reached from the current screen)
(3) Importantly, some functionalities retrieved from long-term memory that are associated with the current screen (semantically related functionalities from other screens in the same app)

Based on these three sources of information, you should fully associate, imagine, and generate long-range, high-level tasks/instructions that are possible within the current app.
"""

GUIDELINES = """
## Guidelines

1. The provided screenshots and functionalities are only a portion of your recalled memories serving as context. Your ONLY task is to synthesize clear multi-step GUI instructions. The instructions you synthesize do not need to have direct connections with the current screen or operations, but can be inferred from the context. However, to ensure the difficulty and complexity of generated tasks, you are encouraged to analyze, associate, and combine functionalities from your memories.

2. There are two types of tasks to generate:
   - **Action tasks**: Require performing a series of actions to accomplish a goal. For example: "Set an alarm for tomorrow at 8 AM that repeats every weekday."
   - **Question-answering tasks**: Require performing a series of actions and answering a question related to the environment's content. For example: "In my to-do list, how many tasks need to be completed this Wednesday? Answer the question with a single number." Or synthesize Verify-type tasks targeting the environment's state (such as checkboxes, system status). You should specify the answer format clearly after the question, e.g., "Answer the question with a single number."
   You should decide which type of task is appropriate to generate based on the context.

3. Synthesized tasks **must be clear and explicit. Generated tasks should be specific with sufficient details**, so that executors will not feel confused. For example, "Help me create a new event in the calendar" is too broad - the executor is uncertain about the specific event settings. It should be changed to a more specific instruction that includes concrete configurations, e.g., date, time, title, description, duration, location, etc. You are encouraged to generate tasks that are as specific as possible, containing as many details or steps as possible.

4. Synthesized tasks must be executable. **If you want to generate a task that involves operating on app data (for example, deleting an entry in the calendar), you MUST make sure the data you want to operate on is present in the given screenshots.** At the same time, **if the screenshots do contain such data, you are encouraged to generate such data-operation tasks**.

5. Generated tasks should be diverse. Do not only focus on the app's main functions. Try to cover all functionalities of the app as much as possible, for example, elements or functions in corners of screens, or functionalities you associate from memories.

6. Generated tasks should be long-range. Do not generate single-step tasks such as clicking a button. You should associate which single-step tasks can be combined with which functionalities to construct more difficult tasks. You are encouraged to generate tasks that require executors to reason, plan, and complete in multiple steps. **You can also consider combining different sub-functions or sub-tasks into a long-range task, but ensure reasonableness.**

7. Generated tasks should be high-level. **Do not generate step-by-step instructions and detailed actions.** Instead, integrate multi-step instructions into a high-level intent to increase task difficulty. Generated instructions should be concise. **They should be a single command that contains specific details, rather than step-by-step operations for completing a task (but the app name can be included).**

8. Generated tasks should start from the phone's home screen, not from the currently provided screen. Therefore, do not generate tasks that are bound to temporary states of the current interface (for example, a popup dialog that appears). You should assume that executors start from the phone's home screen and may not be able to reach the current temporary state.

9. The operating environment is a virtual device with no network connection. Therefore, do not generate tasks that require internet connection or login. However, you can freely use data that is already saved in the existing app.
"""

EXAMPLES = """
## Example Tasks

Here are examples showing bad tasks and their improved versions:

<Example_1>
- Bad Task: Access and manage the list of all saved Bluetooth devices.
- Reason: It does not clearly specify what "manage" means or what action should be taken.
- Good Task: View all existing Bluetooth devices, and if any exist, delete all of them.
</Example_1>

<Example_2>
- Bad Task: Add a new recipe to the list using the plus button on the main recipe screen.
- Reason: It does not specify the concrete content of the recipe. Tasks should be as specific as possible.
- Good Task: In the Broccoli app, add a new recipe for "Tomato and Egg Stir-fry", set the category to "Stir-fry", and fill in the description as "Mom's favorite dish".
</Example_2>

<Example_3>
- Bad Task: Check the battery usage statistics and enable Battery Saver mode if necessary.
- Reason: It does not specify what exactly to check. The phrase "if necessary" will confuse the executor.
- Good Task: Write the top three items from battery usage statistics into the Markor app for recording and save it as "battery_usage_statistics", and enable Battery Saver mode.
</Example_3>

<Example_4>
- Bad Task: Dismiss the voice search connection error by tapping the 'Keyboard' button, then manually type 'The Beatles' in the search bar to find their songs.
- Reason: This task includes a temporary state (the voice search connection error) and assumes we start from the search interface. However, we should assume tasks start from the phone's home screen.
- Good Task: In <app name>, how many songs are included for The Beatles and Taylor Swift respectively? Answer the question with numbers separated by a comma.
</Example_4>

<Example_5>
- Bad Task: In the Broccoli app, use the search function to find the recipe 'Salmon with Dill Sauce'. Open its details page and answer how many servings it yields and the total preparation time required.
- Reason: This task contains too many specific operations. Tasks should be more high-level rather than step-by-step instructions.
- Good Task: In the Broccoli app, how many servings does 'Salmon with Dill Sauce' provide, and what is the total preparation time required?
</Example_5>

<Example_6>
- Bad Task: In Simple Calendar Pro, navigate to the 'Customize colors' menu, attempt to change the App icon color, and dismiss the warning popup regarding launcher compatibility if it appears.
- Reason: This task contains unnecessary specific operations, such as navigating to the 'Customize colors' menu, and temporary states like warning popups.
- Good Task: Set the app color of Simple Calendar Pro to blue.
</Example_6>

<Example_7>
- Bad Task: In the Tasks app, what tasks do I have?
- Reason: This instruction is too vague; it should ask a more specific question.
- Good Task: In the Tasks app, which tasks due this week are not completed yet? Answer with titles only; if there are multiple, separate them with commas.
</Example_7>

<Example_8>
- Bad Task: In the Audio Recorder app, configure the settings for high-fidelity recording. After entering the app, navigate to the setup menu and change the recording format to Wav, set the sample rate to 48kHz, and ensure the channel count is set to Stereo. Finally, tap Apply to save the configuration.
- Reason: This task contains too many step-by-step operations. To increase difficulty, we should propose an imperative goal rather than listing each operation step by step.
- Good Task: Record an audio file in Wav format with 48kHz sample rate and Stereo channel using Audio Recorder, and save it as test_audio.
</Example_8>
"""

# Build the full system prompt.
SYSTEM_PROMPT = SYSTEM_PROMPT_BASE + GUIDELINES + EXAMPLES


def image_to_base64(img):
    """Convert PIL Image to base64 string"""
    buffer = BytesIO()
    img.save(buffer, format='PNG')
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


def build_user_prompt(context):
    """
    Build the user prompt from the context returned by `get_context(...)`.

    `context` contains:
        - current: current-screen information
        - predecessors: predecessor screen list
        - successors: transition target list
        - related_functionalities: related functionality list
    """
    parts = []

    # ==================== Part 1: current screen ====================
    current = context["current"]
    parts.append(f"## Current Screen")
    parts.append(f"**App**: {current['app']}")

    # Elements on the current screen (functionality + data)
    current_elements = [
        elem for elem in (current.get("elements") or [])
        if elem.get("type") in ["functionality", "data"]
    ]
    if current_elements:
        parts.append(f"\n**Elements on Current Screen** ({len(current_elements)} items):")
        for i, elem in enumerate(current_elements, 1):
            parts.append(f"  {i}. \"type\": {elem['type']}, \"description\": {elem['description']}")
    
    # ==================== Part 2: predecessor screens ====================
    predecessors = context.get("predecessors") or []
    if predecessors:
        parts.append("These are screens that can transition into the current screen:")
        for i, trans in enumerate(predecessors, 1):
            source = trans["source_screen"]
            parts.append(f"\n### Preceding Screen {i}")
            parts.append(f"**Screen**: {source['screen']}")

            source_elements = [
                elem for elem in (source.get("elements") or [])
                if elem.get("type") in ["functionality", "data"]
            ]
            if source_elements:
                parts.append(f"**Elements** ({len(source_elements)} items):")
                for j, elem in enumerate(source_elements, 1):
                    parts.append(f"  {j}. \"type\": {elem['type']}, \"description\": {elem['description']}")

    # ==================== Part 3: successor screens ====================
    successors = context.get("successors") or context.get("transitions") or []
    if successors:
        parts.append("These are screens that can be reached from the current screen:")
        for i, trans in enumerate(successors, 1):
            target = trans["target_screen"]
            parts.append(f"\n### Associated Screen {i}")
            parts.append(f"**Screen**: {target['screen']}")

            # Elements on successor screens (functionality + data)
            target_elements = [
                elem for elem in (target.get("elements") or [])
                if elem.get("type") in ["functionality", "data"]
            ]
            if target_elements:
                parts.append(f"**Elements** ({len(target_elements)} items):")
                for j, elem in enumerate(target_elements, 1):
                    parts.append(f"  {j}. \"type\": {elem['type']}, \"description\": {elem['description']}")

    # ==================== Part 4: related functionalities ====================
    related = context["related_functionalities"]
    if related:
        parts.append(f"\n## Related Functionalities from Other Screens ({len(related)} items)")
        parts.append("These are semantically related functionalities from other screens in the same app:")
        for i, rf in enumerate(related, 1):
            parts.append(f"  {i}. {rf['element']['description']}")

    # ==================== task-generation instruction ====================
    parts.append("\n## Your Task")
    parts.append("Based on the above context, carefully analyze and think, then generate 1-3 high-quality GUI tasks. Each task should be a concise but high-level instruction in English. If the current screen (the first screenshot) is not suitable for task generation, you may not generate any tasks.")
    parts.append("Each example should include an analysis of the task and the final task content.")
    parts.append("\nOutput format (JSON array):")
    parts.append("""[
    {"reasoning": "your analysis of why this task is good and how it relates to the context", "task": "task instruction 1"},
    {"reasoning": "your analysis of why this task is good and how it relates to the context", "task": "task instruction 2"},
    ...
]""")

    return "\n".join(parts)


def parse_result(result_str):
    """Parse model output into a list. Return None on failure."""
    try:
        # Find the first `[` and the last `]`.
        start = result_str.find('[')
        end = result_str.rfind(']')
        if start != -1 and end != -1 and start < end:
            result_str = result_str[start:end+1]
        return json.loads(result_str)
    except Exception as e:
        print(f"Parse error: {e}")
        return None


def process_screen(
    screen,
    screenshots_dir: str,
    num_predecessors: int = 1,
    n: int = 3,
    m: int = 30,
):
    """
    Process a single screen and return generated tasks.
    """
    try:
        # Get context.
        context = process_context.get_context(
            screen,
            num_predecessors=int(num_predecessors),
            n=int(n),
            m=int(m),
        )

        # Build the text prompt.
        text_prompt = build_user_prompt(context)

        # Build the multimodal image content.
        user_content = []

        # Current screen image.
        with Image.open(os.path.join(screenshots_dir, screen["screen"])) as current_img:
            current_img_b64 = image_to_base64(current_img)
        user_content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{current_img_b64}"}
        })

        # Predecessor screen images.
        for trans in context.get("predecessors") or []:
            with Image.open(os.path.join(screenshots_dir, trans["source_screen"]["screen"])) as source_img:
                source_img_b64 = image_to_base64(source_img)
            user_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{source_img_b64}"}
            })

        # Successor screen images.
        for trans in context.get("successors") or context.get("transitions") or []:
            with Image.open(os.path.join(screenshots_dir, trans["target_screen"]["screen"])) as target_img:
                target_img_b64 = image_to_base64(target_img)
            user_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{target_img_b64}"}
            })

        # Add the text prompt.
        user_content.append({
            "type": "text",
            "text": text_prompt
        })

        client = get_client()
        response = client.chat.completions.create(
            model="gemini-3.1-pro-preview",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content}
            ]
        )

        result = response.choices[0].message.content

        tasks = parse_result(result)

        print(result)

        if tasks is None:
            print(f"Failed to parse result for {screen['screen']}")
            print(f"Raw result: {result[:500]}...")

        return {
            "screen": screen["screen"],
            "app": screen["app"],
            "task": screen["task"],
            "tasks": tasks,
            "raw_response": result
        }

    except Exception as e:
        print(f"Error processing {screen['screen']}: {e}")
        return {
            "screen": screen["screen"],
            "app": screen["app"],
            "task": screen["task"],
            "tasks": None,
            "error": str(e)
        }


def synthesize_tasks(
    filtered_unique_screen_path: str,
    screenshots_dir: str,
    output_path: str,
    context_unique_screen_path: str,
    context_state_transfer_path: str,
    *,
    max_workers: int = 64,
    max_num: int = 10000,
    num_predecessors: int = 1,
    n: int = 3,
    m: int = 30,
    context_embedding_model: str = "all-MiniLM-L6-v2",
):
    """
    Generate tasks for the filtered screens while using the full
    `unique_screen_with_elements.json` knowledge base for `get_context(...)`.

    This is the functional wrapper for the original `__main__` workflow:
    - `filtered_unique_screen_path`: the `*_with_elements_filter.json` file from step 3
    - `context_unique_screen_path`: the full `*_with_elements.json` file from step 2
    - no periodic checkpoint saving; only the final `syn_tasks_eval` is written to `output_path`
    """
    # Initialize context with the full knowledge base.
    process_context.init_context(
        unique_screen_path=context_unique_screen_path,
        state_transfer_path=context_state_transfer_path,
        use_embedding_cache=False,
        embedding_model_name=context_embedding_model,
    )

    filtered_screens = json.load(open(filtered_unique_screen_path, "r"))

    random.shuffle(filtered_screens)
    filtered_screens = filtered_screens[: int(max_num)]
    print(f"Total filtered screens: {len(filtered_screens)}")

    valid_screens = [s for s in filtered_screens if s.get("elements")]
    print(f"Total valid screens: {len(valid_screens)} (from {filtered_unique_screen_path})")

    remaining = list(valid_screens)
    print(f"Remaining to process: {len(remaining)}")
    random.shuffle(remaining)

    processed_results = []

    with ThreadPoolExecutor(max_workers=int(max_workers)) as executor:
        future_to_screen = {
            executor.submit(
                process_screen,
                s,
                screenshots_dir,
                int(num_predecessors),
                int(n),
                int(m),
            ): s
            for s in remaining
        }

        for future in tqdm(as_completed(future_to_screen), total=len(future_to_screen)):
            screen = future_to_screen[future]
            try:
                result = future.result()
            except Exception as e:
                result = {
                    "screen": screen.get("screen"),
                    "app": screen.get("app"),
                    "task": None,
                    "tasks": None,
                    "error": f"Unhandled future exception: {e}",
                }

            processed_results.append(result)

    # ===== Build the flattened format used by later evaluation/filtering =====
    syn_tasks_eval = []
    for item in processed_results:
        if not isinstance(item, dict):
            continue
        if item.get("tasks") is None:
            continue
        for task in item.get("tasks") or []:
            if "task" not in task or "reasoning" not in task:
                print(f"task or reasoning not in: {task}")
                continue
            syn_tasks_eval.append(
                {
                    "screen": item.get("screen"),
                    "app": item.get("app"),
                    "task_aw": item.get("task"),
                    "task": task["task"],
                    "reason": task["reasoning"],
                }
            )

    random.shuffle(syn_tasks_eval)
    print(f"Number of synthesized tasks: {len(syn_tasks_eval)}")

    json.dump(syn_tasks_eval, open(output_path, "w"), indent=2, ensure_ascii=False)
    print(f"\nDone! Saved eval-format tasks to: {output_path}")

    return syn_tasks_eval


# ============ Main program ============
if __name__ == "__main__":
    # ============ Configuration: concurrency & saving ============
    MAX_WORKERS = 64
    SAVE_EVERY = 20  # Write once after every 20 completed samples.
    MAX_NUM = 1000

    # Parameter-free context initialization is no longer supported.
    # Call this through `pipeline.py`, or call `synthesize_tasks(...)` directly
    # and pass:
    #   - context_unique_screen_path (the full unique_screen_with_elements.json)
    #   - context_state_transfer_path
    raise ValueError(
        "Please call synthesize_tasks(...) with explicit context_unique_screen_path/context_state_transfer_path "
        "or run via pipeline.py"
    )
