import json
from tqdm import tqdm
from collections import defaultdict
from screen_transition_index import create_index
import random


# ==================== Step 1: state_transfer -> unique_screen ====================
def build_unique_screens(
    data_path: str,
    screenshots_dir: str,
    cache_path: str,
    similarity_threshold: float = 0.95,
    output_path: str | None = None,
):
    """
    Cluster screen_before/screen_after images by pHash similarity and build
    a deduplicated unique screen list.
    """
    # Build the transition index (loads data and hash cache automatically).
    index = create_index(
        data_path=data_path,
        screenshots_dir=screenshots_dir,
        cache_path=cache_path,
        similarity_threshold=similarity_threshold,
    )

    # Build helper indices.
    image_to_app = {}
    screen_before_index = defaultdict(list)
    screen_after_index = defaultdict(list)

    for idx, item in enumerate(index.data):
        screen_before_index[item["screen_before"]].append(idx)
        screen_after_index[item["screen_after"]].append(idx)
        image_to_app[item["screen_before"]] = item["app"]
        image_to_app[item["screen_after"]] = item["app"]

    # Find unique screens.
    print("Finding unique screens...")
    unique_screens = []  # [(representative_image, [all_similar_images])]

    for img_name in tqdm(index.image_hashes.keys()):
        img_hash = index.image_hashes[img_name]

        found = False
        for rep_img, similar_list in unique_screens:
            rep_hash = index.image_hashes[rep_img]
            if index.compute_hash_distance(img_hash, rep_hash) >= similarity_threshold:
                similar_list.append(img_name)
                found = True
                break

        if not found:
            unique_screens.append((img_name, [img_name]))

    print(f"Unique screens: {len(unique_screens)}")

    # Build the output records.
    print("Generating output...")
    output = []

    for rep_img, similar_images in tqdm(unique_screens):
        as_before_ids = []
        as_after_ids = []
        for img in similar_images:
            as_before_ids.extend(screen_before_index.get(img, []))
            as_after_ids.extend(screen_after_index.get(img, []))

        as_before_ids = list(set(as_before_ids))
        as_after_ids = list(set(as_after_ids))

        output.append(
            {
                "screen": rep_img,
                "app": image_to_app.get(rep_img, "unknown"),
                "as_screen_before": as_before_ids,
                "as_screen_after": as_after_ids,
                "similar_images": similar_images,
            }
        )

    if output_path is not None:
        json.dump(output, open(output_path, "w"), indent=2, ensure_ascii=False)
        print(f"Saved to {output_path}")
        print(f"Total unique screens: {len(output)}")

    return output


# ==================== Step 2: unique_screen -> unique_screen_with_elements ====================
def annotate_unique_screens_with_elements(
    unique_screen_path: str,
    output_path: str,
    state_transfer_path: str,
    screenshots_dir: str,
    cache_path: str,
    similarity_threshold: float = 0.95,
    model: str = "gemini-3.1-pro-preview",
    num_threads: int = 64,
    save_every: int = 20,
):
    """
    Generate element annotations for each screen in unique_screen.json using a
    multimodal model.

    Behavior intentionally follows the original script:
    - If output_path already exists, use it as a resume file; otherwise load
      unique_screen_path.
    - Only process samples whose elements field is None.
    - Keep parse_result / process_screen tolerance unchanged.
    """
    client = get_client()

    # Used to recover screen_before / action_type / bbox from as_screen_after ids.
    index = create_index(
        data_path=state_transfer_path,
        screenshots_dir=screenshots_dir,
        cache_path=cache_path,
        similarity_threshold=similarity_threshold,
    )

    if os.path.exists(output_path):
        unique_screens = json.load(open(output_path, "r"))
        print(f"Loaded resume file: {output_path}")
    else:
        unique_screens = json.load(open(unique_screen_path, "r"))
        print(f"Loaded fallback file: {unique_screen_path}")

    print(f"Number of unique screens: {len(unique_screens)}")

    def _process_one(screen_idx, screen):
        try:
            screen_after_name = screen["screen"]
            app_name = screen["app"]

            img_after = Image.open(os.path.join(screenshots_dir, screen_after_name))
            img_after_b64 = image_to_base64(img_after)

            if screen["as_screen_after"]:
                screen_as_after_item = index.data[screen["as_screen_after"][0]]
                screen_before_name = screen_as_after_item["screen_before"]
                action_type = screen_as_after_item["action_type"]
                print(
                    f"screen before: {screen_before_name}, screen after: {screen_after_name}, action type: {action_type}"
                )

                img_before = Image.open(os.path.join(screenshots_dir, screen_before_name))
                img_before_with_action = draw_action_on_image(img_before, screen_as_after_item)
                img_before_b64 = image_to_base64(img_before_with_action)

                user_content = [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_before_b64}"}},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_after_b64}"}},
                    {
                        "type": "text",
                        "text": f"App: {app_name}\nAction: {action_type}\n\nThe first image is Screen Before (with action area marked in red). The second image is Screen After. Please analyze the elements on the second image.",
                    },
                ]
            else:
                user_content = [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_after_b64}"}},
                    {
                        "type": "text",
                        "text": f"App: {app_name}\n\nNote: There is no preceding screen context available. Only one screenshot is provided. Please analyze the elements on this image.",
                    },
                ]

            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ],
            )

            result = response.choices[0].message.content
            elements = parse_result(result)
            if elements is None:
                print("Elements is None")
                print(response)
            return screen_idx, elements
        except Exception as e:
            print(f"Error processing {screen['screen']}: {e}")
            return screen_idx, None

    # Only process samples whose elements field is None.
    to_process = [(i, s) for i, s in enumerate(unique_screens) if s.get("elements") is None]
    print(f"To process: {len(to_process)} / {len(unique_screens)}")

    completed = 0
    with ThreadPoolExecutor(max_workers=int(num_threads)) as executor:
        futures = {executor.submit(_process_one, i, s): i for i, s in to_process}

        for future in tqdm(as_completed(futures), total=len(to_process)):
            idx, elements = future.result()
            unique_screens[idx]["elements"] = elements

            completed += 1
            if save_every > 0 and completed % int(save_every) == 0:
                json.dump(unique_screens, open(output_path, "w"), indent=2, ensure_ascii=False)
                print(f"Progress saved: {completed}/{len(to_process)}")

    json.dump(unique_screens, open(output_path, "w"), indent=2, ensure_ascii=False)
    print(f"Done! Saved to {output_path}")
    return unique_screens


def filter_unique_screens_with_elements_by_task(
    unique_screen_with_elements_path: str,
    state_transfer_path: str,
    output_path: str,
    min_as_screen_before: int = 2,
):
    """
    Generate the filtered unique_screen_with_elements file:
    - assign task to each screen by directly looking up screen filenames in
      state_transfer
    - keep only as_screen_before / as_screen_after ids with the same task
    - then filter by len(as_screen_before) >= min_as_screen_before

    This intentionally keeps the original strict behavior: no extra tolerance
    is added here.
    """
    data = json.load(open(unique_screen_with_elements_path, "r"))
    print(f"Total samples: {len(data)}")

    transfer_data = json.load(open(state_transfer_path, "r"))
    id_to_task = [item.get("task") for item in transfer_data]

    image_to_task = {}
    for item in transfer_data:
        t = item.get("task")
        if t is None:
            continue
        sb = item.get("screen_before")
        sa = item.get("screen_after")
        if sb and sb not in image_to_task:
            image_to_task[sb] = t
        if sa and sa not in image_to_task:
            image_to_task[sa] = t

    filtered_before_total = 0
    filtered_after_total = 0
    before_total = 0
    after_total = 0

    for s in data:
        as_before = s.get("as_screen_before") or []
        as_after = s.get("as_screen_after") or []

        screen_name = s.get("screen")
        screen_task = image_to_task.get(screen_name)
        s["task"] = screen_task

        before_total += len(as_before)
        after_total += len(as_after)

        if screen_task is not None:
            new_as_before = [
                i for i in as_before if 0 <= i < len(id_to_task) and id_to_task[i] == screen_task
            ]
            new_as_after = [
                i for i in as_after if 0 <= i < len(id_to_task) and id_to_task[i] == screen_task
            ]
        else:
            new_as_before = []
            new_as_after = []

        filtered_before_total += (len(as_before) - len(new_as_before))
        filtered_after_total += (len(as_after) - len(new_as_after))
        s["as_screen_before"] = new_as_before
        s["as_screen_after"] = new_as_after

    data = [s for s in data if len(s.get("as_screen_before", [])) >= int(min_as_screen_before)]
    print(f"Total samples after filtering: {len(data)}")

    app_data = defaultdict(list)
    for s in data:
        app_data[s.get("app", "unknown")].append(s)

    for app, app_samples in app_data.items():
        print(f"App: {app}, Total samples: {len(app_samples)}")

    json.dump(data, open(output_path, "w"), indent=2, ensure_ascii=False)
    print(f"Saved filtered data to: {output_path}")
    print(
        f"as_screen_before filtered: {filtered_before_total} / {before_total} "
        f"({(filtered_before_total / before_total * 100) if before_total else 0:.2f}%)"
    )

    return data


import os
import base64
from io import BytesIO
from PIL import Image, ImageDraw
from openai import OpenAI
from utils import get_client

SYSTEM_PROMPT = """You are a GUI screenshot analysis expert. You will be provided with:
1. A screenshot of a UI screen (Screen Before) with the action area marked in red
2. The action type performed
3. The resulting screenshot after the action (Screen After)
4. The name of the Android app

Your task is to analyze the elements on the **second screenshot (Screen After)** ONLY. The first screenshot is provided only as context to help you understand the app's state.

Each element should be output as a dictionary:
{
    "type": "functionality" or "data",  // Most elements are functionalities provided by the app, such as options or buttons; Some elements are user data in the app, such as calendar events, set alarms, etc.
    "label": "A short phrase describing its identifier on this screen",  // e.g., "Settings button in top-left corner"
    "description": "A few sentences describing this element's functionality"  // This should clearly and comprehensively describe the functionality this element provides to users.
}

The description should be **comprehensive and detailed**:
- Include the hierarchical location within the app (e.g., which menu, which settings page, which sub-section)
- Explain what this element does **at the phone/device level**, so that someone reading this description can fully understand the element's role and functionality without seeing the screenshot.
Here are examples showing bad descriptions and their improved versions:

<Example_1>
- Bad description: "A WiFi toggle that enables or disables WiFi connectivity."
- Reason: Too vague; it does not specify where the element is located which app or what changes at the device level.
- Good description: "This toggle under System Settings > Network & Internet > Wi‑Fi enables or disables Wi‑Fi on the device, allowing the phone to scan for available wireless networks and connect/disconnect from them."
</Example_1>

<Example_2>
- Bad description: "A Reminder option enables users to set a reminder."
- Reason: Too vague; it does not explain what scenario it is used for.
- Good description: "In the calendar app's event creation/edit screen, this reminder option schedules a notification before the event starts (e.g., 10 minutes in advance), helping the user receive an alert at the chosen lead time."
</Example_2>


Output a JSON list only, for example:
[
{"type": "functionality", "label": "xxx", "description": "xxx"},
{"type": "data", "label": "xxx", "description": "xxx"},
...
]

IMPORTANT: Output ONLY a valid JSON array. No markdown, no comments, no extra text. Start with [ and end with ]."""


def draw_action_on_image(img, action_item):
    """Draw action bbox on image"""
    img_draw = img.copy()
    draw = ImageDraw.Draw(img_draw)
    bbox = action_item.get('bbox')
    if bbox:  # bbox exists and is not None
        draw.rectangle(
            [(bbox['x_min'], bbox['y_min']), (bbox['x_max'], bbox['y_max'])],
            outline='red', width=4
        )
    return img_draw


def image_to_base64(img):
    """Convert PIL Image to base64 string"""
    buffer = BytesIO()
    img.save(buffer, format='PNG')
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


from concurrent.futures import ThreadPoolExecutor, as_completed

def parse_result(result_str):
    """Parse model output as a list. Return None on failure."""
    try:
        # Extract the outermost JSON array if extra text is present.
        start = result_str.find('[')
        end = result_str.rfind(']')
        if start != -1 and end != -1 and start < end:
            result_str = result_str[start:end+1]
        return json.loads(result_str)
    except:
        return None


def process_screen(screen_idx, screen):
    """Process one screen and return (idx, elements)."""
    try:
        screen_after_name = screen["screen"]
        app_name = screen["app"]
        
        img_after = Image.open(os.path.join(SCREENSHOTS_DIR, screen_after_name))
        img_after_b64 = image_to_base64(img_after)
        
        if screen["as_screen_after"]:
            screen_as_after_item = index.data[screen["as_screen_after"][0]]
            screen_before_name = screen_as_after_item["screen_before"]
            action_type = screen_as_after_item["action_type"]
            print(f"screen before: {screen_before_name}, screen after: {screen_after_name}, action type: {action_type}")
            
            img_before = Image.open(os.path.join(SCREENSHOTS_DIR, screen_before_name))
            img_before_with_action = draw_action_on_image(img_before, screen_as_after_item)
            img_before_b64 = image_to_base64(img_before_with_action)
            
            user_content = [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_before_b64}"}},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_after_b64}"}},
                {"type": "text", "text": f"App: {app_name}\nAction: {action_type}\n\nThe first image is Screen Before (with action area marked in red). The second image is Screen After. Please analyze the elements on the second image."},
            ]
        else:
            user_content = [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_after_b64}"}},
                {"type": "text", "text": f"App: {app_name}\n\nNote: There is no preceding screen context available. Only one screenshot is provided. Please analyze the elements on this image."},
            ]
        
        response = client.chat.completions.create(
            model="gemini-3-pro-preview",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content}
            ]
        )
        
        result = response.choices[0].message.content
        elements = parse_result(result)
        if elements == None:
            print("Elements is None")
            print(response)
        return screen_idx, elements
    except Exception as e:
        print(f"Error processing {screen['screen']}: {e}")
        return screen_idx, None


if __name__ == "__main__":
    # NOTE: The block below preserves the original standalone element-annotation
    # workflow, but uses generic local paths and the shared OpenAI-compatible
    # client config so the file can be safely published.

    SCREENSHOTS_DIR = "screenshots"

    client = get_client()

    index = create_index(
        data_path="state_transfer_explore.json",
        screenshots_dir=SCREENSHOTS_DIR,
        cache_path="screen_transition_cache.pkl",
        similarity_threshold=0.95,
    )

    # Resume from a partially written with-elements file if it exists.
    RESUME_PATH = "unique_screen_with_elements.json"
    FALLBACK_PATH = "unique_screen.json"

    if os.path.exists(RESUME_PATH):
        unique_screens = json.load(open(RESUME_PATH, "r"))
        print(f"Loaded resume file: {RESUME_PATH}")
    else:
        unique_screens = json.load(open(FALLBACK_PATH, "r"))
        print(f"Loaded fallback file: {FALLBACK_PATH}")

    print(f"Number of unique screens: {len(unique_screens)}")

    OUTPUT_PATH = "unique_screen_with_elements.json"

    # Process only samples whose `elements` field is still None.
    to_process = [(i, s) for i, s in enumerate(unique_screens) if s.get("elements") is None]
    print(f"To process: {len(to_process)} / {len(unique_screens)}")

    NUM_THREADS = 64
    completed = 0

    with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
        futures = {executor.submit(process_screen, i, s): i for i, s in to_process}

        for future in tqdm(as_completed(futures), total=len(to_process)):
            idx, elements = future.result()
            unique_screens[idx]["elements"] = elements

            completed += 1
            if completed % 20 == 0:
                json.dump(unique_screens, open(OUTPUT_PATH, "w"), indent=2, ensure_ascii=False)
                print(f"Progress saved: {completed}/{len(to_process)}")

        json.dump(unique_screens, open(OUTPUT_PATH, "w"), indent=2, ensure_ascii=False)
    print(f"Done! Saved to {OUTPUT_PATH}")