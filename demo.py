import datetime
import json
import os
import threading
import logging
from queue import Queue
import gradio as gr
import config
from explor_auto import run_task
from chain_evolve import evolve_chain_to_action
from chain_understand import process_and_update_chain, Neo4jDatabase
from data.State import State
from data.data_storage import state2json, json2db
from explor_human import single_human_explor, capture_and_parse_page
from tool.screen_content import list_all_devices, get_device_size, set_controller
from device import ADBController

# =============================================================================
# DEBUG LOGGING SETUP
# =============================================================================
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('/tmp/demo_debug.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
logger.info("="*60)
logger.info("DEMO.PY STARTED - Debug logging enabled")
logger.info("="*60)

auto_log_storage = []  # Global log storage for automatic exploration
auto_page_storage = []  # Global page history storage for automatic exploration
user_log_storage = []  # Global log storage for user exploration
user_page_storage = []  # Global page history storage for user exploration
temp_state = None  # Temporary state storage for saving user exploration state


def update_inputs(action):
    if action == "tap" or action == "long press":
        return (
            gr.update(visible=True),
            gr.update(visible=False),
            gr.update(visible=False),
        )
    elif action == "text":
        return (
            gr.update(visible=True),
            gr.update(visible=True),
            gr.update(visible=False),
        )
    elif action == "swipe":
        return (
            gr.update(visible=True),
            gr.update(visible=False),
            gr.update(visible=True),
        )
    else:
        return (
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
        )


def get_adb_devices():
    devices = list_all_devices()
    return devices if devices else ["No devices found"]


def initialize_device(device, task_info, save_screenshots=True):
    global temp_state
    logger.info("="*40)
    logger.info("INITIALIZE_DEVICE called")
    logger.info(f"  device param: '{device}'")
    logger.info(f"  task_info param: '{task_info}'")
    logger.info(f"  save_screenshots: {save_screenshots}")
    logger.info("="*40)

    if not task_info:
        logger.error("INIT FAILED: task_info is empty")
        return "Error: Task Information cannot be empty."
    if not device or device == "No devices found":
        logger.error("INIT FAILED: no valid device")
        return "Error: Please select a valid ADB device."

    # Generate unique task ID (timestamp-based)
    task_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    logger.info(f"Generated task_id: {task_id}")

    # Create and set the device controller
    logger.info(f"Creating ADBController for device: {device}")
    controller = ADBController(device_id=device)
    set_controller(controller)

    # Get device size from the controller
    width, height = controller.screen_size
    device_size = {"width": width, "height": height}
    logger.info(f"Device size: {device_size}")

    temp_state = State(
        task_id=task_id,
        save_screenshots=save_screenshots,
        tsk=task_info,
        app_name="",
        completed=False,
        agent_signals_complete=False,
        step=0,
        history_steps=[],
        page_history=[],
        current_page_screenshot=None,
        recommend_action="",
        clicked_elements=[],
        action_reflection=[],
        tool_results=[],
        device=device,
        device_info=device_size,
        controller=controller,
        context=[],
        errors=[],
        current_page_json=None,
        callback=None,
    )

    # Create task folder and save metadata
    task_folder = f"./log/tasks/{task_id}"
    os.makedirs(task_folder, exist_ok=True)
    os.makedirs(f"{task_folder}/screenshots", exist_ok=True)

    # Save task metadata
    task_metadata = {
        "task_id": task_id,
        "task_description": task_info,
        "device": device,
        "device_info": device_size,
        "save_screenshots": save_screenshots,
        "created_at": datetime.datetime.now().isoformat(),
        "status": "initialized"
    }
    with open(f"{task_folder}/task_meta.json", "w") as f:
        json.dump(task_metadata, f, indent=2)

    logger.info(f"Task folder created: {task_folder}")

    logger.info("temp_state created successfully:")
    logger.info(f"  task_id: '{temp_state.get('task_id', 'NOT SET')}'")
    logger.info(f"  tsk: '{temp_state.get('tsk', 'NOT SET')}'")
    logger.info(f"  device: '{temp_state.get('device', 'NOT SET')}'")
    logger.info(f"  controller: {temp_state.get('controller', 'NOT SET')}")
    logger.info(f"  step: {temp_state.get('step', 'NOT SET')}")

    return f"Task ID: {task_id}\nDevice: {device}\nTask: {task_info}\nSave Screenshots: {save_screenshots}"


def auto_exploration():
    global auto_log_storage, auto_page_storage, temp_state

    logger.info("="*60)
    logger.info("AUTO_EXPLORATION called")
    logger.info("="*60)

    # Helper function to get current timestamp with timezone and AM/PM
    def get_timestamp():
        return datetime.datetime.now().astimezone().strftime("%I:%M:%S %p %Z")

    if not temp_state:
        logger.error("AUTO_EXPLORATION: temp_state is None!")
        logger.error("User did not initialize device/task first")
        auto_log_storage.append(f"[{get_timestamp()}] Error: Please initialize device and task info first.")
        yield "\n".join(auto_log_storage), auto_page_storage, gr.update(value="Start Exploration", interactive=True)
        return

    # Log the temp_state before exploration
    logger.info("temp_state BEFORE exploration:")
    logger.info(f"  tsk: '{temp_state.get('tsk', 'NOT SET')}'")
    logger.info(f"  app_name: '{temp_state.get('app_name', 'NOT SET')}'")
    logger.info(f"  device: '{temp_state.get('device', 'NOT SET')}'")
    logger.info(f"  controller: {temp_state.get('controller', 'NOT SET')}")
    logger.info(f"  step: {temp_state.get('step', 'NOT SET')}")
    logger.info(f"  completed: {temp_state.get('completed', 'NOT SET')}")
    logger.info(f"  history_steps count: {len(temp_state.get('history_steps', []))}")

    # Mark exploration as in progress
    exploration_start_time = datetime.datetime.now()
    auto_log_storage.append(f"[{get_timestamp()}] 🚀 Exploration started")
    yield "\n".join(auto_log_storage), auto_page_storage, gr.update(value="⏳ Exploration in Progress...", interactive=False)

    q = Queue()
    final_state_queue = Queue()

    def callback(state, node_name=None, info=None):
        timestamp = get_timestamp()
        logger.info(f"CALLBACK: node={node_name}, step={state.get('step')}")
        auto_log_storage.append("")
        auto_log_storage.append("=" * 50)

        if info and isinstance(info, dict):
            # Display step and action details prominently
            step_num = info.get("step", state.get("step", "?"))

            if node_name == "page_understand_start":
                # Intermediate: Starting screenshot capture
                auto_log_storage.append(f"[{timestamp}] 📸 STEP {step_num}: Starting Screenshot")
                auto_log_storage.append("-" * 30)
                if info.get("message"):
                    auto_log_storage.append(info["message"])

            elif node_name == "screenshot_captured":
                # Intermediate: Screenshot captured, sending to OmniParser
                auto_log_storage.append(f"[{timestamp}] 🔄 STEP {step_num}: Analyzing Screen")
                auto_log_storage.append("-" * 30)
                if info.get("message"):
                    auto_log_storage.append(info["message"])
                # Show screenshot immediately
                screenshot_path = info.get("screenshot_path")
                if screenshot_path and screenshot_path not in auto_page_storage:
                    auto_page_storage.append(screenshot_path)
                    logger.info(f"  CALLBACK: Added raw screenshot: {screenshot_path}")

            elif node_name == "page_understand":
                auto_log_storage.append(f"[{timestamp}] ✅ STEP {step_num}: Screen Analysis Complete")
                auto_log_storage.append("-" * 30)
                if info.get("message"):
                    auto_log_storage.append(info["message"])
                # Show timing breakdown if available
                if info.get("screenshot_time") and info.get("omniparser_time"):
                    auto_log_storage.append(f"   ⏱️ Screenshot: {info['screenshot_time']:.1f}s | OmniParser: {info['omniparser_time']:.1f}s")

            elif node_name == "gridded_screenshot":
                # Show the gridded image that will be sent to Claude
                auto_log_storage.append(f"[{timestamp}] 🔲 STEP {step_num}: Grid Overlay Created")
                auto_log_storage.append("-" * 30)
                if info.get("message"):
                    auto_log_storage.append(info["message"])
                # Add gridded image to gallery so user can see what Claude sees
                gridded_image = info.get("labeled_image_path")
                if gridded_image and gridded_image not in auto_page_storage:
                    auto_page_storage.append(gridded_image)
                    logger.info(f"  CALLBACK: Added gridded screenshot BEFORE LLM call: {gridded_image}")

            elif node_name == "action_agent_thinking":
                # Intermediate: LLM is analyzing and deciding
                auto_log_storage.append(f"[{timestamp}] 🤔 STEP {step_num}: Claude Thinking")
                auto_log_storage.append("-" * 30)
                if info.get("message"):
                    auto_log_storage.append(info["message"])

            elif node_name == "perform_action":
                action_details = info.get("action_details", "Unknown action")
                llm_time = info.get("llm_time", 0)
                auto_log_storage.append(f"[{timestamp}] 🤖 STEP {step_num}: Action Executed")
                auto_log_storage.append("-" * 30)

                # Show LLM reasoning FIRST (before action)
                llm_reasoning = info.get("llm_reasoning", "")
                if llm_reasoning:
                    auto_log_storage.append("💭 PRE-ACTION REASONING:")
                    # Wrap long reasoning text
                    for line in llm_reasoning.split('\n'):
                        if line.strip():
                            auto_log_storage.append(f"   {line.strip()}")
                    auto_log_storage.append("")

                # Then show the action taken
                auto_log_storage.append(f"▶️ ACTION: {action_details}")
                if llm_time:
                    auto_log_storage.append(f"   ⏱️ Claude response time: {llm_time:.1f}s")

                # Show actual ADB command executed (critical for debugging)
                adb_command = info.get("adb_command", "")
                adb_status = info.get("adb_status", "")
                if adb_command:
                    auto_log_storage.append(f"🔧 [ADB EXECUTED]: {adb_command}")
                    auto_log_storage.append(f"   Status: {adb_status.upper()}")
                else:
                    auto_log_storage.append(f"⚠️ [ADB]: No command captured - tool may not have been called!")
                auto_log_storage.append("")

            elif node_name == "tsk_setting":
                auto_log_storage.append(f"[{timestamp}] ⚙️ Task Setup")
                auto_log_storage.append("-" * 30)
                auto_log_storage.append(f"Task: {info.get('task', 'N/A')}")
                auto_log_storage.append(f"App: {info.get('app_name', 'N/A')}")

            elif node_name == "task_judgment":
                auto_log_storage.append(f"[{timestamp}] 🔍 STEP {step_num}: Task Completion Check")
                auto_log_storage.append("-" * 30)

                # Show completion criteria
                criteria = info.get("completion_criteria", "")
                if criteria:
                    auto_log_storage.append("📋 COMPLETION CRITERIA:")
                    for line in criteria.split('\n')[:5]:  # Limit to 5 lines
                        if line.strip():
                            auto_log_storage.append(f"   {line.strip()}")
                    auto_log_storage.append("")

                # Show judge reasoning (the important part!)
                judge_reasoning = info.get("judge_reasoning", "")
                if judge_reasoning:
                    auto_log_storage.append("⚖️ JUDGE LLM ANALYSIS:")
                    for line in judge_reasoning.split('\n'):
                        if line.strip():
                            auto_log_storage.append(f"   {line.strip()}")
                    auto_log_storage.append("")

                # Show verdict prominently
                verdict = info.get("verdict", "UNKNOWN")
                if "COMPLETE" in verdict and "NOT" not in verdict:
                    auto_log_storage.append(f"✅ VERDICT: {verdict}")
                else:
                    auto_log_storage.append(f"❌ VERDICT: {verdict}")

                # Add the judgment screenshot
                screenshot_path = info.get("screenshot_path")
                if screenshot_path and screenshot_path not in auto_page_storage:
                    auto_page_storage.append(screenshot_path)
                    logger.info(f"  CALLBACK: Added judgment screenshot: {screenshot_path}")

            else:
                auto_log_storage.append(f"[{timestamp}] Step {step_num}: {node_name}")
                for k, v in info.items():
                    if k not in ["llm_reasoning", "tool_output", "labeled_image_path", "judge_reasoning", "completion_criteria"]:
                        auto_log_storage.append(f"   {k}: {v}")

            # Add image from info directly if available
            labeled_image = info.get("labeled_image_path")
            if labeled_image and labeled_image not in auto_page_storage:
                auto_page_storage.append(labeled_image)
                logger.info(f"  CALLBACK: Added image from info: {labeled_image}")
        else:
            auto_log_storage.append(f"Step {state.get('step', '?')}: {node_name}")

        # Also check tool_results for any images we might have missed
        if state.get("tool_results"):
            for tool_result in state["tool_results"]:
                if isinstance(tool_result, dict):
                    result = tool_result.get("result", {})
                    if isinstance(result, dict):
                        labeled_image = result.get("labeled_image_path")
                        if labeled_image and labeled_image not in auto_page_storage:
                            auto_page_storage.append(labeled_image)
                            logger.info(f"  CALLBACK: Added image from tool_results: {labeled_image}")

        q.put(("\n".join(auto_log_storage), auto_page_storage))

    def run_exploration():
        logger.info("run_exploration THREAD started")
        try:
            logger.info("Calling run_task(temp_state, callback)...")
            final_state = run_task(temp_state, callback)
            logger.info("run_task returned!")
            logger.info(f"  final_state type: {type(final_state)}")
            if final_state:
                logger.info(f"  final_state.tsk: '{final_state.get('tsk', 'NOT SET')}'")
                logger.info(f"  final_state.step: {final_state.get('step', 'NOT SET')}")
                logger.info(f"  final_state.history_steps count: {len(final_state.get('history_steps', []))}")
                logger.info(f"  final_state.app_name: '{final_state.get('app_name', 'NOT SET')}'")
            else:
                logger.error("  final_state is None!")
            final_state_queue.put(final_state)
        except Exception as e:
            logger.exception(f"run_exploration EXCEPTION: {e}")
            final_state_queue.put(None)

    # Start the exploration task in a new thread
    logger.info("Starting exploration thread...")
    t = threading.Thread(target=run_exploration)
    t.start()

    # Continuously get status updates from the queue
    logger.info("Waiting for exploration thread...")
    while t.is_alive() or not q.empty():
        try:
            msg, pages = q.get(timeout=1)
            yield msg, pages, gr.update(value="⏳ Exploration in Progress...", interactive=False)
        except:
            pass

    logger.info("Exploration thread finished")
    exploration_end_time = datetime.datetime.now()
    elapsed = exploration_end_time - exploration_start_time
    elapsed_str = str(elapsed).split('.')[0]  # Remove microseconds
    auto_log_storage.append(f"[{get_timestamp()}] ✅ Exploration finished. Total time: {elapsed_str}")

    # Get the final state from the queue
    try:
        final_state = final_state_queue.get(timeout=5)  # Wait for the final state
        logger.info("Got final_state from queue")
        if final_state:
            logger.info(f"  final_state.tsk: '{final_state.get('tsk', 'NOT SET')}'")
            logger.info(f"  final_state.history_steps: {len(final_state.get('history_steps', []))}")

            # Update task metadata with final status
            task_id = final_state.get('task_id')
            if task_id:
                task_folder = f"./log/tasks/{task_id}"
                meta_file = f"{task_folder}/task_meta.json"
                if os.path.exists(meta_file):
                    with open(meta_file, "r") as f:
                        task_metadata = json.load(f)
                    task_metadata["status"] = "completed" if final_state.get("completed") else "incomplete"
                    task_metadata["app_name"] = final_state.get("app_name", "")
                    task_metadata["total_steps"] = final_state.get("step", 0)
                    task_metadata["completed_at"] = datetime.datetime.now().isoformat()
                    with open(meta_file, "w") as f:
                        json.dump(task_metadata, f, indent=2)
                    logger.info(f"Updated task metadata: {meta_file}")

                    # Rename folder to include app_name if available
                    app_name = final_state.get("app_name", "").strip()
                    if app_name:
                        new_folder = f"./log/tasks/{task_id}_{app_name}"
                        if not os.path.exists(new_folder):
                            import shutil
                            old_folder = task_folder
                            shutil.move(task_folder, new_folder)
                            logger.info(f"Renamed task folder to: {new_folder}")
                            task_folder = new_folder

                            # Update all image paths in auto_page_storage to use new folder
                            updated_pages = []
                            for img_path in auto_page_storage:
                                if old_folder in img_path:
                                    new_path = img_path.replace(old_folder, new_folder)
                                    updated_pages.append(new_path)
                                    logger.info(f"Updated image path: {img_path} -> {new_path}")
                                else:
                                    # Handle relative paths like "log/tasks/..."
                                    old_rel = f"log/tasks/{task_id}"
                                    new_rel = f"log/tasks/{task_id}_{app_name}"
                                    if old_rel in img_path:
                                        new_path = img_path.replace(old_rel, new_rel)
                                        updated_pages.append(new_path)
                                        logger.info(f"Updated image path: {img_path} -> {new_path}")
                                    else:
                                        updated_pages.append(img_path)
                            auto_page_storage.clear()
                            auto_page_storage.extend(updated_pages)
                            logger.info(f"Updated {len(updated_pages)} image paths after folder rename")

                            # Also update paths in final_state.history_steps
                            if final_state.get("history_steps"):
                                for step in final_state["history_steps"]:
                                    if step.get("source_page") and old_rel in step["source_page"]:
                                        step["source_page"] = step["source_page"].replace(old_rel, new_rel)
                                    if step.get("source_json") and old_rel in step["source_json"]:
                                        step["source_json"] = step["source_json"].replace(old_rel, new_rel)
                                logger.info(f"Updated paths in final_state.history_steps")

                            # Update current_page_screenshot path
                            if final_state.get("current_page_screenshot") and old_rel in final_state["current_page_screenshot"]:
                                final_state["current_page_screenshot"] = final_state["current_page_screenshot"].replace(old_rel, new_rel)

                            # Update current_page_json path if it's a string
                            if isinstance(final_state.get("current_page_json"), str) and old_rel in final_state["current_page_json"]:
                                final_state["current_page_json"] = final_state["current_page_json"].replace(old_rel, new_rel)
                            elif isinstance(final_state.get("current_page_json"), dict):
                                for key in final_state["current_page_json"]:
                                    if isinstance(final_state["current_page_json"][key], str) and old_rel in final_state["current_page_json"][key]:
                                        final_state["current_page_json"][key] = final_state["current_page_json"][key].replace(old_rel, new_rel)

                # Handle screenshot cleanup if disabled
                save_screenshots = final_state.get("save_screenshots", True)
                if not save_screenshots:
                    screenshots_dir = f"{task_folder}/screenshots"
                    if os.path.exists(screenshots_dir):
                        import shutil
                        shutil.rmtree(screenshots_dir)
                        os.makedirs(screenshots_dir)  # Keep empty folder
                        logger.info(f"Deleted screenshots (save_screenshots=False)")
                        auto_log_storage.append("Screenshots deleted (retention disabled)")
        else:
            logger.error("  final_state is None!")
        # Convert the result to JSON format and store it
        state2json_result = state2json(final_state)
        logger.info(f"state2json result: {state2json_result}")
        auto_log_storage.append(state2json_result)
    except Exception as e:
        logger.exception(f"Failed to get final state: {e}")
        auto_log_storage.append(f"[{get_timestamp()}] Error: Failed to get final state")

    yield "\n".join(auto_log_storage), auto_page_storage, gr.update(value="✅ Exploration Complete", interactive=False)


def user_exploration(action, element_number, text_input, swipe_direction):
    global user_log_storage, user_page_storage, temp_state

    if not temp_state:
        user_log_storage.append("Error: Please initialize device and task info first.")
        return "\n".join(user_log_storage), user_page_storage

    # Call the single human exploration logic
    temp_state = single_human_explor(
        temp_state,
        action,
        element_number=element_number,
        text_input=text_input,
        swipe_direction=swipe_direction,
    )

    # Log additional operation details
    log_entry = f"Step: {temp_state.get('step', 0)} | Action: {action}"
    if action in ["tap", "long press"]:
        log_entry += f" on element {element_number}"
    elif action == "text":
        log_entry += f" with text input '{text_input}' on element {element_number}"
    elif action == "swipe":
        log_entry += f" swiped {swipe_direction} on element {element_number}"

    # Check task completion status
    if temp_state.get("completed", False):
        log_entry += " | Task completed."
    else:
        log_entry += " | Task in progress."

    user_log_storage.append(log_entry)

    # Check for updates in page_history and update page storage
    if "page_history" in temp_state:
        for page in temp_state["page_history"]:
            if page not in user_page_storage:
                user_page_storage.append(page)

    # Add the latest labeled screenshot to the page storage
    if temp_state.get("history_steps"):
        latest_step = temp_state["history_steps"][-1]
        if latest_step.get("source_json"):
            for tool_result in reversed(temp_state["tool_results"]):
                if tool_result.get("tool_name") == "screen_element":
                    labeled_image_path = tool_result["result"].get("labeled_image_path")
                    if (
                        labeled_image_path
                        and labeled_image_path not in user_page_storage
                    ):
                        user_page_storage.append(labeled_image_path)
                    break

    return "\n".join(user_log_storage), user_page_storage


def self_evolution(task, data):
    return f"Task: {task} with Data: {data} is evolving."


def get_json_files(directory: str = "./log/json_state") -> list:
    """Get all JSON files and their details in the specified directory"""
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
        json_files = []
        for file in os.listdir(directory):
            if file.endswith(".json"):
                file_path = os.path.join(directory, file)
                mod_time = os.path.getmtime(file_path)
                mod_time_str = datetime.datetime.fromtimestamp(mod_time).strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = json.load(f)
                        steps_count = len(content.get("history_steps", []))
                        task_desc = content.get("tsk", "N/A")
                        app_name = content.get("app_name", "N/A")
                except Exception as e:
                    print(f"Error reading JSON file {file}: {str(e)}")
                    steps_count = 0
                    task_desc = "Error reading file"
                    app_name = "N/A"

                json_files.append(
                    {
                        "name": file,
                        "path": file_path,
                        "modified": mod_time_str,
                        "steps": steps_count,
                        "task": task_desc,
                        "app": app_name,
                    }
                )
        return sorted(json_files, key=lambda x: x["modified"], reverse=True)
    except Exception as e:
        print(f"Error reading directory: {str(e)}")
        return []


with gr.Blocks(
    css="""
    #json_files_table table { border-collapse: collapse; width: 100%; }
    #json_files_table thead { background-color: #f3f4f6; }
    #json_files_table th { padding: 10px; text-align: left; border-bottom: 2px solid #e5e7eb; }
    #json_files_table td { padding: 8px 10px; border-bottom: 1px solid #e5e7eb; }
    #json_files_table tr:hover { background-color: #f9fafb; }
    #json_files_table .scroll-hide { overflow-x: hidden !important; }
    #json_files_table { user-select: none; }
    #auto_exploration_log textarea { font-family: monospace; font-size: 12px; }
    """,
    js="""
    function setupAutoScroll() {
        // Auto-scroll log panels to bottom when content changes
        const observer = new MutationObserver(function(mutations) {
            mutations.forEach(function(mutation) {
                if (mutation.type === 'childList' || mutation.type === 'characterData') {
                    const target = mutation.target;
                    // Find the textarea inside the log panel
                    let textarea = target;
                    if (!textarea.tagName || textarea.tagName.toLowerCase() !== 'textarea') {
                        textarea = target.querySelector('textarea');
                    }
                    if (textarea && textarea.tagName && textarea.tagName.toLowerCase() === 'textarea') {
                        textarea.scrollTop = textarea.scrollHeight;
                    }
                }
            });
        });

        // Observe the exploration log container
        function observeLogPanels() {
            const logPanels = document.querySelectorAll('#auto_exploration_log, .prose');
            logPanels.forEach(function(panel) {
                observer.observe(panel, { childList: true, subtree: true, characterData: true });
            });

            // Also observe textareas directly
            const textareas = document.querySelectorAll('textarea');
            textareas.forEach(function(textarea) {
                const parentObserver = new MutationObserver(function() {
                    textarea.scrollTop = textarea.scrollHeight;
                });
                if (textarea.parentElement) {
                    parentObserver.observe(textarea.parentElement, { childList: true, subtree: true });
                }
            });
        }

        // Initial setup and re-setup on DOM changes
        observeLogPanels();
        setInterval(observeLogPanels, 2000);

        return [];
    }
    """
) as demo:
    with gr.Tabs():
        # Tab 1: Initialization
        with gr.Tab("Initialization"):
            gr.Markdown("### Initialization Page")
            devices_output = gr.Textbox(
                label="Available ADB Devices", interactive=False
            )
            refresh_button = gr.Button("Refresh Device List")
            device_selector = gr.Radio(
                label="Select ADB Device", choices=[]
            )  # Updated dynamically
            task_info_input = gr.Textbox(
                label="Set Task Information", placeholder="e.g., Task Description"
            )
            save_screenshots_checkbox = gr.Checkbox(
                label="Save screenshots after task completion",
                value=True,
                info="Uncheck to delete screenshots when task finishes (saves disk space)"
            )
            init_button = gr.Button("Initialize")
            init_output = gr.Textbox(label="Initialization Output", interactive=False)

            def update_devices():
                devices = get_adb_devices()
                return gr.update(choices=devices), "\n".join(devices)

            refresh_button.click(
                update_devices, inputs=[], outputs=[device_selector, devices_output]
            )

            init_button.click(
                initialize_device,
                inputs=[device_selector, task_info_input, save_screenshots_checkbox],
                outputs=[init_output],
            )

        # Tab 2: Auto Exploration
        with gr.Tab("Auto Exploration"):
            gr.Markdown("### Auto Exploration Page")
            with gr.Row():
                with gr.Column():
                    exploration_output = gr.TextArea(
                        label="Exploration Log", interactive=False,
                        elem_id="auto_exploration_log", lines=20
                    )
                    explore_button = gr.Button("Start Exploration")
                    clear_images_button = gr.Button(
                        "Clear Images", elem_id="clear-images-button"
                    )

                with gr.Column():
                    screenshot_gallery_auto = gr.Gallery(
                        label="Screenshots", height=700
                    )
            explore_button.click(
                fn=auto_exploration,
                outputs=[
                    exploration_output,
                    screenshot_gallery_auto,
                    explore_button,
                ],
                queue=True,
                api_name="explore",
            )

            def clear_images():
                return []

            clear_images_button.click(
                fn=clear_images,
                inputs=[],
                outputs=[screenshot_gallery_auto],
            )

        # Tab 3: User Exploration
        with gr.Tab("User Exploration"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### User Exploration Page")
                    with gr.Row():
                        start_button = gr.Button(
                            "Start Session", elem_id="start-button"
                        )
                        stop_button = gr.Button(
                            "Stop Session", elem_id="stop-button", interactive=False
                        )
                    action = gr.Radio(
                        ["tap", "text", "long press", "swipe", "wait"],
                        label="Action",
                        interactive=False,
                    )
                    element_number = gr.Number(
                        label="Element Number",
                        precision=0,
                        visible=False,
                        interactive=False,
                    )
                    text_input = gr.Textbox(
                        label="Text Input", visible=False, interactive=False
                    )
                    swipe_direction = gr.Radio(
                        ["up", "down", "left", "right"],
                        label="Swipe Direction",
                        visible=False,
                        interactive=False,
                    )
                    action_button = gr.Button("Perform Action", interactive=False)
                    human_demo_output = gr.Textbox(
                        label="Action Output", interactive=False
                    )

                with gr.Column():
                    screenshot_gallery_user = gr.Gallery(
                        label="Screenshots", height=700
                    )

            action.change(
                update_inputs,
                inputs=[action],
                outputs=[element_number, text_input, swipe_direction],
            )

            def start_session():
                global temp_state
                if not temp_state:
                    return None

                temp_state = capture_and_parse_page(
                    temp_state
                )  # Capture and initialize the first screenshot
                # Check for updates in page_history and update page storage
                if "page_history" in temp_state:
                    for page in temp_state["page_history"]:
                        if page not in user_page_storage:
                            user_page_storage.append(page)
                return (
                    gr.update(interactive=False),  # Disable start button
                    gr.update(interactive=True),  # Enable stop button
                    gr.update(interactive=True),  # Enable action selection
                    gr.update(interactive=True),  # Enable action button
                    gr.update(interactive=True),  # Enable Element Number
                    gr.update(interactive=True),  # Enable Text Input
                    gr.update(interactive=True),  # Enable Swipe Direction
                    gr.update(interactive=True),  # Enable store_to_db_btn
                    human_demo_output,  # Return output box
                    user_page_storage,  # Return page history
                )

            def stop_session():
                global temp_state
                if temp_state:
                    temp_state["completed"] = True
                user_log_storage.append("Session stopped.")
                user_page_storage.clear()
                state2json_result = state2json(temp_state)
                user_log_storage.append(state2json_result)
                return (
                    gr.update(interactive=True),  # Enable start button
                    gr.update(interactive=False),  # Disable stop button
                    gr.update(interactive=False),  # Disable action selection
                    gr.update(interactive=False),  # Disable action button
                    gr.update(interactive=False),  # Disable Element Number
                    gr.update(interactive=False),  # Disable Text Input
                    gr.update(interactive=False),  # Disable Swipe Direction
                    gr.update(interactive=False),  # Disable store_to_db_btn
                    "\n".join(user_log_storage),  # Return logs
                    user_page_storage,  # Return page history
                )

            action_button.click(
                user_exploration,
                inputs=[action, element_number, text_input, swipe_direction],
                outputs=[human_demo_output, screenshot_gallery_user],
            )

            start_button.click(
                start_session,
                inputs=[],
                outputs=[
                    start_button,
                    stop_button,
                    action,
                    action_button,
                    element_number,
                    text_input,
                    swipe_direction,
                    human_demo_output,
                    screenshot_gallery_user,
                ],
            )

            stop_button.click(
                stop_session,
                inputs=[],
                outputs=[
                    start_button,
                    stop_button,
                    action,
                    action_button,
                    element_number,
                    text_input,
                    swipe_direction,
                    human_demo_output,
                    screenshot_gallery_user,
                ],
            )

        # Tab 4: Chain Understanding & Evolution
        with gr.Tab("Chain Understanding & Evolution"):
            with gr.Column():
                with gr.Row():
                    # Left side: File selection
                    with gr.Column(scale=1):
                        json_files = gr.Radio(
                            label="Select Operation Record File",
                            choices=[],
                            value=None,
                            interactive=True,
                            elem_id="json_files_radio",
                        )

                    # Right side: File details
                    with gr.Column(scale=1):
                        file_info = gr.Markdown(
                            value="Please select a file to view details",
                            visible=True,
                            elem_id="file_info",
                        )

                evolution_log = gr.TextArea(
                    label="Execution Log", interactive=False, lines=8
                )

                with gr.Row():
                    refresh_btn = gr.Button(
                        "1. Refresh File List", variant="secondary", scale=1
                    )
                    store_to_db_btn = gr.Button(
                        "2. Store Record to Database", variant="primary", scale=1
                    )
                    understand_btn = gr.Button(
                        "3. Understand Operation Path", variant="primary", scale=1
                    )
                    generate_btn = gr.Button(
                        "4. Generate Advanced Path", variant="primary", scale=1
                    )

            # Store file list global variable
            current_files = []
            # Store current task ID
            current_task_id = None

            def format_file_choice(file_info):
                """Format file information as option text"""
                return file_info["name"]  # Return only file name

            def format_file_details(file_info):
                """Format file details"""
                return f"""
### File Details
- **File Name**: {file_info['name']}
- **Modified Time**: {file_info['modified']}
- **Step Count**: {file_info['steps']}
- **Task Description**: {file_info['task']}
- **Application Name**: {file_info['app']}
"""

            def update_file_list():
                """Update file list and return options"""
                global current_files
                try:
                    files = get_json_files()
                    current_files = files  # Update global variable

                    if not files:
                        return (
                            gr.update(choices=[], value=None),
                            "No JSON files found in directory.",
                            "No JSON files found in directory.",
                        )

                    # Generate option list
                    choices = [format_file_choice(f) for f in files]
                    # Default select first file
                    default_choice = choices[0] if choices else None

                    return (
                        gr.update(choices=choices, value=default_choice),
                        (
                            on_file_select(default_choice)
                            if default_choice
                            else "Please select a file to view details"
                        ),
                        (
                            "File list updated. First file selected as default."
                            if default_choice
                            else "File list updated, but no usable files."
                        ),
                    )
                except Exception as e:
                    import traceback

                    error_msg = (
                        f"Error updating file list: {str(e)}\n{traceback.format_exc()}"
                    )
                    return (
                        gr.update(choices=[], value=None),
                        error_msg,
                        error_msg,
                    )

            def on_file_select(choice):
                """When file is selected, update file details"""
                global current_files
                if not choice or not current_files:
                    return "Please select a file to view details"

                # Find corresponding file information based on selected text
                for file in current_files:
                    if format_file_choice(file) == choice:
                        return format_file_details(file)

                return "Failed to retrieve file information"

            # Bind refresh button event
            refresh_btn.click(
                fn=update_file_list, outputs=[json_files, file_info, evolution_log]
            )

            # Initialize with automatic load of file list
            demo.load(
                fn=update_file_list, outputs=[json_files, file_info, evolution_log]
            )

            # Define a function to refresh task ID from current file
            def refresh_task_id_from_file(selected_file):
                global current_files, current_task_id

                if not selected_file:
                    return "No file selected, cannot get task ID"

                # Get file path
                file_path = None
                for file in current_files:
                    if format_file_choice(file) == selected_file:
                        file_path = file["path"]
                        break

                if not file_path:
                    return "Failed to retrieve selected file path"

                try:
                    # Read JSON file content
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = json.load(f)

                    # Extract task ID from filename
                    import re

                    file_name = os.path.basename(file_path)
                    match = re.search(r"state_(\d+)_", file_name)
                    if match:
                        date_str = match.group(1)
                        # Generate a temporary task ID, format as "temp_" + date part
                        temp_task_id = f"temp_{date_str}"
                        current_task_id = temp_task_id
                        return f"Temporary task ID generated from filename: {temp_task_id}, Note: This is not the actual ID stored in the database"

                    return "Failed to extract date information to generate temporary task ID"
                except Exception as e:
                    return f"Error extracting task ID: {str(e)}"

            # Modify Radio component change event, update file details and attempt to get task ID
            def update_file_and_task_id(selected_file):
                file_details = on_file_select(selected_file)
                task_id_info = refresh_task_id_from_file(selected_file)
                return file_details, task_id_info

            json_files.change(
                fn=update_file_and_task_id,
                inputs=[json_files],
                outputs=[file_info, evolution_log],
            )

            # Define function to store to database
            def store_to_database(selected_file):
                global current_files, current_task_id

                if not selected_file:
                    return "Error: Please select a file first"

                # Find corresponding file information based on selected text
                file_path = None
                for file in current_files:
                    if format_file_choice(file) == selected_file:
                        file_path = file["path"]
                        break

                if not file_path:
                    return "Error: Failed to retrieve selected file path"

                try:
                    # Call json2db function and get task ID
                    task_id = json2db(file_path)
                    current_task_id = task_id  # Ensure update global variable

                    return f"File {selected_file} successfully stored to database!\nTask ID: {task_id}\nThis ID will be used for subsequent operation path understanding."
                except Exception as e:
                    import traceback

                    return (
                        f"Error storing to database: {str(e)}\n{traceback.format_exc()}"
                    )

            # Bind store to database button event
            store_to_db_btn.click(
                fn=store_to_database,
                inputs=[json_files],
                outputs=[evolution_log],
            )

            # Button click event
            def on_button_click(action_type, selected_file):
                if not selected_file:
                    return "Please select a file first"
                return f"{action_type} - Selected file: {selected_file}"

            # Define function to understand operation path
            def understand_chain(selected_file, progress=gr.Progress()):
                global current_files, current_task_id

                if not selected_file:
                    return "Error: Please select a file first"

                # Create variable to store current log status
                current_log = ["Starting to understand operation path..."]

                # Add log function to update UI
                def add_log(message):
                    current_log.append(message)
                    return "\n".join(current_log)

                # Check if task ID exists, if not try to generate from filename
                if not current_task_id:
                    add_log(
                        "Warning: Task ID not found, attempting to generate temporary task ID from filename..."
                    )
                    result = refresh_task_id_from_file(selected_file)
                    add_log(result)

                    if not current_task_id:
                        add_log(
                            "Error: Could not get task ID, cannot continue understanding operation path"
                        )
                        return "\n".join(current_log)

                add_log(f"Current task ID: {current_task_id}")
                progress(0.1, "Initializing database connection...")

                try:
                    # Connect to Neo4j database
                    db = Neo4jDatabase(uri=config.Neo4j_URI, auth=config.Neo4j_AUTH)

                    # Get all start nodes
                    add_log("Getting start nodes...")
                    progress(0.2, "Getting start nodes...")
                    start_nodes = db.get_chain_start_nodes()

                    if not start_nodes:
                        add_log("❌ No start nodes found")
                        return "\n".join(current_log)

                    add_log(f"Found {len(start_nodes)} start nodes")
                    progress(0.3, "Matching start nodes...")

                    # Find start node matching current task ID
                    matching_node = None
                    for idx, node in enumerate(start_nodes):
                        try:
                            if "other_info" in node:
                                other_info = node["other_info"]
                                if isinstance(other_info, str):
                                    other_info = json.loads(other_info)

                                    add_log(
                                        f"Checking node {idx+1}/{len(start_nodes)}: {node['page_id']}"
                                    )

                                    if (
                                        "task_info" in other_info
                                        and "task_id" in other_info["task_info"]
                                    ):
                                        node_task_id = other_info["task_info"][
                                            "task_id"
                                        ]
                                        add_log(f"  - Node task ID: {node_task_id}")

                                        # Check for match, including handling temporary IDs
                                        if node_task_id == current_task_id or (
                                            current_task_id.startswith("temp_")
                                            and node == start_nodes[-1]
                                        ):
                                            matching_node = node
                                            add_log(f"  - ✓ Found matching node!")
                                            break
                                    else:
                                        add_log("  - Node missing task ID information")
                            else:
                                add_log("  - Node missing other_info field")
                        except Exception as e:
                            add_log(f"  - Error parsing node information: {str(e)}")

                    # If no matching node found but using temporary ID, use first node
                    if (
                        not matching_node
                        and current_task_id.startswith("temp_")
                        and start_nodes
                    ):
                        matching_node = start_nodes[0]
                        add_log(
                            f"Using temporary ID, defaulting to first start node: {matching_node['page_id']}"
                        )

                    if not matching_node:
                        add_log(
                            f"❌ No start node found matching task ID {current_task_id}"
                        )
                        return "\n".join(current_log)

                    add_log(f"✓ Using start node: {matching_node['page_id']}")
                    progress(0.4, "Processing operation chain...")

                    # Use async function to process chain
                    async def process_chain():
                        nonlocal current_log
                        add_log("Starting to process operation chain...")
                        progress(0.5, "Analyzing operation chain...")

                        try:
                            add_log(
                                "Calling AI for chain analysis, this may take some time..."
                            )
                            processed_triplets = await process_and_update_chain(
                                matching_node["page_id"]
                            )
                            progress(0.8, "Analysis complete, processing results...")

                            if not processed_triplets:
                                add_log("❌ No processable triplets found")
                                return

                            add_log(
                                f"✓ Successfully processed {len(processed_triplets)} triplets"
                            )

                            # Show example of first triplet processing result
                            if processed_triplets:
                                first_triplet = processed_triplets[0]
                                add_log("\nExample triplet processing result:")

                                # Source page information
                                add_log(
                                    f"Source page ID: {first_triplet['source_page']['page_id']}"
                                )
                                description = first_triplet["source_page"].get(
                                    "description", "N/A"
                                )
                                add_log(
                                    f"Page description: {description[:150]}..."
                                    if len(description) > 150
                                    else f"Page description: {description}"
                                )

                                # Element information
                                add_log(
                                    f"Element ID: {first_triplet['element']['element_id']}"
                                )
                                ele_desc = first_triplet["element"].get(
                                    "description", "N/A"
                                )
                                add_log(
                                    f"Element description: {ele_desc[:150]}..."
                                    if len(ele_desc) > 150
                                    else f"Element description: {ele_desc}"
                                )

                                # Target page information
                                add_log(
                                    f"Target page ID: {first_triplet['target_page']['page_id']}"
                                )
                                target_desc = first_triplet["target_page"].get(
                                    "description", "N/A"
                                )
                                add_log(
                                    f"Target description: {target_desc[:150]}..."
                                    if len(target_desc) > 150
                                    else f"Target description: {target_desc}"
                                )

                                # Reasoning result summary
                                if "reasoning" in first_triplet:
                                    reasoning = first_triplet["reasoning"]
                                    add_log("\nReasoning Result Summary:")

                                    if "user_intent" in reasoning:
                                        user_intent = reasoning["user_intent"]
                                        add_log(
                                            f"User Intent: {user_intent[:150]}..."
                                            if len(user_intent) > 150
                                            else f"User Intent: {user_intent}"
                                        )

                                    if "context" in reasoning:
                                        context = reasoning["context"]
                                        add_log(
                                            f"Operation Context: {context[:150]}..."
                                            if len(context) > 150
                                            else f"Operation Context: {context}"
                                        )

                                    if "state_change" in reasoning:
                                        state_change = reasoning["state_change"]
                                        add_log(
                                            f"State Change: {state_change[:150]}..."
                                            if len(state_change) > 150
                                            else f"State Change: {state_change}"
                                        )
                        except Exception as e:
                            add_log(f"Error processing chain: {str(e)}")
                            import traceback

                            add_log(traceback.format_exc())

                        add_log("\n✨ Operation path understanding complete! ✨")
                        progress(1.0, "Complete!")

                    # Start background task
                    import asyncio
                    import time

                    # Create a new event loop
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)

                    # Run background task in background thread
                    bg_thread = threading.Thread(
                        target=lambda: loop.run_until_complete(process_chain())
                    )
                    bg_thread.start()

                    # Wait for task to complete and update UI in real-time
                    start_time = time.time()
                    while bg_thread.is_alive():
                        # Update every 500ms
                        time.sleep(0.5)
                        yield "\n".join(current_log)

                    # Ensure final update
                    yield "\n".join(current_log)

                except Exception as e:
                    add_log(f"❌ Error understanding operation path: {str(e)}")
                    import traceback

                    add_log(traceback.format_exc())
                    yield "\n".join(current_log)

            understand_btn.click(
                fn=understand_chain,
                inputs=[json_files],
                outputs=[evolution_log],
                queue=True,  # Enable queue processing
            )

            # Define function to generate advanced path
            def generate_high_level_action(selected_file, progress=gr.Progress()):
                global current_files, current_task_id

                if not selected_file:
                    return "Error: Please select a file first"

                # Create variable to store current log status
                current_log = ["Starting to generate advanced path..."]

                # Add log function to update UI
                def add_log(message):
                    current_log.append(message)
                    return "\n".join(current_log)

                # Check if task ID exists, if not try to generate from filename
                if not current_task_id:
                    add_log(
                        "Warning: Task ID not found, attempting to generate temporary task ID from filename..."
                    )
                    result = refresh_task_id_from_file(selected_file)
                    add_log(result)

                    if not current_task_id:
                        add_log(
                            "Error: Could not get task ID, cannot continue generating advanced path"
                        )
                        return "\n".join(current_log)

                add_log(f"Current task ID: {current_task_id}")
                progress(0.1, "Initializing database connection...")

                try:
                    # Connect to Neo4j database
                    db = Neo4jDatabase(uri=config.Neo4j_URI, auth=config.Neo4j_AUTH)

                    # Get all start nodes
                    add_log("Getting start nodes...")
                    progress(0.2, "Getting start nodes...")
                    start_nodes = db.get_chain_start_nodes()

                    if not start_nodes:
                        add_log("❌ No start nodes found")
                        return "\n".join(current_log)

                    add_log(f"Found {len(start_nodes)} start nodes")
                    progress(0.3, "Matching start nodes...")

                    # Find start node matching current task ID
                    matching_node = None
                    for idx, node in enumerate(start_nodes):
                        try:
                            if "other_info" in node:
                                other_info = node["other_info"]
                                if isinstance(other_info, str):
                                    other_info = json.loads(other_info)

                                    add_log(
                                        f"Checking node {idx+1}/{len(start_nodes)}: {node['page_id']}"
                                    )

                                    if (
                                        "task_info" in other_info
                                        and "task_id" in other_info["task_info"]
                                    ):
                                        node_task_id = other_info["task_info"][
                                            "task_id"
                                        ]
                                        add_log(f"  - Node task ID: {node_task_id}")

                                        # Check for match, including handling temporary IDs
                                        if node_task_id == current_task_id or (
                                            current_task_id.startswith("temp_")
                                            and node == start_nodes[-1]
                                        ):
                                            matching_node = node
                                            add_log(f"  - ✓ Found matching node!")
                                            break
                                    else:
                                        add_log("  - Node missing task ID information")
                            else:
                                add_log("  - Node missing other_info field")
                        except Exception as e:
                            add_log(f"  - Error parsing node information: {str(e)}")

                    # If no matching node found but using temporary ID, use first node
                    if (
                        not matching_node
                        and current_task_id.startswith("temp_")
                        and start_nodes
                    ):
                        matching_node = start_nodes[0]
                        add_log(
                            f"Using temporary ID, defaulting to first start node: {matching_node['page_id']}"
                        )

                    if not matching_node:
                        add_log(
                            f"❌ No start node found matching task ID {current_task_id}"
                        )
                        return "\n".join(current_log)

                    add_log(f"✓ Using start node: {matching_node['page_id']}")
                    progress(0.4, "Evaluating chain templatability...")

                    # Use async function to process chain evolution
                    async def process_evolution():
                        nonlocal current_log
                        add_log("Starting chain evolution process...")
                        progress(0.5, "Evaluating chain...")

                        try:
                            # Redirect standard output to capture evolve_chain_to_action output
                            import io
                            import sys
                            from contextlib import redirect_stdout

                            # Call evolve_chain_to_action function
                            add_log(
                                "Evaluating chain templatability, this may take some time..."
                            )
                            progress(0.6, "Evaluating chain...")

                            # Use StringIO to capture output
                            f = io.StringIO()
                            with redirect_stdout(f):
                                # Call chain evolution function
                                action_id = await evolve_chain_to_action(
                                    matching_node["page_id"]
                                )

                            # Get captured output and add to log
                            output = f.getvalue()
                            for line in output.splitlines():
                                if line.strip():  # Ignore empty lines
                                    add_log(line)

                            progress(
                                0.9, "Chain evolution completed, processing results..."
                            )

                            # Process return result
                            if action_id:
                                add_log(
                                    f"✓ Successfully created advanced action node, ID: {action_id}"
                                )
                                # Query database for more advanced action node details
                                add_log("Getting advanced action node details...")

                                # Here, you can add code to query advanced action node details
                                # For example: action_node = db.get_action_node(action_id)

                                add_log(
                                    "Advanced path generation completed successfully!"
                                )
                            else:
                                add_log(
                                    "❌ Chain evolution failed, unable to create advanced action node"
                                )
                                add_log(
                                    "Possible reasons: Chain evaluation as not templatable, or error occurred during generation"
                                )

                        except Exception as e:
                            add_log(f"Chain evolution error: {str(e)}")
                            import traceback

                            add_log(traceback.format_exc())

                        add_log("\n✨ Chain evolution processing completed! ✨")
                        progress(1.0, "Complete!")

                    # Start background task
                    import asyncio
                    import time

                    # Create a new event loop
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)

                    # Run background task in background thread
                    bg_thread = threading.Thread(
                        target=lambda: loop.run_until_complete(process_evolution())
                    )
                    bg_thread.start()

                    # Wait for task to complete and update UI in real-time
                    start_time = time.time()
                    while bg_thread.is_alive():
                        # Update every 500ms
                        time.sleep(0.5)
                        yield "\n".join(current_log)

                    # Ensure final update
                    yield "\n".join(current_log)

                except Exception as e:
                    add_log(f"❌ Error generating advanced path: {str(e)}")
                    import traceback

                    add_log(traceback.format_exc())
                    yield "\n".join(current_log)

            # Modify generate advanced path button binding
            generate_btn.click(
                fn=generate_high_level_action,
                inputs=[json_files],
                outputs=[evolution_log],
                queue=True,  # Enable queue processing
            )

        # Tab 5: Action Execution
        with gr.Tab("Action Execution"):
            gr.Markdown("### Action Execution Management")
            with gr.Row():
                # Left side: Control panel
                with gr.Column(scale=1):
                    # Device selection and task input
                    device_selector_exec = gr.Radio(
                        label="Select Execution Device",
                        choices=get_adb_devices(),
                        interactive=True,
                    )
                    refresh_device_btn = gr.Button(
                        "Refresh Device List", variant="secondary"
                    )

                    task_input = gr.Textbox(
                        label="Enter Task Description",
                        placeholder="e.g., Open settings and switch to airplane mode",
                        lines=3,
                    )

                    # Execution control
                    with gr.Row():
                        execute_btn = gr.Button(
                            "Execute Task", variant="primary", scale=2
                        )
                        stop_btn = gr.Button(
                            "Stop Execution", variant="stop", scale=1, interactive=False
                        )

                    # Execution result status
                    execution_status = gr.Textbox(
                        label="Execution Status", interactive=False
                    )

                # Right side: Logs and screenshots
                with gr.Column(scale=2):
                    # Log output
                    execution_log = gr.TextArea(
                        label="Execution Log", interactive=False, lines=15
                    )

                    # Screenshot display
                    screenshot_gallery_exec = gr.Gallery(
                        label="Execution Process Screenshots", columns=2, height=400
                    )

            # Import deployment module (renamed to avoid collision with explor_auto.run_task)
            from deployment import run_task as deployment_run_task

            # Refresh device list
            def update_execution_devices():
                devices = get_adb_devices()
                return gr.update(choices=devices)

            refresh_device_btn.click(
                fn=update_execution_devices, outputs=[device_selector_exec]
            )

            # Execute task function
            def execute_deployment_task(
                device, task_description, progress=gr.Progress()
            ):
                if not device or device == "No devices found":
                    yield (
                        "Error: Please select a valid device",
                        "Execution failed: No valid device selected",
                        [],
                    )
                    return

                if not task_description or task_description.strip() == "":
                    yield (
                        "Error: Please enter task description",
                        "Execution failed: Task description is empty",
                        [],
                    )
                    return

                # Create storage for logs and screenshots
                logs = []
                screenshots = []
                update_queue = Queue()
                result_queue = Queue()

                def add_log(message):
                    logs.append(message)

                # Callback function that will be called by deployment module
                def execution_callback(state, node_name=None, info=None):
                    logger.info(f"DEPLOYMENT CALLBACK: node={node_name}")

                    if info and isinstance(info, dict):
                        step_num = info.get("step", state.get("current_step", "?"))

                        # Handle different node types
                        if node_name == "capture_screen":
                            logs.append("")
                            logs.append("=" * 50)
                            logs.append(f"📸 STEP {step_num}: Screen Capture")
                            logs.append("-" * 30)
                            if info.get("message"):
                                logs.append(info["message"])
                            if info.get("elements_count"):
                                logs.append(f"   Detected {info['elements_count']} UI elements")

                            # Add screenshot
                            screenshot_path = info.get("screenshot_path")
                            if screenshot_path and screenshot_path not in screenshots:
                                screenshots.append(screenshot_path)
                                logger.info(f"  Added screenshot: {screenshot_path}")

                        elif node_name == "react_capture":
                            logs.append("")
                            logs.append("=" * 50)
                            logs.append(f"📸 STEP {step_num}: React Mode - Screen Capture")
                            logs.append("-" * 30)
                            if info.get("message"):
                                logs.append(info["message"])

                            screenshot_path = info.get("screenshot_path")
                            if screenshot_path and screenshot_path not in screenshots:
                                screenshots.append(screenshot_path)
                                logger.info(f"  Added screenshot: {screenshot_path}")

                        elif node_name == "react_action":
                            logs.append("")
                            logs.append("=" * 50)
                            logs.append(f"🤖 STEP {step_num}: React Mode - Action")
                            logs.append("-" * 30)

                            action_details = info.get("action_details", "")
                            if action_details:
                                logs.append(f"ACTION: {action_details}")

                            llm_reasoning = info.get("llm_reasoning", "")
                            if llm_reasoning:
                                logs.append("")
                                logs.append("💭 LLM REASONING:")
                                for line in llm_reasoning.split('\n')[:10]:  # Limit to 10 lines
                                    if line.strip():
                                        logs.append(f"   {line.strip()}")

                            screenshot_path = info.get("screenshot_path")
                            if screenshot_path and screenshot_path not in screenshots:
                                screenshots.append(screenshot_path)

                        elif node_name == "execute_action":
                            logs.append("")
                            logs.append("=" * 50)
                            logs.append(f"🚀 STEP {step_num}: Execute Action")
                            logs.append("-" * 30)
                            if info.get("message"):
                                logs.append(info["message"])

                            screenshot_path = info.get("screenshot_path")
                            if screenshot_path and screenshot_path not in screenshots:
                                screenshots.append(screenshot_path)

                        elif node_name == "fallback":
                            logs.append("")
                            logs.append("=" * 50)
                            logs.append(f"⚠️ STEP {step_num}: Fallback Mode")
                            logs.append("-" * 30)
                            if info.get("message"):
                                logs.append(info["message"])

                            screenshot_path = info.get("screenshot_path")
                            if screenshot_path and screenshot_path not in screenshots:
                                screenshots.append(screenshot_path)

                        else:
                            # Generic handler for other node types
                            logs.append(f"Step {step_num}: {node_name}")
                            if info.get("message"):
                                logs.append(f"   {info['message']}")

                    # Signal that an update is available
                    update_queue.put(("update", "\n".join(logs), list(screenshots)))

                try:
                    add_log(f"🚀 Starting task execution: '{task_description}'")
                    add_log(f"📱 Using device: {device}")
                    add_log("")
                    progress(0.1, "Initializing execution environment...")

                    # Execute task in background thread
                    def run_in_background():
                        try:
                            add_log("Initializing deployment workflow...")
                            update_queue.put(("update", "\n".join(logs), list(screenshots)))

                            # Call deployment_run_task with the callback
                            result = deployment_run_task(
                                task_description,
                                device,
                                callback=execution_callback
                            )

                            # Put final result in queue
                            result_queue.put(result)

                        except Exception as e:
                            import traceback
                            error_message = f"Error during execution: {str(e)}\n{traceback.format_exc()}"
                            logs.append(f"❌ {error_message}")
                            result_queue.put({
                                "status": "error",
                                "message": str(e),
                            })

                    # Start background thread
                    thread = threading.Thread(target=run_in_background)
                    thread.start()

                    # Continuously yield updates until task completes
                    import time
                    step_count = 0

                    while thread.is_alive():
                        try:
                            # Check for updates (non-blocking)
                            while not update_queue.empty():
                                update_type, log_text, screenshot_list = update_queue.get_nowait()
                                step_count += 1
                                progress_val = min(0.1 + step_count * 0.05, 0.9)
                                progress(progress_val, "Executing task...")
                                yield log_text, "Executing...", screenshot_list
                        except:
                            pass

                        time.sleep(0.3)
                        # Yield current state even if no updates
                        yield "\n".join(logs), "Executing...", list(screenshots)

                    # Process any remaining updates
                    while not update_queue.empty():
                        try:
                            update_type, log_text, screenshot_list = update_queue.get_nowait()
                            yield log_text, "Executing...", screenshot_list
                        except:
                            break

                    # Get final result
                    try:
                        result = result_queue.get(timeout=5)
                        status = result.get("status", "unknown")
                        message = result.get("message", "")

                        if status == "success" or status == "completed":
                            final_status = "✅ Execution successful: Task completed"
                            add_log("")
                            add_log("=" * 50)
                            add_log("✅ Task execution completed!")
                        elif status == "error":
                            final_status = f"❌ Execution failed: {message}"
                            add_log("")
                            add_log(f"❌ Task execution failed: {message}")
                        else:
                            final_status = f"⚠️ Execution status: {status}"
                            add_log(f"⚠️ Final status: {status}")
                    except:
                        final_status = "❓ Unable to retrieve execution result"
                        add_log("❓ Unable to retrieve final execution result")

                    progress(1.0, "Execution completed")
                    add_log("Task execution process completed")

                    yield "\n".join(logs), final_status, list(screenshots)

                except Exception as e:
                    import traceback
                    error_message = f"Error during execution: {str(e)}\n{traceback.format_exc()}"
                    add_log(error_message)
                    yield "\n".join(logs), f"Execution error: {str(e)}", list(screenshots)

            # Handle task execution button click
            execute_btn.click(
                fn=execute_deployment_task,
                inputs=[device_selector_exec, task_input],
                outputs=[execution_log, execution_status, screenshot_gallery_exec],
                queue=True,
            )

            # Stop execution feature can be implemented in future version
            # Currently only set basic UI structure
            def toggle_execution_buttons(executing=True):
                if executing:
                    return gr.update(interactive=False), gr.update(interactive=True)
                else:
                    return gr.update(interactive=True), gr.update(interactive=False)

            execute_btn.click(
                fn=lambda: toggle_execution_buttons(True),
                outputs=[execute_btn, stop_btn],
            )

            stop_btn.click(
                fn=lambda: toggle_execution_buttons(False),
                outputs=[execute_btn, stop_btn],
            )

            # Initialize with automatic refresh of device list
            demo.load(fn=update_execution_devices, outputs=[device_selector_exec])


if __name__ == "__main__":
    demo.launch()
