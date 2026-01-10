import base64
import datetime
import json
import logging
import os

import config
from PIL import Image, ImageDraw, ImageFont
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_anthropic import ChatAnthropic
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import create_react_agent
from langgraph.types import RetryPolicy
from pydantic import SecretStr
from data.State import State
from tool.screen_content import (
    screen_action,
    request_completion_check,
    take_screenshot,
    get_device_size,
    set_controller,
    set_square_coords,
)

# Setup logging
logger = logging.getLogger(__name__)


def get_dominant_color(img: Image.Image, threshold: float = 0.15) -> tuple:
    """
    Analyze image to find the dominant (most common) color.

    Args:
        img: PIL Image to analyze
        threshold: Minimum percentage of pixels required to consider a color dominant

    Returns:
        Tuple of (r, g, b) for dominant color, or None if no dominant color found
    """
    # Resize for faster analysis
    small = img.copy()
    small.thumbnail((100, 100))

    # Get all pixels
    pixels = list(small.getdata())
    total_pixels = len(pixels)

    # Quantize colors to reduce unique colors (bin into 32-color buckets)
    def quantize(color):
        if len(color) == 4:  # RGBA
            r, g, b, a = color
        else:  # RGB
            r, g, b = color
        return (r // 32 * 32, g // 32 * 32, b // 32 * 32)

    # Count color frequencies
    color_counts = {}
    for pixel in pixels:
        q = quantize(pixel)
        color_counts[q] = color_counts.get(q, 0) + 1

    # Find most common color
    if not color_counts:
        return None

    most_common = max(color_counts.items(), key=lambda x: x[1])
    most_common_color, count = most_common

    # Check if it exceeds threshold
    if count / total_pixels >= threshold:
        return most_common_color
    return None


def get_complement_color(color: tuple) -> tuple:
    """Get the complement (opposite) color."""
    r, g, b = color
    return (255 - r, 255 - g, 255 - b)


def add_coordinate_grid(image_path: str, save_path: str = None, cols: int = 9, rows: int = 22) -> tuple[str, dict]:
    """
    Add a numbered grid overlay to an image to help Claude identify tap locations.

    Divides the screen into numbered squares. Claude identifies which square contains
    the target element, and we tap the center of that square programmatically.

    Args:
        image_path: Path to the original screenshot
        save_path: Optional path to save the gridded image (defaults to adding _grid suffix)
        cols: Number of columns in the grid (default 9 for ~120px cells on 1080 width)
        rows: Number of rows in the grid (default 22 for ~109px cells on 2400 height)

    Returns:
        Tuple of (path to gridded image, dict mapping square numbers to center coordinates)
    """
    img = Image.open(image_path).convert("RGBA")
    width, height = img.size

    # Create a transparent overlay for the grid
    overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    # Calculate cell dimensions
    cell_width = width / cols
    cell_height = height / rows

    # Determine grid line color based on dominant background color
    dominant = get_dominant_color(img)
    if dominant:
        complement = get_complement_color(dominant)
        line_color = (*complement, 180)  # Use complement with alpha
        logger.info(f"Dominant color: {dominant}, using complement: {complement}")
    else:
        line_color = (0, 255, 0, 180)  # Default to green
        logger.info("No dominant color found, using default green")

    # Try to load a font - smaller size for top-left positioning
    try:
        font_size = min(int(cell_width * 0.22), int(cell_height * 0.22), 18)
        font_size = max(font_size, 12)  # Minimum readable size
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", font_size)
    except:
        font = ImageFont.load_default()

    # Colors for text - always dark background for visibility
    text_bg = (0, 0, 0, 200)  # Dark background
    text_color = (255, 255, 255)  # White text for contrast

    # Build mapping of square numbers to center coordinates
    square_coords = {}

    # Draw grid and number each square
    square_num = 1
    for row in range(rows):
        for col in range(cols):
            # Calculate cell boundaries
            x1 = int(col * cell_width)
            y1 = int(row * cell_height)
            x2 = int((col + 1) * cell_width)
            y2 = int((row + 1) * cell_height)

            # Calculate center of cell (for tapping)
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            square_coords[square_num] = {"x": center_x, "y": center_y}

            # Draw cell border
            draw.rectangle([(x1, y1), (x2, y2)], outline=line_color, width=2)

            # Draw square number in TOP-LEFT corner with dark background
            label = str(square_num)
            bbox = draw.textbbox((0, 0), label, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]

            # Position at top-left with small padding
            padding = 2
            label_x = x1 + padding + 2
            label_y = y1 + padding + 2

            # Draw dark background rectangle for number
            draw.rectangle(
                [(label_x - padding, label_y - padding),
                 (label_x + text_width + padding, label_y + text_height + padding)],
                fill=text_bg
            )
            draw.text((label_x, label_y), label, fill=text_color, font=font)

            square_num += 1

    # Composite the overlay onto the original image
    img = Image.alpha_composite(img, overlay)
    img = img.convert("RGB")

    # Save gridded image
    if save_path is None:
        base, ext = os.path.splitext(image_path)
        save_path = f"{base}_grid{ext}"

    img.save(save_path)
    return save_path, square_coords


def extract_text_content(content):
    """
    Extract text from Claude's message content.
    Claude can return content as either:
    - A string (simple text response)
    - A list of content blocks like [{"type": "text", "text": "..."}]

    Args:
        content: Message content (string or list)

    Returns:
        Extracted text as a string
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        text_parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                text_parts.append(block.get("text", ""))
            elif isinstance(block, str):
                text_parts.append(block)
        return " ".join(text_parts).strip()
    return str(content).strip()

os.environ["LANGCHAIN_TRACING_V2"] = config.LANGCHAIN_TRACING_V2
os.environ["LANGCHAIN_ENDPOINT"] = config.LANGCHAIN_ENDPOINT
os.environ["LANGCHAIN_API_KEY"] = config.LANGCHAIN_API_KEY
os.environ["LANGCHAIN_PROJECT"] = config.LANGCHAIN_PROJECT

model = ChatAnthropic(
    api_key=SecretStr(config.ANTHROPIC_API_KEY),
    model_name=config.LLM_MODEL,
    timeout=config.LLM_REQUEST_TIMEOUT,
    max_retries=config.LLM_MAX_RETRIES,
    max_tokens=config.LLM_MAX_TOKEN,
)


def tsk_setting(state: State):
    # Task-related settings
    message = [
        SystemMessage("Please reply with only the application name"),
        HumanMessage(
            f"The task goal is: {state['tsk']}, please infer the related application name. (The application name should not contain spaces) and reply with only one"
        ),
    ]
    llm_response = model.invoke(message)
    app_name = extract_text_content(llm_response.content)
    state["app_name"] = app_name
    state["context"] = [
        HumanMessage(
            f"The task goal is: {state['tsk']}, the inferred application name is: {app_name}"
        )
    ]

    # Use device_info from state if already set (e.g., from controller initialization)
    # Otherwise fetch it using the tool
    if not state.get("device_info"):
        state["device_info"] = get_device_size.invoke({"device": state["device"]})

    # Prepare additional information to pass to the callback function
    callback_info = {
        "app_name": state["app_name"],
        "device_info": state["device_info"],
        "task": state["tsk"],
    }

    # Call the callback function (if any)
    if state.get("callback"):
        # Pass both the current node name and additional information to the callback function
        state["callback"](state, node_name="tsk_setting", info=callback_info)

    return state


def page_understand(state: State):
    """
    Capture the current screen - Claude will analyze the raw screenshot directly
    (OmniParser removed - Claude's vision handles element detection)
    """
    import time
    step_num = state["step"]

    # Log: Starting screenshot capture
    if state.get("callback"):
        state["callback"](state, node_name="page_understand_start", info={
            "step": step_num,
            "message": f"📸 Step {step_num}: Taking screenshot via ADB...",
        })

    screenshot_start = time.time()
    screen_img = take_screenshot.invoke(
        {
            "device": state["device"],
            "app_name": state["app_name"],
            "step": state["step"],
            "task_id": state.get("task_id"),
        }
    )
    screenshot_elapsed = time.time() - screenshot_start

    state["current_page_screenshot"] = screen_img

    # Add tool result to state BEFORE calling callback so screenshots are available
    if not isinstance(state["tool_results"], list):
        state["tool_results"] = []

    state["tool_results"].append(
        {"tool_name": "screenshot", "result": {"screenshot_path": screen_img}}
    )

    # Call the callback function (if any) - AFTER tool_results are updated
    if state.get("callback"):
        callback_info = {
            "labeled_image_path": screen_img,  # Use raw screenshot
            "step": step_num,
            "message": f"📷 Step {step_num}: Screenshot captured in {screenshot_elapsed:.1f}s",
            "screenshot_time": screenshot_elapsed,
        }
        state["callback"](state, node_name="page_understand", info=callback_info)

    return state


def perform_action(state: State):
    """
    Perform actions based on the current state
    Claude analyzes the raw screenshot directly and decides what action to take.
    No external parsing - Claude's vision capabilities handle element detection.
    """

    # Create action_agent, used for decision making and executing operations on the page
    # Tools available: screen_action (for device interaction) and request_completion_check (to signal task done)
    tools = [screen_action, request_completion_check]
    model_with_tools = model.bind_tools(tools)
    action_agent = create_react_agent(model_with_tools, tools)

    # Get the screenshot path from state
    screenshot_path = state.get("current_page_screenshot")
    user_intent = state.get("tsk", "No specific task")
    device = state.get("device", "Unknown device")
    device_size = state.get("device_info", {})

    # Helper function to detect image media type
    def get_image_media_type(file_path):
        ext = os.path.splitext(file_path)[1].lower()
        if ext == '.png':
            return 'image/png'
        elif ext in ['.jpg', '.jpeg']:
            return 'image/jpeg'
        elif ext == '.gif':
            return 'image/gif'
        elif ext == '.webp':
            return 'image/webp'
        else:
            return 'image/png'  # Default to PNG

    # Create numbered grid overlay - returns path and square-to-coordinates mapping
    gridded_screenshot_path, square_coords = add_coordinate_grid(screenshot_path)
    logger.info(f"Created gridded screenshot with {len(square_coords)} numbered squares: {gridded_screenshot_path}")

    # Store square_coords in state for reference
    state["square_coords"] = square_coords

    # Set square coords in the tool module so screen_action can translate squares to coordinates
    set_square_coords(square_coords)

    # Show the gridded image in UI BEFORE sending to Claude (so user can verify what Claude sees)
    if state.get("callback"):
        state["callback"](state, node_name="gridded_screenshot", info={
            "step": state["step"],
            "labeled_image_path": gridded_screenshot_path,
            "message": f"🔲 Step {state['step']}: Numbered grid overlay ({len(square_coords)} squares) - this is what Claude will see",
        })

    # Read GRIDDED screenshot file and encode to base64
    with open(gridded_screenshot_path, "rb") as f:
        image_content = f.read()
    image_data = base64.b64encode(image_content).decode("utf-8")
    image_media_type = get_image_media_type(gridded_screenshot_path)
    logger.info(f"Sending gridded image to Claude: {len(image_data)} bytes base64")

    # Build message list - Claude analyzes the screenshot with numbered grid
    messages = [
        SystemMessage(
            content=f"""You are an AI agent controlling a mobile device. Analyze the screenshot and perform ONE action.

AVAILABLE TOOLS:
1. screen_action - Interact with the device. For tap/long_press/swipe: provide the "square" parameter with the grid square number. For text: provide "input_str". For back/enter: no extra params needed.
2. request_completion_check - Call this when you believe the task is FULLY complete and you can SEE the result

NUMBERED GRID SYSTEM:
The screenshot is divided into {len(square_coords)} NUMBERED SQUARES. Each square has a small number label in its top-left corner.
To interact with an element, simply identify which numbered square contains your target element and provide that square number to screen_action.
The system will automatically translate the square number to the correct screen coordinates.

HOW TO USE:
1. Find the UI element you want to interact with
2. Identify which NUMBERED SQUARE contains that element (look at the number in the top-left of the square)
3. Call screen_action with action="tap" and square=<number>

RESPONSE FORMAT:
<reasoning>
1. OBSERVATION: Describe what you see on the current screen
2. ANALYSIS: Based on the task, what needs to happen next?
3. TARGET: Which UI element? Which square number contains it?
4. ACTION: What action and square number?
</reasoning>

Then call screen_action with the square number. Example: screen_action(device="...", action="tap", square=75)

RULES:
1. Execute exactly ONE tool call per turn
2. Do not assume actions succeeded - verify in the next screenshot
3. For tap/long_press/swipe: use the "square" parameter, NOT x/y coordinates
4. Call request_completion_check only when you can SEE the task is complete"""
        ),
        HumanMessage(
            content=f"Device: {device}\nScreen size: {device_size}\nTask: {user_intent}"
        ),
        HumanMessage(
            content=[
                {
                    "type": "text",
                    "text": "Current screenshot with NUMBERED GRID. Each square has a number in its top-left corner. Find your target element and identify which square number contains it:",
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{image_media_type};base64,{image_data}"},
                },
            ],
        ),
    ]

    # Add these messages to state's context for maintaining dialog continuity
    state["context"].extend(messages)

    # Log: LLM is analyzing and deciding action
    if state.get("callback"):
        state["callback"](state, node_name="action_agent_thinking", info={
            "step": state["step"],
            "message": f"🤔 Step {state['step']}: Claude is analyzing screen and deciding next action...",
        })

    # Call action_agent for decision and execute operation
    import time
    llm_start = time.time()
    action_result = action_agent.invoke({"messages": state["context"][-4:]})
    llm_elapsed = time.time() - llm_start

    # The last message of final_message as the final decision output
    final_messages = action_result.get("messages", [])
    if final_messages:
        # Add final_message to context for continuity
        state["context"].append(final_messages[-1])
    else:
        # If no return message, it indicates an error
        state["context"].append(
            SystemMessage(content="No action decided due to an error.")
        )
        state["errors"].append(
            {"step": state["step"], "error": "No messages returned by action_agent"}
        )
        return state

    # Extract LLM reasoning - only the PRE-ACTION reasoning (before tool execution)
    # Message flow: [context msgs...] -> AI (reasoning + tool_call) -> Tool (result) -> AI (summary)
    # IMPORTANT: After step 0, the first AI message may be stale from context.
    # The actual reasoning is in the AI message that has BOTH content AND tool_calls.
    llm_reasoning = ""
    ai_messages = [msg for msg in final_messages if msg.type == "ai"]

    logger.info(f"🔍 DEBUG: Found {len(ai_messages)} AI messages in action_result")
    for i, msg in enumerate(ai_messages):
        content_preview = str(msg.content)[:200] if msg.content else "EMPTY"
        has_tool_calls = hasattr(msg, 'tool_calls') and msg.tool_calls
        logger.info(f"🔍 DEBUG: AI msg {i}: content={content_preview}, has_tool_calls={has_tool_calls}")

    # Find the AI message that has BOTH content AND tool_calls - that's the reasoning message
    reasoning_msg = None
    for msg in ai_messages:
        has_content = hasattr(msg, 'content') and msg.content
        has_tool_calls = hasattr(msg, 'tool_calls') and msg.tool_calls
        if has_content and has_tool_calls:
            reasoning_msg = msg
            break

    if reasoning_msg:
        llm_reasoning = extract_text_content(reasoning_msg.content)
        logger.info(f"🔍 DEBUG: Found reasoning message with tool_calls: {llm_reasoning[:100]}...")
    elif ai_messages:
        # Fallback: try first AI message with non-empty content
        for msg in ai_messages:
            if hasattr(msg, 'content') and msg.content:
                content = extract_text_content(msg.content)
                if content and '<reasoning>' in content.lower():
                    llm_reasoning = content
                    break

        # If still no reasoning, try to build from tool_calls
        if not llm_reasoning:
            for msg in ai_messages:
                if hasattr(msg, 'tool_calls') and msg.tool_calls:
                    tool_call = msg.tool_calls[0]
                    tool_name = tool_call.get('name', 'unknown')
                    tool_args = tool_call.get('args', {})
                    if tool_name == 'screen_action':
                        action = tool_args.get('action', 'unknown')
                        x = tool_args.get('x', '?')
                        y = tool_args.get('y', '?')
                        llm_reasoning = f"[Tool call: {action} at ({x}, {y})]"
                    else:
                        llm_reasoning = f"[Tool call: {tool_name}]"
                    logger.info(f"🔍 DEBUG: Extracted reasoning from tool_calls: {llm_reasoning}")
                    break

    # Fallback to last message if no reasoning found
    recommended_action = llm_reasoning if llm_reasoning else extract_text_content(final_messages[-1].content)
    logger.info(f"🔍 DEBUG: Final reasoning: {recommended_action[:200] if recommended_action else 'EMPTY'}")
    state["recommend_action"] = recommended_action

    # Parse all tool_messages to get tool execution results
    tool_messages = [msg for msg in final_messages if msg.type == "tool"]
    logger.info(f"🔍 DEBUG: Found {len(tool_messages)} tool messages in action_result")
    tool_output = {}
    agent_signals_complete = False
    completion_check_reason = ""

    for tool_message in tool_messages:
        logger.info(f"🔍 DEBUG: Tool message content: {tool_message.content}")
        try:
            parsed_output = json.loads(tool_message.content)
            # Check if this is the completion check tool
            if parsed_output.get("tool") == "request_completion_check" or parsed_output.get("status") == "completion_check_requested":
                agent_signals_complete = True
                completion_check_reason = parsed_output.get("reason", "")
                logger.info(f"🔍 Agent called request_completion_check tool: {completion_check_reason}")
                print(f"🔍 [COMPLETION CHECK TOOL CALLED] Reason: {completion_check_reason}")
            else:
                # Regular tool output (screen_action)
                tool_output.update(parsed_output)
        except (json.JSONDecodeError, TypeError) as e:
            logger.warning(f"Failed to parse tool message: {e}")

    state["agent_signals_complete"] = agent_signals_complete
    if completion_check_reason:
        state["completion_check_reason"] = completion_check_reason

    if tool_output:
        # Ensure tool_results is a list
        if not isinstance(state["tool_results"], list):
            state["tool_results"] = []
        # Add tool name for front-end recognition
        tool_output["tool_name"] = tool_output.get("action", "screen_action")
        state["tool_results"].append(tool_output)

    # Add this operation record to history step record for future query
    step_record = {
        "step": state["step"],
        "recommended_action": recommended_action,
        "tool_result": tool_output,
        "source_page": state["current_page_screenshot"],
        "source_json": state["current_page_json"],
        "timestamp": datetime.datetime.now().isoformat(),
    }
    state["history_steps"].append(step_record)

    # Update step counter
    state["step"] += 1

    # Call callback with detailed action info
    if state.get("callback"):
        # Format action details for display
        action_details = ""
        if tool_output:
            action_type = tool_output.get("action", "unknown")
            if action_type == "tap":
                coords = tool_output.get("clicked_element", {})
                square_num = tool_output.get("square", "?")
                action_details = f"🖱️ TAP square {square_num} → ({coords.get('x', '?')}, {coords.get('y', '?')})"
            elif action_type == "text":
                text = tool_output.get("input_str", "")
                action_details = f"⌨️ TYPE: \"{text}\""
            elif action_type == "swipe":
                direction = tool_output.get("direction", "?")
                square_num = tool_output.get("square", "?")
                action_details = f"👆 SWIPE {direction} from square {square_num}"
            elif action_type == "long_press":
                coords = tool_output.get("long_press", {})
                square_num = tool_output.get("square", "?")
                action_details = f"👆 LONG PRESS square {square_num} → ({coords.get('x', '?')}, {coords.get('y', '?')})"
            elif action_type == "back":
                action_details = "⬅️ BACK button pressed"
            elif action_type == "enter":
                action_details = "⏎ ENTER key pressed"
            else:
                action_details = f"Action: {action_type}"

        # Get ADB command from tool output (for debugging)
        adb_command = tool_output.get("adb_command", "") if tool_output else ""
        adb_status = tool_output.get("status", "unknown") if tool_output else "no_tool"

        callback_info = {
            "step": state["step"] - 1,  # Show the step we just completed
            "llm_reasoning": recommended_action,
            "action_details": action_details,
            "tool_output": tool_output,
            "labeled_image_path": gridded_screenshot_path,  # Show gridded image in UI
            "message": f"🤖 Step {state['step'] - 1}: {action_details}",
            "llm_time": llm_elapsed,
            # ADB debugging info - shows the actual command executed
            "adb_command": adb_command,
            "adb_status": adb_status,
        }
        state["callback"](state, node_name="perform_action", info=callback_info)

    return state


def tsk_completed(state: State):
    """
    Check if the task is complete. Only invokes the judge LLM when:
    1. The action agent signals task completion (TASK_COMPLETE: YES), OR
    2. The safety limit is reached (step > 20)

    This optimization avoids wasting API calls by checking every step.
    """

    # Allow at least 2 steps before any completion check
    if state["step"] < 2:
        return state["completed"]

    # Check if agent signaled completion
    agent_signals_complete = state.get("agent_signals_complete", False)

    # Only invoke judge LLM if agent signals completion OR safety limit reached
    if not agent_signals_complete and state["step"] <= 20:
        # Agent hasn't signaled completion and we're under safety limit - continue working
        logger.info(f"Step {state['step']}: Agent has not signaled completion, continuing...")
        return False

    if state["step"] > 20:
        logger.warning(f"Step {state['step']}: Safety limit reached, invoking judge")
    else:
        logger.info(f"Step {state['step']}: Agent signaled TASK_COMPLETE, invoking judge to verify")

    logger.info(f"TASK COMPLETION CHECK at step {state['step']}")

    # Get user task description
    user_task = state.get("tsk", "No task description")

    # FIRST: Take a fresh screenshot BEFORE judgment to see current state
    logger.info("Taking fresh screenshot for judgment...")
    current_screen_img = take_screenshot.invoke(
        {
            "device": state["device"],
            "app_name": state["app_name"],
            "step": state["step"],
            "task_id": state.get("task_id"),
        }
    )
    state["current_page_screenshot"] = current_screen_img

    # First step: let LLM reflect on user task, generate task completion judgment criteria
    reflection_messages = [
        SystemMessage(
            content="You are a task completion analyst. Generate SPECIFIC, VERIFIABLE criteria for determining when a task is fully complete."
        ),
        HumanMessage(
            content=f"The user's task is: {user_task}\n\n"
            f"What is the user's INTENDED OUTCOME? What would success look like?\n"
            f"Generate clear completion criteria - describe what MUST be visible on screen to confirm the task achieved its intended result.\n"
            f"Focus on the FINAL outcome, not intermediate steps."
        ),
    ]

    # Call LLM to generate completion criteria
    reflection_response = model.invoke(reflection_messages)
    completion_criteria = extract_text_content(reflection_response.content)
    logger.info(f"Generated completion criteria: {completion_criteria[:200]}...")

    # Add generated completion criteria to context
    state["context"].append(
        SystemMessage(
            content=f"Generated task completion judgment criteria: {completion_criteria}"
        )
    )

    # Second step: use the CURRENT screenshot plus recent history for judgment
    # Include the fresh screenshot we just took as the MOST RECENT
    recent_images = []

    # Add 2 previous screenshots from history if available
    if len(state["page_history"]) >= 2:
        recent_images = state["page_history"][-2:]
    elif len(state["page_history"]) >= 1:
        recent_images = state["page_history"][-1:]

    # Add the CURRENT fresh screenshot as the final (most recent) image
    recent_images.append(current_screen_img)

    # Helper function to detect image media type
    def get_image_media_type(file_path):
        ext = os.path.splitext(file_path)[1].lower()
        if ext == '.png':
            return 'image/png'
        elif ext in ['.jpg', '.jpeg']:
            return 'image/jpeg'
        elif ext == '.gif':
            return 'image/gif'
        elif ext == '.webp':
            return 'image/webp'
        else:
            return 'image/png'  # Default to PNG

    # Convert screenshots to base64 and package as LLM messages
    image_messages = []
    for idx, img_path in enumerate(recent_images, start=1):
        label = "CURRENT STATE" if idx == len(recent_images) else f"Previous screenshot {idx}"
        if os.path.exists(img_path):
            with open(img_path, "rb") as f:
                img_data = base64.b64encode(f.read()).decode("utf-8")
            img_media_type = get_image_media_type(img_path)
            image_messages.append(
                HumanMessage(
                    content=[
                        {
                            "type": "text",
                            "text": f"Screenshot {idx} ({label}):",
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{img_media_type};base64,{img_data}"},
                        },
                    ]
                )
            )
        else:
            image_messages.append(
                HumanMessage(
                    content=f"Cannot find screenshot path: {img_path}"
                )
            )

    # Build final judgment dialog information - TASK AGNOSTIC
    judgement_messages = [
        SystemMessage(
            content="""You are a strict task completion judge. Evaluate whether the user's intended task has been FULLY accomplished based on what is visible on screen.

KEY PRINCIPLES:
1. Consider the user's INTENT, not just the literal words. What outcome were they trying to achieve?
2. Distinguish between INTERMEDIATE states and FINAL outcomes. A task is only complete when the intended result is achieved.
3. If an action was initiated but not yet executed/confirmed, the task is NOT complete.
4. Partial completion is NOT completion. ALL parts of a multi-step task must be done.
5. Look at the CURRENT STATE screenshot carefully - does it show the expected end result?

You must provide your reasoning FIRST, then give your final answer.
Format your response as:
REASONING: [Your detailed analysis of what you see on the current screen and whether it shows the intended outcome was achieved]
VERDICT: [yes/no]"""
        ),
        HumanMessage(
            content=f"TASK: {user_task}\n\n"
            f"COMPLETION CRITERIA: {completion_criteria}\n\n"
            f"Analyze the screenshots (especially the CURRENT STATE - the last screenshot).\n"
            f"Ask yourself: Does the screen show that the user's intended outcome was achieved? Or is it still in an intermediate state?"
        ),
    ] + image_messages

    # Call LLM for final judgment
    judgement_response = model.invoke(judgement_messages)
    judgement_answer = extract_text_content(judgement_response.content)

    logger.info(f"JUDGE LLM RESPONSE:\n{judgement_answer}")

    # Parse the verdict from the response
    verdict_completed = False
    if "VERDICT:" in judgement_answer.upper():
        verdict_part = judgement_answer.upper().split("VERDICT:")[-1].strip()
        if verdict_part.startswith("YES"):
            verdict_completed = True
    elif "yes" in judgement_answer.lower() and "no" not in judgement_answer.lower():
        # Fallback for responses without proper format
        verdict_completed = True

    # Update state
    state["completed"] = verdict_completed

    # Call callback to log the judgment to UI
    if state.get("callback"):
        state["callback"](state, node_name="task_judgment", info={
            "step": state["step"],
            "message": f"🔍 Task Completion Check",
            "completion_criteria": completion_criteria,
            "judge_reasoning": judgement_answer,
            "verdict": "COMPLETE" if verdict_completed else "NOT COMPLETE",
            "screenshot_path": current_screen_img,
        })

    # Add final judgment to context
    state["context"].append(
        SystemMessage(
            content=f"Judge LLM reasoning and verdict: {judgement_answer}"
        )
    )
    state["context"].append(
        SystemMessage(content=f"Final task completion status: {state['completed']}")
    )

    # Safety limit to prevent infinite loops
    if state["step"] > 20:
        logger.warning("Hit 20 step safety limit - forcing completion")
        state["completed"] = True
        if state.get("callback"):
            state["callback"](state, node_name="task_judgment", info={
                "step": state["step"],
                "message": "⚠️ Hit 20 step safety limit - forcing task end",
                "verdict": "FORCED COMPLETE (step limit)",
            })
        return True

    return state["completed"]


# User interaction interface
def run_task(initial_state: State, progress_callback=None):
    logger.info("="*60)
    logger.info("EXPLOR_AUTO.RUN_TASK called")
    logger.info("="*60)
    logger.info(f"initial_state type: {type(initial_state)}")
    logger.info(f"initial_state.tsk: '{initial_state.get('tsk', 'NOT SET')}'")
    logger.info(f"initial_state.device: '{initial_state.get('device', 'NOT SET')}'")
    logger.info(f"initial_state.controller: {initial_state.get('controller', 'NOT SET')}")
    logger.info(f"progress_callback: {progress_callback}")

    # Build StateGraph
    logger.info("Building StateGraph...")
    graph_builder = StateGraph(State)
    # Define nodes in the graph
    graph_builder.add_node(
        "tsk_setting", tsk_setting, retry=RetryPolicy(max_attempts=5)
    )
    graph_builder.add_node(
        "page_understand", page_understand, retry=RetryPolicy(max_attempts=5)
    )
    graph_builder.add_node("perform_action", perform_action)

    # Define edges in the graph
    graph_builder.add_edge(START, "tsk_setting")
    graph_builder.add_edge("tsk_setting", "page_understand")
    graph_builder.add_conditional_edges(
        "page_understand", tsk_completed, {True: END, False: "perform_action"}
    )
    graph_builder.add_edge("perform_action", "page_understand")

    # Compile graph
    logger.info("Compiling graph...")
    graph = graph_builder.compile()
    logger.info("Graph compiled successfully")

    # Visualize graph
    # graph.get_graph().draw_mermaid_png(output_file_path="graph_vis.png")

    # Put callback into state
    if progress_callback is not None:
        initial_state["callback"] = progress_callback
        logger.info("Callback attached to state")

    logger.info("Invoking graph.invoke(initial_state)...")
    try:
        result = graph.invoke(initial_state)
        logger.info("graph.invoke completed!")
        logger.info(f"result type: {type(result)}")
        if result:
            logger.info(f"result.tsk: '{result.get('tsk', 'NOT SET')}'")
            logger.info(f"result.step: {result.get('step', 'NOT SET')}")
            logger.info(f"result.app_name: '{result.get('app_name', 'NOT SET')}'")
            logger.info(f"result.history_steps count: {len(result.get('history_steps', []))}")
        else:
            logger.error("result is None!")
        return result
    except Exception as e:
        logger.exception(f"graph.invoke EXCEPTION: {e}")
        raise
