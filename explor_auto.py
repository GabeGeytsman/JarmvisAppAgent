import logging
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_anthropic import ChatAnthropic
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import create_react_agent
from langgraph.types import RetryPolicy
from pydantic import SecretStr
from data.State import State
from tool.screen_content import *

# Setup logging
logger = logging.getLogger(__name__)


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
    Understand the current page
    """
    screen_img = take_screenshot.invoke(
        {
            "device": state["device"],
            "app_name": state["app_name"],
            "step": state["step"],
            "task_id": state.get("task_id"),
        }
    )
    screen_result = screen_element.invoke(
        {
            "image_path": screen_img,
        }
    )
    state["current_page_screenshot"] = screen_img
    state["current_page_json"] = screen_result["parsed_content_json_path"]

    # Add tool result to state BEFORE calling callback so screenshots are available
    if not isinstance(state["tool_results"], list):
        state["tool_results"] = []

    state["tool_results"].append(
        {"tool_name": "screen_element", "result": screen_result}
    )

    # Call the callback function (if any) - AFTER tool_results are updated
    if state.get("callback"):
        callback_info = {
            "labeled_image_path": screen_result.get("labeled_image_path"),
            "step": state["step"],
            "message": f"📷 Step {state['step']}: Captured and analyzed screenshot",
        }
        state["callback"](state, node_name="page_understand", info=callback_info)

    return state


def perform_action(state: State):
    """
    Perform actions based on the current state
    Specifically, this node does two things:
    1. Use LLM to understand the interface and generate recommended actions (based on the current page screenshot, parsed JSON, and user intent)
    2. Execute the recommended action (by calling the relevant tools through the React agent)

    In this step:
    - Need to get the annotated screenshot and parsed JSON result from state
    - Pass these data along with the user's intent to LLM, let React agent analyze and decide
    - Execute the corresponding tool operation
    - Update state's step count, history information, etc.
    """

    # Create action_agent, used for decision making and executing operations on the page
    action_agent = create_react_agent(model, [screen_action])

    # Get the annotated screenshot path and parsed JSON data from state
    labeled_image_path = state.get("current_page_screenshot")
    json_labeled_path = state.get("current_page_json")
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

    # Read screenshot file and encode to base64
    with open(labeled_image_path, "rb") as f:
        image_content = f.read()
    image_data = base64.b64encode(image_content).decode("utf-8")
    image_media_type = get_image_media_type(labeled_image_path)

    # Read parsed JSON file content
    with open(json_labeled_path, "r", encoding="utf-8") as f:
        page_json = f.read()

    # Build message list to pass to LLM, including user intent, page parsed result, and screenshot information
    messages = [
        SystemMessage(
            content="""You are an AI agent controlling a mobile device. Analyze the current screen and determine the SINGLE next action to take.

IMPORTANT RULES:
1. THINK STEP-BY-STEP: Before acting, explain what you see on screen and why you're taking this action.
2. ONE ACTION AT A TIME: Only perform ONE action per turn.
3. TAP BEFORE TYPING: You MUST tap on a text input field to focus it BEFORE you can type text into it. Never use the "text" action unless you have already tapped on an input field in a PREVIOUS step.
4. VERIFY FOCUS: If you need to enter text, first tap the input field. Only type text AFTER confirming the field is focused.
5. BE PRECISE: Use the element IDs and bounding boxes to calculate exact tap coordinates.

TASK COMPLETION:
- If you believe the task has been FULLY completed (the intended outcome is visible on screen), include "TASK_COMPLETE: YES" at the END of your response.
- Only signal completion when the FINAL desired outcome is achieved, not intermediate steps.
- If you're still working toward the goal, do NOT include any TASK_COMPLETE signal.

FORMAT YOUR RESPONSE:
- First, describe what you observe on the current screen
- Then, explain your reasoning for the next action
- Execute the action using the appropriate tool
- If task is done, end with: TASK_COMPLETE: YES

All tool calls must include device to specify the operation device. Only execute one tool call."""
        ),
        HumanMessage(
            content=f"The current device is: {device}, the screen size of the device is {device_size}."
            f"The current task intent is: {user_intent}"
        ),
        HumanMessage(
            content="Below is the parsed JSON data of the current page (the bbox is relative, please convert it to actual operation position based on screen size): \n"
            + page_json
        ),
        HumanMessage(
            content=[
                {
                    "type": "text",
                    "text": "Below is the base64 data of the annotated page screenshot:",
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

    # Call action_agent for decision and execute operation
    action_result = action_agent.invoke({"messages": state["context"][-4:]})

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

    # Extract LLM reasoning - look for the AI message BEFORE tool execution
    # Message flow: [input msgs...] -> AI (reasoning) -> Tool (result) -> AI (summary)
    llm_reasoning = ""
    ai_messages = [msg for msg in final_messages if msg.type == "ai"]
    if ai_messages:
        # First AI message contains the reasoning before tool call
        first_ai_msg = ai_messages[0]
        if hasattr(first_ai_msg, 'content') and first_ai_msg.content:
            llm_reasoning = extract_text_content(first_ai_msg.content)
        # If there's a second AI message, that's the summary after tool call
        if len(ai_messages) > 1:
            summary_msg = ai_messages[-1]
            if hasattr(summary_msg, 'content') and summary_msg.content:
                llm_reasoning += f"\n\n[After action]: {extract_text_content(summary_msg.content)}"

    # Fallback to last message if no reasoning found
    recommended_action = llm_reasoning if llm_reasoning else extract_text_content(final_messages[-1].content)
    state["recommend_action"] = recommended_action

    # Check if agent signals task completion
    agent_signals_complete = False
    for msg in ai_messages:
        if hasattr(msg, 'content') and msg.content:
            msg_text = extract_text_content(msg.content).upper()
            if "TASK_COMPLETE: YES" in msg_text or "TASK_COMPLETE:YES" in msg_text:
                agent_signals_complete = True
                logger.info("Action agent signaled TASK_COMPLETE: YES")
                break
    state["agent_signals_complete"] = agent_signals_complete

    # Parse all tool_messages to get tool execution results
    tool_messages = [msg for msg in final_messages if msg.type == "tool"]
    tool_output = {}
    for tool_message in tool_messages:
        tool_output.update(json.loads(tool_message.content))

    if tool_output:
        # Ensure tool_results is a list
        if not isinstance(state["tool_results"], list):
            state["tool_results"] = []
        # Add tool name for front-end recognition
        tool_output["tool_name"] = "screen_action"  # Or other corresponding tool name
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
                action_details = f"🖱️ TAP at ({coords.get('x', '?')}, {coords.get('y', '?')})"
            elif action_type == "text":
                text = tool_output.get("input_str", "")
                action_details = f"⌨️ TYPE: \"{text}\""
            elif action_type == "swipe":
                direction = tool_output.get("direction", "?")
                action_details = f"👆 SWIPE {direction}"
            elif action_type == "long_press":
                coords = tool_output.get("long_press", {})
                action_details = f"👆 LONG PRESS at ({coords.get('x', '?')}, {coords.get('y', '?')})"
            elif action_type == "back":
                action_details = "⬅️ BACK button pressed"
            else:
                action_details = f"Action: {action_type}"

        callback_info = {
            "step": state["step"] - 1,  # Show the step we just completed
            "llm_reasoning": recommended_action,
            "action_details": action_details,
            "tool_output": tool_output,
            "labeled_image_path": state.get("current_page_screenshot"),
            "message": f"🤖 Step {state['step'] - 1}: {action_details}",
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
    current_screen_result = screen_element.invoke(
        {
            "image_path": current_screen_img,
        }
    )
    state["current_page_screenshot"] = current_screen_img
    state["current_page_json"] = current_screen_result["parsed_content_json_path"]

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
