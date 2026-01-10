import base64
import datetime
import json
import logging
import os
from pathlib import Path
from typing import Dict, Optional, TYPE_CHECKING

import requests
from langchain_core.tools import tool

import config  # Import configuration module

# Setup logging
logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from device.controller import DeviceController

# Module-level controller reference
_controller: Optional["DeviceController"] = None

# Module-level square coordinates mapping (updated each step)
_square_coords: Dict[int, Dict[str, int]] = {}


def set_controller(controller: "DeviceController"):
    """Set the device controller for tools to use."""
    global _controller
    _controller = controller


def get_controller() -> "DeviceController":
    """Get the current device controller."""
    if _controller is None:
        raise RuntimeError("Device controller not initialized. Call set_controller() first.")
    return _controller


def has_controller() -> bool:
    """Check if a controller has been set."""
    return _controller is not None


def set_square_coords(coords: Dict[int, Dict[str, int]]):
    """
    Set the current square-to-coordinates mapping.
    Called before each action agent invocation with the current grid's coordinates.
    """
    global _square_coords
    _square_coords = coords
    logger.info(f"📍 Square coords updated: {len(coords)} squares available")


def get_square_coords() -> Dict[int, Dict[str, int]]:
    """Get the current square coordinates mapping."""
    return _square_coords


def translate_square_to_coords(square: int) -> Optional[tuple]:
    """
    Translate a square number to (x, y) coordinates.

    Args:
        square: The grid square number

    Returns:
        Tuple of (x, y) coordinates for the center of the square, or None if invalid
    """
    if square in _square_coords:
        coord = _square_coords[square]
        return (coord["x"], coord["y"])
    return None


# Re-export device listing from the device module for backwards compatibility
def list_all_devices() -> list:
    """
    List all devices currently connected via ADB (Android Debug Bridge).

    Returns:
        - Success: Returns a list of device IDs.
        - Failure: Returns an empty list.
    """
    from device import list_adb_devices
    return list_adb_devices()


@tool
def get_device_size(device: str = "emulator") -> dict | str:
    """
    Get the screen size (width and height) of a mobile device.

    Note: If a controller is set, uses the controller's screen_size.
    Otherwise falls back to ADB for backwards compatibility.

    Parameters:
        - device (str): Device ID (used only in fallback mode)

    Returns:
        - Success: Dictionary with width and height in pixels.
        - Failure: Error message string.
    """
    if has_controller():
        controller = get_controller()
        width, height = controller.screen_size
        return {"width": width, "height": height}
    else:
        # Fallback to direct ADB for backwards compatibility
        from device.adb_controller import execute_adb
        adb_command = f"adb -s {device} shell wm size"
        result = execute_adb(adb_command)
        if result != "ERROR":
            size_str = result.split(": ")[1]
            width, height = map(int, size_str.split("x"))
            return {"width": width, "height": height}
        return "Failed to get device size. Please check device connection or permissions."


@tool
def take_screenshot(
    device: str = "emulator",
    save_dir: str = "./log/screenshots",
    app_name: str = None,
    step: int = 0,
    task_id: str = None,
) -> str:
    """
    Take a screenshot of the device and save it to a directory organized by task.

    Parameters:
        - device (str): Device ID (used only in fallback mode)
        - save_dir (str): Directory path to save the screenshot (fallback if no task_id)
        - app_name (str): Name of the current application
        - step (int): Step number of the current operation
        - task_id (str): Unique task identifier for folder organization

    Returns:
        - Success: Path string where the screenshot is saved.
        - Failure: Error message string.
    """
    if app_name is None:
        app_name = "unknown_app"

    # Use task_id-based folder structure if available
    if task_id:
        screenshot_dir = f"./log/tasks/{task_id}/screenshots"
    else:
        # Fallback to old app-based structure
        screenshot_dir = os.path.join(save_dir, app_name)

    os.makedirs(screenshot_dir, exist_ok=True)

    # Generate filename
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if step is not None:
        filename = f"{app_name}_step{step}_{timestamp}.png"
    else:
        filename = f"{app_name}_{timestamp}.png"

    screenshot_file = Path(screenshot_dir) / filename

    if has_controller():
        try:
            controller = get_controller()
            result_path = controller.get_screenshot(screenshot_file)
            return str(result_path)
        except Exception as e:
            return f"Screenshot failed, error information: {str(e)}"
    else:
        # Fallback to direct ADB for backwards compatibility
        from device.adb_controller import execute_adb
        from time import sleep

        remote_file = f"/sdcard/{filename}"

        cap_command = f"adb -s {device} shell screencap -p {remote_file}"
        pull_command = f"adb -s {device} pull {remote_file} {screenshot_file}"
        delete_command = f"adb -s {device} shell rm {remote_file}"

        sleep(3)

        try:
            if execute_adb(cap_command) != "ERROR":
                if execute_adb(pull_command) != "ERROR":
                    execute_adb(delete_command)
                    return str(screenshot_file)
        except Exception as e:
            return f"Screenshot failed, error information: {str(e)}"

        return "Screenshot failed. Please check device connection or permissions."


@tool
def screen_element(image_path: str) -> Dict:
    """
    Call the OmniParser page understanding tool interface.

    This is platform-agnostic - just sends image to OmniParser service.

    Parameters:
        - image_path (str): File path of the screenshot (local path).

    Returns:
        - Success: Dictionary with labeled_image_path and parsed_content_json_path.
        - Failure: Dictionary containing error information.
    """
    api_url = f"{config.Omni_URI}/process_image/"

    if not os.path.exists(image_path):
        return {"error": "Screenshot file does not exist. Please check the path."}

    # Dynamically generate save directory based on image_path
    save_dir = os.path.join(os.path.dirname(image_path), "processed_images")
    os.makedirs(save_dir, exist_ok=True)

    try:
        with open(image_path, "rb") as file:
            files = [("file", (os.path.basename(image_path), file, "image/png"))]
            response = requests.post(api_url, files=files)

        if response.status_code != 200:
            return {
                "error": f"Interface call failed, status code: {response.status_code}, information: {response.text}"
            }

        data = response.json()
        if data.get("status") != "success":
            return {
                "error": "Interface returned failed status. Please check interface logic.",
                "details": data,
            }

        parsed_content = data.get("parsed_content", [])
        labeled_image_base64 = data.get("labeled_image", "")
        elapsed_time = data.get("e_time", None)

        # Save the labeled image
        if labeled_image_base64:
            labeled_image_data = base64.b64decode(labeled_image_base64)
            labeled_image_path = os.path.join(
                save_dir, f"labeled_{os.path.basename(image_path)}"
            )
            with open(labeled_image_path, "wb") as labeled_image_file:
                labeled_image_file.write(labeled_image_data)
        else:
            return {
                "error": "Labeled image data missing, unable to save labeled image."
            }

        # Save parsed content to JSON file
        json_file_path = os.path.join(
            save_dir, f"{os.path.splitext(os.path.basename(image_path))[0]}.json"
        )
        with open(json_file_path, "w", encoding="utf-8") as json_file:
            json_file.write(json.dumps(parsed_content, ensure_ascii=False, indent=4))

        return {
            "labeled_image_path": labeled_image_path,
            "parsed_content_json_path": json_file_path,
            "elapsed_time": elapsed_time,
        }

    except Exception as e:
        return {"error": f"An exception occurred during tool execution: {str(e)}"}


@tool
def screen_action(
    device: str = "emulator",
    action: str = "tap",
    square: int = None,
    x: int = None,
    y: int = None,
    input_str: str = None,
    duration: int = 1000,
    direction: str = None,
    dist: str = "medium",
    quick: bool = False,
    start: tuple = None,
    end: tuple = None,
) -> str:
    """
    Perform screen operations on devices, including tap, back, text, enter, swipe, long press, and drag.

    Parameters:
        - device (str): Device ID
        - action (str): Type of screen operation:
            - "tap": Tap at a grid square (requires square number)
            - "back": Back key operation
            - "text": Enter text (requires input_str)
            - "enter": Press Enter/Search key (use after typing to submit search or form)
            - "long_press": Long press at grid square (requires square)
            - "swipe": Swipe in direction from grid square (requires square, direction)
            - "swipe_precise": Precise swipe (requires start, end)
        - square (int): Grid square number to interact with. The system automatically translates this to screen coordinates.

    Returns:
        JSON string with status and operation details.
    """
    # Translate square to coordinates if provided
    if square is not None:
        coords = translate_square_to_coords(square)
        if coords:
            x, y = coords
            logger.info(f"🎯 Translated square {square} to coordinates ({x}, {y})")
        else:
            return json.dumps({
                "status": "error",
                "action": action,
                "device": device,
                "message": f"Invalid square number: {square}. Square not found in current grid.",
            })

    logger.info(f"🎯 screen_action CALLED: action={action}, device={device}, square={square}, x={x}, y={y}, input_str={input_str}, direction={direction}")
    try:
        result_data = {"action": action, "device": device}
        if square is not None:
            result_data["square"] = square
        success = False

        if has_controller():
            controller = get_controller()
            logger.info(f"📱 Using controller: {controller}")

            if action == "back":
                adb_cmd = f"adb -s {controller.device_id} shell input keyevent KEYCODE_BACK"
                result_data["adb_command"] = adb_cmd
                print(f"🔧 [ADB] EXECUTING: {adb_cmd}")
                success = controller.press_back()
                print(f"🔧 [ADB] RESULT: {'SUCCESS' if success else 'FAILED'}")

            elif action == "tap":
                if x is None or y is None:
                    return json.dumps({
                        "status": "error",
                        "action": action,
                        "device": device,
                        "message": "Missing required parameters for click action (x, y)",
                    })
                adb_cmd = f"adb -s {controller.device_id} shell input tap {x} {y}"
                result_data["adb_command"] = adb_cmd
                print(f"🔧 [ADB] EXECUTING: {adb_cmd}")
                logger.info(f"👆 Executing TAP at ({x}, {y}) via controller...")
                success = controller.tap(x, y)
                print(f"🔧 [ADB] RESULT: {'SUCCESS' if success else 'FAILED'}")
                logger.info(f"👆 TAP result: success={success}")
                result_data["clicked_element"] = {"x": x, "y": y}

            elif action == "text":
                if not input_str:
                    return json.dumps({
                        "status": "error",
                        "action": action,
                        "device": device,
                        "message": "Missing required parameter for text action (input_str)",
                    })
                sanitized_text = input_str.replace(" ", "%s").replace("'", "")
                adb_cmd = f"adb -s {controller.device_id} shell input text {sanitized_text}"
                result_data["adb_command"] = adb_cmd
                print(f"🔧 [ADB] EXECUTING: {adb_cmd}")
                success = controller.input_text(input_str)
                print(f"🔧 [ADB] RESULT: {'SUCCESS' if success else 'FAILED'}")
                result_data["input_str"] = input_str

            elif action == "enter":
                # Press Enter key (useful after typing in search fields)
                adb_cmd = f"adb -s {controller.device_id} shell input keyevent KEYCODE_ENTER"
                result_data["adb_command"] = adb_cmd
                print(f"🔧 [ADB] EXECUTING: {adb_cmd}")
                logger.info("⏎ Executing ENTER key press via controller...")
                success = controller.press_enter()
                print(f"🔧 [ADB] RESULT: {'SUCCESS' if success else 'FAILED'}")
                logger.info(f"⏎ ENTER result: success={success}")

            elif action == "long_press":
                if x is None or y is None:
                    return json.dumps({
                        "status": "error",
                        "action": action,
                        "device": device,
                        "message": "Missing required parameters for long_press action (x, y)",
                    })
                adb_cmd = f"adb -s {controller.device_id} shell input swipe {x} {y} {x} {y} {duration}"
                result_data["adb_command"] = adb_cmd
                print(f"🔧 [ADB] EXECUTING: {adb_cmd}")
                success = controller.long_press(x, y, duration)
                print(f"🔧 [ADB] RESULT: {'SUCCESS' if success else 'FAILED'}")
                result_data["long_press"] = {"x": x, "y": y, "duration": duration}

            elif action == "swipe":
                if x is None or y is None or direction is None:
                    return json.dumps({
                        "status": "error",
                        "action": action,
                        "device": device,
                        "message": "Missing required parameters for swipe action (x, y, direction)",
                    })
                # Calculate swipe offsets
                unit_dist = 100
                offset_x, offset_y = 0, 0
                if direction == "up":
                    offset_y = -2 * unit_dist if dist == "medium" else -3 * unit_dist
                elif direction == "down":
                    offset_y = 2 * unit_dist if dist == "medium" else 3 * unit_dist
                elif direction == "left":
                    offset_x = -2 * unit_dist if dist == "medium" else -3 * unit_dist
                elif direction == "right":
                    offset_x = 2 * unit_dist if dist == "medium" else 3 * unit_dist
                else:
                    return json.dumps({
                        "status": "error",
                        "action": action,
                        "device": device,
                        "message": "Invalid direction for swipe",
                    })
                swipe_duration = 100 if quick else 400
                adb_cmd = f"adb -s {controller.device_id} shell input swipe {x} {y} {x + offset_x} {y + offset_y} {swipe_duration}"
                result_data["adb_command"] = adb_cmd
                print(f"🔧 [ADB] EXECUTING: {adb_cmd}")
                success = controller.swipe(x, y, x + offset_x, y + offset_y, swipe_duration)
                print(f"🔧 [ADB] RESULT: {'SUCCESS' if success else 'FAILED'}")
                result_data["swipe"] = {
                    "start": (x, y),
                    "end": (x + offset_x, y + offset_y),
                    "duration": swipe_duration,
                    "direction": direction,
                }

            elif action == "swipe_precise":
                if not start or not end:
                    return json.dumps({
                        "status": "error",
                        "action": action,
                        "device": device,
                        "message": "Missing required parameters for swipe_precise action (start, end)",
                    })
                start_x, start_y = start
                end_x, end_y = end
                success = controller.swipe(start_x, start_y, end_x, end_y, duration)
                result_data["swipe_precise"] = {
                    "start": start,
                    "end": end,
                    "duration": duration,
                }

            else:
                return json.dumps({
                    "status": "error",
                    "action": action,
                    "device": device,
                    "message": "Invalid action",
                })

            result_data["status"] = "success" if success else "error"
            if not success:
                result_data["message"] = "Action execution failed"

        else:
            # Fallback to direct ADB for backwards compatibility
            from device.adb_controller import execute_adb

            adb_command = None

            if action == "back":
                adb_command = f"adb -s {device} shell input keyevent KEYCODE_BACK"

            elif action == "tap":
                if x is None or y is None:
                    return json.dumps({
                        "status": "error",
                        "action": action,
                        "device": device,
                        "message": "Missing required parameters for click action (x, y)",
                    })
                adb_command = f"adb -s {device} shell input tap {x} {y}"
                result_data["clicked_element"] = {"x": x, "y": y}

            elif action == "text":
                if not input_str:
                    return json.dumps({
                        "status": "error",
                        "action": action,
                        "device": device,
                        "message": "Missing required parameter for text action (input_str)",
                    })
                sanitized_input_str = input_str.replace(" ", "%s").replace("'", "")
                adb_command = f"adb -s {device} shell input text {sanitized_input_str}"
                result_data["input_str"] = input_str

            elif action == "long_press":
                if x is None or y is None:
                    return json.dumps({
                        "status": "error",
                        "action": action,
                        "device": device,
                        "message": "Missing required parameters for long_press action (x, y)",
                    })
                adb_command = f"adb -s {device} shell input swipe {x} {y} {x} {y} {duration}"
                result_data["long_press"] = {"x": x, "y": y, "duration": duration}

            elif action == "swipe":
                if x is None or y is None or direction is None:
                    return json.dumps({
                        "status": "error",
                        "action": action,
                        "device": device,
                        "message": "Missing required parameters for swipe action (x, y, direction)",
                    })
                unit_dist = 100
                offset_x, offset_y = 0, 0
                if direction == "up":
                    offset_y = -2 * unit_dist if dist == "medium" else -3 * unit_dist
                elif direction == "down":
                    offset_y = 2 * unit_dist if dist == "medium" else 3 * unit_dist
                elif direction == "left":
                    offset_x = -2 * unit_dist if dist == "medium" else -3 * unit_dist
                elif direction == "right":
                    offset_x = 2 * unit_dist if dist == "medium" else 3 * unit_dist
                else:
                    return json.dumps({
                        "status": "error",
                        "action": action,
                        "device": device,
                        "message": "Invalid direction for swipe",
                    })
                swipe_duration = 100 if quick else 400
                adb_command = f"adb -s {device} shell input swipe {x} {y} {x + offset_x} {y + offset_y} {swipe_duration}"
                result_data["swipe"] = {
                    "start": (x, y),
                    "end": (x + offset_x, y + offset_y),
                    "duration": swipe_duration,
                    "direction": direction,
                }

            elif action == "swipe_precise":
                if not start or not end:
                    return json.dumps({
                        "status": "error",
                        "action": action,
                        "device": device,
                        "message": "Missing required parameters for swipe_precise action (start, end)",
                    })
                start_x, start_y = start
                end_x, end_y = end
                adb_command = f"adb -s {device} shell input swipe {start_x} {start_y} {end_x} {end_y} {duration}"
                result_data["swipe_precise"] = {
                    "start": start,
                    "end": end,
                    "duration": duration,
                }

            else:
                return json.dumps({
                    "status": "error",
                    "action": action,
                    "device": device,
                    "message": "Invalid action",
                })

            # Execute ADB command
            ret = execute_adb(adb_command)
            if ret is not None and "ERROR" not in ret.upper():
                result_data["status"] = "success"
            else:
                result_data["status"] = "error"
                result_data["message"] = f"ADB command execution failed: {ret}"

        return json.dumps(result_data, ensure_ascii=False)

    except Exception as e:
        return json.dumps(
            {"status": "error", "action": action, "device": device, "message": str(e)},
            ensure_ascii=False,
        )


@tool
def request_completion_check(reason: str) -> str:
    """
    Request a task completion check from the judge LLM.

    Call this tool ONLY when you believe the task has been fully completed and the
    expected outcome is visible on the current screen. The judge will take a fresh
    screenshot and verify whether the task is actually complete.

    Parameters:
        - reason (str): Brief explanation of why you believe the task is complete.
                       Describe what you see on screen that indicates success.

    Returns:
        JSON string indicating the completion check has been requested.
        The actual judgment will happen after this tool returns.

    IMPORTANT: Only call this when you can SEE the final result on screen.
    Do not call this immediately after performing an action - wait for the
    next screenshot to verify the action's result first.
    """
    logger.info(f"🔍 COMPLETION CHECK REQUESTED: {reason}")
    print(f"🔍 [COMPLETION CHECK] Agent requests verification: {reason}")

    return json.dumps({
        "tool": "request_completion_check",
        "status": "completion_check_requested",
        "reason": reason,
        "message": "Completion check requested. The judge LLM will verify if the task is complete."
    }, ensure_ascii=False)
