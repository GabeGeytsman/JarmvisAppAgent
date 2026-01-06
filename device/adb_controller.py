import subprocess
import datetime
import os
from pathlib import Path
from typing import Tuple, Optional, List
from time import sleep


def execute_adb(adb_command: str) -> str:
    """Execute an ADB command and return the output."""
    result = subprocess.run(
        adb_command,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode == 0:
        return result.stdout.strip()
    print(f"Command execution failed: {adb_command}")
    print(result.stderr)
    return "ERROR"


def list_adb_devices() -> List[str]:
    """
    List all connected ADB devices.

    Returns:
        List of device IDs, or empty list if none found.
    """
    adb_command = "adb devices"
    device_list = []
    result = execute_adb(adb_command)
    if result != "ERROR":
        devices = result.split("\n")[1:]
        for d in devices:
            parts = d.split()
            if parts:
                device_list.append(parts[0])
    return device_list


class ADBController:
    """DeviceController implementation for Android devices via ADB."""

    def __init__(self, device_id: str, screenshot_dir: str = "./log/screenshots"):
        """
        Initialize controller for a specific ADB device.

        Args:
            device_id: ADB device identifier (from 'adb devices')
            screenshot_dir: Directory to save screenshots
        """
        self.device_id = device_id
        self._screen_size: Optional[Tuple[int, int]] = None
        self._screenshot_dir = Path(screenshot_dir)
        self._screenshot_dir.mkdir(parents=True, exist_ok=True)
        self._screenshot_counter = 0

    @property
    def screen_size(self) -> Tuple[int, int]:
        """Get screen size from device."""
        if self._screen_size is None:
            adb_command = f"adb -s {self.device_id} shell wm size"
            result = execute_adb(adb_command)
            if result != "ERROR":
                size_str = result.split(": ")[1]
                width, height = map(int, size_str.split("x"))
                self._screen_size = (width, height)
            else:
                # Default fallback
                self._screen_size = (1080, 1920)
        return self._screen_size

    def get_screenshot(self, save_path: Optional[Path] = None) -> Path:
        """
        Capture screenshot and save to disk.

        Args:
            save_path: Where to save the screenshot. If None, use auto-generated path.

        Returns:
            Path to the saved screenshot file.
        """
        if save_path is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            self._screenshot_counter += 1
            filename = f"screenshot_{timestamp}_{self._screenshot_counter}.png"
            save_path = self._screenshot_dir / filename
        else:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)

        remote_file = f"/sdcard/screenshot_temp.png"

        # Construct ADB commands
        cap_command = f"adb -s {self.device_id} shell screencap -p {remote_file}"
        pull_command = f"adb -s {self.device_id} pull {remote_file} {save_path}"
        delete_command = f"adb -s {self.device_id} shell rm {remote_file}"

        # Small delay for screen to stabilize
        sleep(0.5)

        # Execute screenshot commands
        if execute_adb(cap_command) != "ERROR":
            if execute_adb(pull_command) != "ERROR":
                execute_adb(delete_command)
                return save_path

        raise RuntimeError(f"Screenshot failed for device {self.device_id}")

    def tap(self, x: int, y: int) -> bool:
        """
        Tap at the given coordinates.

        Args:
            x: X coordinate in pixels
            y: Y coordinate in pixels

        Returns:
            True if successful, False otherwise.
        """
        adb_command = f"adb -s {self.device_id} shell input tap {x} {y}"
        result = execute_adb(adb_command)
        return result != "ERROR"

    def long_press(self, x: int, y: int, duration_ms: int = 1000) -> bool:
        """
        Long press at coordinates.

        Args:
            x: X coordinate
            y: Y coordinate
            duration_ms: Press duration in milliseconds

        Returns:
            True if successful.
        """
        # ADB implements long press as a swipe to the same position
        adb_command = f"adb -s {self.device_id} shell input swipe {x} {y} {x} {y} {duration_ms}"
        result = execute_adb(adb_command)
        return result != "ERROR"

    def swipe(
        self,
        start_x: int,
        start_y: int,
        end_x: int,
        end_y: int,
        duration_ms: int = 300
    ) -> bool:
        """
        Swipe from start to end coordinates.

        Args:
            start_x, start_y: Starting position
            end_x, end_y: Ending position
            duration_ms: Swipe duration

        Returns:
            True if successful.
        """
        adb_command = f"adb -s {self.device_id} shell input swipe {start_x} {start_y} {end_x} {end_y} {duration_ms}"
        result = execute_adb(adb_command)
        return result != "ERROR"

    def swipe_direction(
        self,
        x: int,
        y: int,
        direction: str,
        dist: str = "medium",
        quick: bool = False
    ) -> bool:
        """
        Swipe in a direction from a starting point.

        Args:
            x: Starting X coordinate
            y: Starting Y coordinate
            direction: "up", "down", "left", or "right"
            dist: "medium" or "long"
            quick: If True, use a faster swipe

        Returns:
            True if successful.
        """
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
            return False

        swipe_duration = 100 if quick else 400
        return self.swipe(x, y, x + offset_x, y + offset_y, swipe_duration)

    def input_text(self, text: str) -> bool:
        """
        Input text (as if typing on keyboard).

        Args:
            text: Text to type

        Returns:
            True if successful.
        """
        # Sanitize input for ADB shell
        sanitized_text = text.replace(" ", "%s").replace("'", "")
        adb_command = f"adb -s {self.device_id} shell input text {sanitized_text}"
        result = execute_adb(adb_command)
        return result != "ERROR"

    def press_back(self) -> bool:
        """
        Press the back button.

        Returns:
            True if successful.
        """
        adb_command = f"adb -s {self.device_id} shell input keyevent KEYCODE_BACK"
        result = execute_adb(adb_command)
        return result != "ERROR"

    def press_home(self) -> bool:
        """
        Press the home button.

        Returns:
            True if successful.
        """
        adb_command = f"adb -s {self.device_id} shell input keyevent KEYCODE_HOME"
        result = execute_adb(adb_command)
        return result != "ERROR"

    def press_enter(self) -> bool:
        """
        Press enter/return key.

        Returns:
            True if successful.
        """
        adb_command = f"adb -s {self.device_id} shell input keyevent KEYCODE_ENTER"
        result = execute_adb(adb_command)
        return result != "ERROR"
