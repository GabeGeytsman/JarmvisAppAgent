from pathlib import Path
from typing import Optional, Tuple, List
import shutil


class StubController:
    """
    Stub controller for testing.
    Returns pre-defined screenshots and logs actions.
    """

    def __init__(
        self,
        screenshot_path: Path,
        screen_size: Tuple[int, int] = (1080, 1920)
    ):
        """
        Initialize stub controller.

        Args:
            screenshot_path: Path to the screenshot to return for all get_screenshot calls
            screen_size: Simulated screen size (width, height)
        """
        self._screenshot_path = Path(screenshot_path)
        self._screen_size = screen_size
        self.action_log: List[dict] = []

    @property
    def screen_size(self) -> Tuple[int, int]:
        return self._screen_size

    def get_screenshot(self, save_path: Optional[Path] = None) -> Path:
        """Return the pre-configured screenshot."""
        self.action_log.append({"action": "get_screenshot", "save_path": str(save_path)})

        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(self._screenshot_path, save_path)
            return save_path
        return self._screenshot_path

    def tap(self, x: int, y: int) -> bool:
        """Log tap action."""
        self.action_log.append({"action": "tap", "x": x, "y": y})
        print(f"[STUB] Tap at ({x}, {y})")
        return True

    def long_press(self, x: int, y: int, duration_ms: int = 1000) -> bool:
        """Log long press action."""
        self.action_log.append({
            "action": "long_press",
            "x": x,
            "y": y,
            "duration_ms": duration_ms
        })
        print(f"[STUB] Long press at ({x}, {y}) for {duration_ms}ms")
        return True

    def swipe(
        self,
        start_x: int,
        start_y: int,
        end_x: int,
        end_y: int,
        duration_ms: int = 300
    ) -> bool:
        """Log swipe action."""
        self.action_log.append({
            "action": "swipe",
            "start_x": start_x,
            "start_y": start_y,
            "end_x": end_x,
            "end_y": end_y,
            "duration_ms": duration_ms
        })
        print(f"[STUB] Swipe from ({start_x}, {start_y}) to ({end_x}, {end_y})")
        return True

    def input_text(self, text: str) -> bool:
        """Log text input action."""
        self.action_log.append({"action": "input_text", "text": text})
        print(f"[STUB] Input text: '{text}'")
        return True

    def press_back(self) -> bool:
        """Log back press action."""
        self.action_log.append({"action": "press_back"})
        print("[STUB] Press back")
        return True

    def press_home(self) -> bool:
        """Log home press action."""
        self.action_log.append({"action": "press_home"})
        print("[STUB] Press home")
        return True

    def press_enter(self) -> bool:
        """Log enter press action."""
        self.action_log.append({"action": "press_enter"})
        print("[STUB] Press enter")
        return True

    def clear_log(self):
        """Clear the action log."""
        self.action_log = []

    def get_actions(self) -> List[dict]:
        """Get all logged actions."""
        return self.action_log.copy()
