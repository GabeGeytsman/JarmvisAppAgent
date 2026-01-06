from typing import Protocol, Tuple, Optional
from pathlib import Path


class DeviceController(Protocol):
    """
    Abstract interface for controlling any device.
    Implement this protocol for each platform (ADB, iPhone rig, desktop, etc.)
    """

    @property
    def screen_size(self) -> Tuple[int, int]:
        """Return (width, height) in pixels."""
        ...

    def get_screenshot(self, save_path: Optional[Path] = None) -> Path:
        """
        Capture screenshot and save to disk.

        Args:
            save_path: Where to save the screenshot. If None, use a default temp location.

        Returns:
            Path to the saved screenshot file.
        """
        ...

    def tap(self, x: int, y: int) -> bool:
        """
        Tap at the given coordinates.

        Args:
            x: X coordinate in pixels
            y: Y coordinate in pixels

        Returns:
            True if successful, False otherwise.
        """
        ...

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
        ...

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
        ...

    def input_text(self, text: str) -> bool:
        """
        Input text (as if typing on keyboard).

        Args:
            text: Text to type

        Returns:
            True if successful.
        """
        ...

    def press_back(self) -> bool:
        """
        Press the back button (or equivalent).

        Returns:
            True if successful.
        """
        ...

    def press_home(self) -> bool:
        """
        Press the home button (or equivalent).

        Returns:
            True if successful.
        """
        ...

    def press_enter(self) -> bool:
        """
        Press enter/return key.

        Returns:
            True if successful.
        """
        ...
