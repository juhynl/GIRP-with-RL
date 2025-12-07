from time import sleep
from typing import Any, Dict, Tuple

import torch

# from pynput.keyboard import Controller
from selenium import webdriver
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait


class GIRPEnv:
    def __init__(self):
        # Initialize driver
        self.driver = None

        # # Initialize Keyboard
        # self.keyboard = Controller()

        # Initialize action mask and environment variables
        self._action_mask = torch.zeros(27)
        self.current_action = None
        self._goal_height = None

        # Define keys
        self._key = [
            "A",
            "B",
            "C",
            "D",
            "E",
            "F",
            "G",
            "H",
            "I",
            "J",
            "K",
            "L",
            "M",
            "N",
            "O",
            "P",
            "Q",
            "R",
            "S",
            "T",
            "U",
            "V",
            "W",
            "X",
            "Y",
            "Z",
        ]

    def open(self) -> None:
        """Open the browser and initializes the game environment."""

        # Check if the browser is already open
        if self.driver is not None:
            print("Browser is already open.")
            return

        # Open browser
        options = webdriver.ChromeOptions()
        options.add_argument("--window-size=1000,1000")

        self.driver = webdriver.Chrome(options=options)
        self.driver.get("http://0.0.0.0:8000/")

        # Wait for game to load
        wait = WebDriverWait(self.driver, 10)
        game_obj = wait.until(EC.element_to_be_clickable((By.TAG_NAME, "ruffle-object")))

        # Start GIRP (Initial interaction)
        actions = ActionChains(self.driver)
        actions.click_and_hold(game_obj)
        actions.pause(2)
        actions.release()
        actions.perform()
        sleep(2)

        # Reset action mask and environment variables
        self.reset()

        print("Browser opened and game initialized.")

    def close(self) -> None:
        """Close the browser window and terminates the driver session."""
        if self.driver:
            self.driver.quit()
            self.driver = None
            print("Browser closed.")

    def release_all_keys(self) -> None:
        """Release all keys."""
        for k in self._key:
            ActionChains(self.driver).key_up(k).perform()

    def release_all_keys_except(self, action: int) -> None:
        """Release all keys except for the one corresponding to the specified action index.

        Args:
            action (int): The action index of the key that should NOT be released.
        """
        for i in range(len(self._key)):
            if i != action:
                ActionChains(self.driver).key_up(self._key[i]).perform()

    def _get_state(self) -> Dict[str, Any]:
        """Fetch the raw game state from the browser and updates the action mask based on reachability.

        Returns:
            Dict[str, Any]: The raw state dictionary returned by the game engine's `getStateForAgent()` function.
        """
        # Execute JavaScript to get state
        state = self.driver.execute_script("return document.getElementById('GIRP').getStateForAgent();")

        # Only allow actions on toeholds that currently exist in the environment
        current_toehold_indices = state["environment"]["toehold_indices"]
        mask = torch.zeros(26)
        for idx in current_toehold_indices:
            mask[idx] = 1

        # Filter out toeholds that are too far away
        for th in state["environment"]["toeholds"]:
            letter = th["letter"]
            i = ord(letter) - ord("A")
            if 0 <= i <= 26:
                dx = th["x"]
                dy = th["y"] + 0.8
                R = (dx * dx + dy * dy) ** 0.5

                if R >= 3.5:
                    mask[i] = 0

        # Update the action mask
        self._action_mask[:26] = self._action_mask[:26] * mask

        return state

    def return_states(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Retrieve the current state, separated into environment and player contexts.

        Returns:
            Tuple[Dict[str, Any], Dict[str, Any]]: A tuple containing two dictionaries (environment data, player data).
        """
        state = self._get_state()
        return (state["environment"], state["player"])

    def wait_for_game_reset(self):
        """Waits until the game environment is reset and ready for a new episode."""
        print("Wait for game reset...")
        while True:
            state = self._get_state()
            if state["environment"]["time"] <= 0.1 and state["environment"]["winResult"] == 0:
                break
            ActionChains(self.driver).key_down(Keys.SPACE).perform()
            sleep(0.1)
            ActionChains(self.driver).key_up(Keys.SPACE).perform()

        self.reset()

    def step(self, action: int) -> None:
        """Execute a step in the environment by attempting to grab a target toehold.

        Args:
            action (int): The index of the action to perform (0-25 for 'A'-'Z', 26 for 'idle').
        """
        # If action is 'idle' (26), skip execution
        if action == 26:
            return

        # Attempt to grab the target toehold
        ActionChains(self.driver).key_down(self._key[action]).perform()
        sleep(0.01)
        ActionChains(self.driver).key_down(Keys.SHIFT).perform()

        # Check if the attempt was successful
        is_hold = False
        for _ in range(10):
            state = self._get_state()
            for th in state["environment"]["toeholds"]:
                if self._key[action] == th["letter"] and th["state"] == 2:
                    is_hold = True
                    break
            if is_hold:
                break
            sleep(0.005)

        # Handle the result
        if is_hold:
            if self.current_action is not None:
                ActionChains(self.driver).key_up(self._key[self.current_action]).perform()
            self._action_mask.fill_(1)
            self._action_mask[action] = 0
            self.current_action = action
        else:
            ActionChains(self.driver).key_up(self._key[action]).perform()
        ActionChains(self.driver).key_up(Keys.SHIFT).perform()
        sleep(0.1)

    def convert_action_to_key(self, action: int) -> str:
        """
        Convert a discrete action index into a human-readable key character or status string.

        Args:
            action (int): The discrete action index to convert.
                            (0-25 maps to alphabet keys and 26 maps to 'idle').

        Returns:
            str: The corresponding character string (e.g., 'A') or 'idle'.
        """
        if action < 26:
            return f"{self._key[action]}"
        else:
            return "idle"

    def get_available_keys(self, mask: torch.Tensor) -> list[str]:
        """Decode the binary action mask into a list of human-readable key strings.

        Args:
            mask (torch.Tensor): A binary or boolean tensor indicating valid actions.
                             Non-zero values are considered available.

        Returns:
            list[str]: A list of string representations for valid actions
                        (e.g., ['A', 'K', 'idle']).
        """
        available_actions = torch.nonzero(mask)
        keys = []
        for a in available_actions:
            if a < 26:
                keys.append(self._key[a])
            else:
                keys.append("idle")
        return keys

    def get_action_mask(self) -> torch.Tensor:
        """Retrieve the action mask for the current step.

        Returns:
            torch.Tensor: A tensor of size 27 (A-Z and Idle) representing the validity of each
            action (1 for valid, 0 for invalid).
        """
        return self._action_mask

    def set_goal_height(self, height: float) -> None:
        """Set the target vertical height that the agent aims to reach.

        Args:
            height (float): The target vertical coordinate (y-position) in the game world.
                            Must be between 0 and 2400.
        """
        if height < 0 or height > 2400:
            raise ValueError(f"Height must be between 0 and 2400. Got: {height}")
        self._goal_height = -1.0 * height

    def reset(self) -> None:
        """Reset the internal state variables for a new episode.

        Reset the action mask, releases all keys, and initializes
        specific starting toeholds (indices 1, 11, 12, 17) required for the
        agent's initial state.
        """
        # Reset action mask
        self._action_mask.fill_(0)
        for i in [1, 11, 12, 17]:
            self._action_mask[i] = 1

        # Reset keys
        self.release_all_keys()

        # Reset the internal state variable
        self.current_action = None

        # Apply custom goal height
        if self._goal_height is not None:
            self.driver.execute_script(f"return document.getElementById('GIRP').setGoalUpper({self._goal_height})")
