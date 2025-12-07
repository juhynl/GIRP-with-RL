from typing import Any, Dict


def compute_reward(
    prev_env_state: Dict[str, Any],
    curr_env_state: Dict[str, Any],
    action: int,
) -> float:
    """Computes the reward based on the state transition after an action.

    Reward Scheme:
    - Win: +10.0
    - Loss (Fall/Timeout): -1.0
    - Upward Climb (y_diff > 0.5): +0.5
    - Downward Climb (y_diff < -0.5): -1.0
    - Miss / Idle / Minor Move: -0.01 (Time penalty)

    Args:
        prev_env_state (Dict[str, Any]): _description_
        curr_env_state (Dict[str, Any]): _description_
        action (int): _description_

    Returns:
        float: _description_
    """
    time_penalty = -0.01

    # Check terminal states (win/loss)
    if curr_env_state["winResult"] == -1:
        return -1.0
    elif curr_env_state["winResult"] == 1:
        return 10.0
    if prev_env_state["time"] != 0 and curr_env_state["time"] == 0:
        return -1.0

    # Check idle action
    if action == 26:
        return time_penalty

    # Identify relevant toeholds
    prev_holded_toehold = None
    prev_target_toehold = None
    curr_target_toehold = None
    for th in prev_env_state["toeholds"]:
        letter = th["letter"]
        i = ord(letter) - ord("A")
        if i == action % 26:
            prev_target_toehold = th
        elif th["state"] == 2:
            prev_holded_toehold = th

    for th in curr_env_state["toeholds"]:
        letter = th["letter"]
        i = ord(letter) - ord("A")
        if i == action % 26:
            curr_target_toehold = th

    # Handle exceptions
    if prev_holded_toehold is None:
        return time_penalty

    if curr_target_toehold is None:
        return time_penalty

    if prev_target_toehold is None or prev_holded_toehold is None:
        return time_penalty

    # Calculate climb reward
    climb_reward = time_penalty
    if curr_target_toehold["state"] == 2:
        y_diff = -1 * (prev_target_toehold["y"] - prev_holded_toehold["y"])
        if y_diff > 0.5:
            climb_reward += 0.5
        elif y_diff < -0.5:
            climb_reward -= 1.0
    else:
        climb_reward = time_penalty

    return climb_reward
