from typing import Any, Dict, List, Optional, Tuple

import numpy as np

SC_W = 10.0  # Scaling factor for x-axis
SC_H = 1000.0  # Scaling factor for y-axis

TOP_K = 5  # Number of nearest toeholds to track per hand


def onehot(letter: str, dim: int = 26) -> np.ndarray:
    """Convert a single character into a one-hot encoded NumPy array.

    Args:
        letter (str): The character to encode (e.g., 'A')
        dim (int, optional): The dimension of the output vector. Defaults to 26.

    Returns:
        np.ndarray: A 1D array of float32 where the index corresponding to
                    the letter is 1.0, and all others are 0.0.
    """
    v = np.zeros(dim, dtype=np.float32)
    if isinstance(letter, str) and len(letter) == 1:
        idx = ord(letter.upper()) - ord("A")
        if 0 <= idx < dim:
            v[idx] = 1.0
    return v


def closest_toehold(hand: Dict[str, Any], toeholds: List[Dict[str, Any]]) -> Optional[str]:
    """Identifiy the letter of the toehold closest to the specified hand.

    Args:
        hand (Dict[str, Any]): The hand's state.
        toeholds (List[Dict[str, Any]]): Available toeholds

    Returns:
        Optional[str]: The letter (e.g., 'A') of the closest toehold if the
            hand is holding or reaching (state > 0).
    """
    if hand.get("state", 0) <= 0:
        return None

    hx, hy = hand["x"], hand["y"]
    t = min(toeholds, key=lambda th: (th["x"] - hx) ** 2 + (th["y"] - hy) ** 2)
    return t["letter"]


def topk_upper_toeholds(hand: Dict[str, Any], toeholds: List[Dict[str, Any]], k: int = 3) -> List[Tuple[Any]]:
    """Select the k nearest toeholds located above the hand and formats them for the agent.

    Args:
        hand (Dict[str, Any]): The hand's state.
        toeholds (List[Dict[str, Any]]): A list of all available toeholds in the environment.
        k (int, optional): The number of nearest neighbors to select.. Defaults to 3.

    Returns:
        List[Tuple[Any]]:A list of k tuples. Each tuple contains:
            - normalized relative x distance (float)
            - normalized relative y distance (float)
            - one-hot encoded vector of the toehold letter (np.ndarray)
            If fewer than k toeholds exist, the list is padded with zeros.
    """
    hx, hy = hand["x"], hand["y"]

    # Filter toeholds to find those 'above' the hand.
    upper = [t for t in toeholds if t["y"] <= hy]

    # If no toeholds are above, consider all toeholds
    if not upper:
        upper = toeholds

    # Sort upper toeholds by distance to the hand (closest first)
    upper = sorted(upper, key=lambda t: (t["x"] - hx) ** 2 + (t["y"] - hy) ** 2)

    # Format data of the upper toeholds
    out = []
    for i in range(k):
        if i < len(upper):
            t = upper[i]
            dx, dy = t["x"] - hx, t["y"] - hy
            dx /= SC_W
            dy /= SC_H
            oh = onehot(t["letter"])
        else:
            dx = dy = 0.0
            oh = np.zeros(26, dtype=np.float32)
        out.append((dx, dy, oh))
    return out


def make_features(env_state: Dict[str, Any], player_state: Dict[str, Any]) -> np.ndarray:
    """Construct a feature vector from the game states.

    Args:
        env_state (Dict[str, Any]): Environment data.
        player_state (Dict[str, Any]): The player's physical state.

    Returns:
        np.ndarray: A 1-dimensional float32 array representing the state.
    """
    obs = []

    # Bird state
    bird = env_state["bird"]
    obs.append(float(bird["landed"]))
    obs.append(bird["x"] / SC_W)
    obs.append(bird["y"] / SC_H)

    # Time and water level
    obs.append(env_state.get("time", 0.0) / 100.0)
    obs.append(env_state.get("waterLevel", 0.0) / SC_H)

    # Game status flags
    obs.append(float(env_state.get("worldPaused", False)))
    obs.append(env_state.get("winResult", 0.0))

    # Player body parts - chest reference
    c_ref = player_state.get("chest_ref", {})
    obs.extend(
        [
            c_ref.get("x", 0.0) / SC_W,
            c_ref.get("y", 0.0) / SC_H,
            c_ref.get("vx", 0.0) / 10.0,
            c_ref.get("vy", 0.0) / 10.0,
        ]
    )

    # Player body parts - head
    head = player_state.get("head", {})
    obs.extend(
        [head.get("x", 0.0) / SC_W, head.get("y", 0.0) / SC_H, head.get("vx", 0.0) / 10.0, head.get("vy", 0.0) / 10.0]
    )

    # Player body parts - left and right hands
    lh = player_state.get("leftHand", {})
    obs.extend(
        [
            lh.get("x", 0.0) / SC_W,
            lh.get("y", 0.0) / SC_H,
            lh.get("vx", 0.0) / 10.0,
            lh.get("vy", 0.0) / 10.0,
            float(lh.get("state", 0)),
        ]
    )

    rh = player_state.get("rightHand", {})
    obs.extend(
        [
            rh.get("x", 0.0) / SC_W,
            rh.get("y", 0.0) / SC_H,
            rh.get("vx", 0.0) / 10.0,
            rh.get("vy", 0.0) / 10.0,
            float(rh.get("state", 0)),
        ]
    )

    # Player body parts - joints
    js = player_state.get("joints", {})
    obs.extend(
        [js.get("leftShoulder", 0.0), js.get("leftElbow", 0.0), js.get("rightShoulder", 0.0), js.get("rightElbow", 0.0)]
    )

    # Toeholds
    toeholds = env_state.get("toeholds", [])

    obs.extend(onehot(closest_toehold(lh, toeholds)))
    obs.extend(onehot(closest_toehold(rh, toeholds)))

    for dx, dy, oh in topk_upper_toeholds(lh, toeholds, k=TOP_K):
        obs.extend([dx, dy])
        obs.extend(oh)

    for dx, dy, oh in topk_upper_toeholds(rh, toeholds, k=TOP_K):
        obs.extend([dx, dy])
        obs.extend(oh)

    return np.array(obs, dtype=np.float32)
