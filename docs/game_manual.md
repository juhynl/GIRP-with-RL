# GIRP Game Manual
The primary goal in GIRP is to reach the top of the cliff.

## Success and Failure Conditions
The moment the player presses a key in an attempt to grab the final "top" toehold , the bird will also set this exact toehold as its immediate target, triggering a final sprint.

There are three terminal "game over" conditions in this environment:
- **Failure (Drowning)** The game ends if the player's head goes underwater.
- **Failure (Lost to Bird)** This occurs if the bird (_bird) reaches the goal at the top of the cliff before the player.
- **Success (Victory)** The player wins by reaching the goal at the top of the cliff before the bird.
