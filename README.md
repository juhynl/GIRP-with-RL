# GIRP with RL

A Reinforcement Learning project to master the physics-based climbing game, GIRP created by Bennett Foddy.

This project aims to train an agent to learn climbing mechanisms and climb as high as possible in a complex game environment.

For detailed explanation and experimental results, please refer to our report at `docs/report.pdf`.

## Prerequisits
- Python >= 3.10
- torch
- numpy
- selenium
- tensorboard

## Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install .
bash set_game.sh
```

## SWF File Preparation
We modified the ActionScript code of the game GIRP to enable a Reinforcement Learning (RL) agent to retrieve the game state.
For detailed information on modifying the SWF file, please refer to the `docs/swf_modification_guide.md` document.

## Model Preperation
You can find two pre-trained models in the `models/` directory:
- `good_model.pth`: Achieved an average score of **4.63**.
- `poor_model.pth`: Achieved an average score of **1.39**

## Run
Before running any python scripts, you must start the local game server.
```py
bash run_game.sh
```
Open the other terminal and execute the training or evaluation program.

To train the PPO agent, run `train.py`.
```py
python train.py
```

To evaluate the pre-trained PPO agent, run `eval.py` with the path to your model file.

```py
python eval.py --path models/model_file.pth
```
