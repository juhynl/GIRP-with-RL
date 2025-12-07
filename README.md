# GIRP with RL

A Reinforcement Learning project to master the physics-based climbing game, GIRP.

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
We should modify the ActionScript code of the game GIRP to enable a Reinforcement Learning (RL) agent to retrieve the game state.
For detailed information on modifying the SWF file, please refer to the `docs/swf_modification_guide.md` document.\

You can download the modified GIRP game SWF file from the Google Drive [Here](https://drive.google.com/drive/folders/1IoBpMxM3MVHSrX-_mreoR-Ma5UuBcthn?usp=sharing).\
Download "Game/GIRP_for_RL.swf", then place it in the `RL-GIRP/game/` directory.

## Model Preperation
You can use pre-trained model in `model/*.pth`.

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

You can find two pre-trained models in the `models/` directory:
- `good_model.pth`: Achieved an average score of **4.63**.
- `poor_model.pth`: Achieved an average score of **1.39**