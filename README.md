# Riichi Mahjong AI

This is an attempt on building Riichi Mahjong AI using Supervised Machine Learning.

## Requirements

* Python 3
* Python packages:
    * Keras
    * Tk
    * NumPy
    * SciKit-learn

## Running the trainer

```shell
python -m tenhou_log_utils.trainer
```

This reads tenhou logs from the directory specified at the top of the file (`train/01` by default).

## Running the bot

```shell
python -m gui_main
```

This runs the bot with the Tk-based GUI.

## Third-party code used

* https://github.com/MahjongRepository/phoenix-logs - for downloading game logs from tenhou.net
* https://github.com/mthrok/tenhou-log-utils (in `tenhou_log_utils`, MIT license) directory - for paring the tenhou.net logs and training the models
* https://github.com/erreurt/MahjongAI (repository root directory) - framework for building the bot, providing the GUI and tenhou.net client
