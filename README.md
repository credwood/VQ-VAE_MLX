# Encodec inspired VQ-VAE


## About
- Implemented using the [MLX framework](https://github.com/ml-explore/mlx)
- A paired down architecture, inspired by [Meta's Encodec model](https://github.com/facebookresearch/encodec)


## Instruction

### Setup
This project requires Python 3.11.9. To get started, once you've cloned this repository, navigate to the root folower, create a virtual environment and install the requirements:

```
CONDA_SUBDIR=osx-arm64 conda env create -f environment.yaml
```

If the command finishes without error, a virtual environment called ```audio_mlx``` will be created. Start the virtual environment by running:

```
conda activate audio_mlx
```

### Training
A dummy dataset consisting of a few audio files is available in the root folder. You can launch a training with:

```
python train.py
```

Training loss will be logged in the ```train_log.log``` file in the root directory. The default settinsg for training are purely to test the model, to modify for other uses please edit  ```config.yaml``` .

## Maintenance and Development
- Developed and maintained By [Charysse Redwood](https://github.com/credwod)
- Contributions and feature requests are welcomed!