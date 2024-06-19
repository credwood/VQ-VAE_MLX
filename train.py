import argparse
import time
from functools import partial
import logging
from pathlib import Path
import random
import yaml

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten

from audio_utils import save_audio
from dataloader import DBDataLoader
from losses import MultiScaleMelSpectrogramLoss
from model.conv_vq import ConvVQ, ConvDUMMY
from model.conv_net import ConvEncoder, ConvDecoder
from model.quantizer import ResidualVectorQuantizer


def reconstruct(model, batch, out_file, sample_rate):
    # Reconstruct a single sample only
    ind = random.randint(0, batch.shape[0]-1)
    audio = mx.expand_dims(mx.array(batch), axis=0)
    audio_recon = model(audio[ind])
    audio_recon = audio_recon.transpose(0, 2, 1)
    save_audio(audio_recon, out_file, sample_rate=sample_rate)

def main(args):
    # Load config
    with open(args.CONFIG, "r") as config:
        config = yaml.safe_load(config)

    # training log
    logger = logging.getLogger(__name__)
    logging.basicConfig(filename=config["train"]["train_log"], level=logging.INFO)
    
    # set random seeds
    np.random.seed(config["train"]["seed"])
    mx.random.seed(config["train"]["seed"])
    # Load the data
    train_iter = DBDataLoader(root_folder=config["data"]["data_location"], 
                                         batch_size=config["train"]["batch_size"], 
                                         subsets="train", 
                                         chunk_duration=config["data"]["chunk_duration"], 
                                         stem=config["data"]["stem"],
                                         num_processes=config["data"]["num_processes"])
    
    test_iter = DBDataLoader(root_folder=config["data"]["data_location"], 
                                         batch_size=config["train"]["test_batch_size"], 
                                         subsets="test", 
                                         chunk_duration=config["data"]["chunk_duration"], 
                                         stem=config["data"]["stem"],
                                         num_processes=config["data"]["num_processes"])

    save_dir = Path(config["train"]["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)
    sample_rate = train_iter.sample_rate
    
    # Load the model
    quantizer = ResidualVectorQuantizer(**config["model"]["quantizer"])
    encoder = ConvEncoder(**config["model"]["encoder"])
    decoder = ConvDecoder(**config["model"]["decoder"])
    model = ConvVQ(encoder=encoder, decoder=decoder, quantizer=quantizer, sample_rate=sample_rate, **config["model"]["convvq"])
    #model = ConvDUMMY(encoder=encoder, decoder=decoder, sample_rate=sample_rate, **config["model"]["convvq"])
    mx.eval(model.parameters())
    num_params = sum(x.size for _, x in tree_flatten(model.trainable_parameters()))
    print("Number of trainable params: {:0.04f} M".format(num_params / 1e6))

    optimizer = optim.AdamW(learning_rate=float(config["train"]["lr"]), betas=[0.5, 0.9])

    # Batches for reconstruction
    train_batch = next(train_iter)
    test_batch = next(test_iter)
    assert train_batch[0].shape[-1] == config["model"]["convvq"]["channels"], print("channel dim and channel value in config must match: array dim:", train_batch[0].shape[-1], config["model"]["convvq"]["channels"])

    state = [model.state, optimizer.state]

    # Multi-Spectrogram loss class and loss function
    multispec_loss = MultiScaleMelSpectrogramLoss(sample_rate=sample_rate, n_mels=80)
    def loss_fn(model: ConvVQ, x: mx.array):
        y = model(x)
        spect_loss = multispec_loss(x, y)
        #vae_spect_loss = multispec_loss(x, model.decoded_for_vae_loss)
        #qt_loss = model.qt_loss
        #enc_quant_loss = model.encoder_loss
        return spect_loss #+ enc_quant_loss[0] + 0.2*qt_loss
    
    
    @partial(mx.compile, inputs=state, outputs=state)
    def step(X):
        loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
        loss, grads = loss_and_grad_fn(model, X)
        optimizer.update(model, grads)
        return loss

    for num_iters in range(1, config["train"]["num_iters"] + 1):
        # Reset iterators and stats at the beginning of each epoch
        model.train()

        # Train one epoch
        tic = time.perf_counter()
        loss_acc = 0.0
        throughput_acc = 0.0

        # Iterate over training batches
        batch = next(train_iter)
        throughput_tic = time.perf_counter()

        # Forward pass + backward pass + update
        loss = step(batch)

        # Evaluate updated model parameters
        mx.eval(state)

        throughput_toc = time.perf_counter()
        throughput_acc += batch.shape[0] / (throughput_toc - throughput_tic)
        loss_acc += loss.item()
        logger.info(f"qt_loss for iteration {num_iters}: {sum(model.qt_loss)/len(model.qt_loss)}")
        logger.info(f"full loss for iteration {num_iters}: {loss.item()}")
        logger.info(f"vae losses: {num_iters}: {loss-0.2*sum(model.qt_loss)/len(model.qt_loss)}")
        logger.info(f"accumulated loss for iteration {num_iters}: {loss_acc}")

        model.qt_loss = []

        toc = time.perf_counter()
        logger.info(f"iteration time: {toc-tic:8.2f} (s)")

        if num_iters > 0 and (num_iters % 100 == 0):
            print(
                "| ".join(
                    [
                        f"Loss {(loss_acc / num_iters):10.2f}",
                        f"Throughput {(throughput_acc / num_iters):8.2f} im/s",
                        f"Iteration {num_iters:5d}",
                        f"Iteration time: {toc-tic:8.2f} (s)"
                    ]
                ),
                end="\r",
            )

        if not num_iters%config["train"]["eval_iters"]:
            model.eval()
            # Reconstruct a batch of training and test images
            sample_audio = Path(config["train"]["sample_audio"])
            sample_audio.mkdir(parents=True, exist_ok=True)
            reconstruct(model, train_batch, f"{sample_audio}/train_{num_iters:03d}.wav", sample_rate=sample_rate)
            reconstruct(model, test_batch, f"{sample_audio}/test_{num_iters:03d}.wav", sample_rate=sample_rate)
        
        if num_iters == config["train"]["num_iters"] or not num_iters%config["train"]["checkpoint"]:
            model.eval()
            model.save_weights(str(save_dir / f"weights_{num_iters}.npz"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-CONFIG", default="config.yaml")
    args = parser.parse_args()
    main(args)