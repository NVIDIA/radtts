# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pylab as plt
import os
import numpy as np
import argparse
import json
import glob


import torch
from torch.cuda import amp

from radtts import RADTTS
from torch.utils.data import DataLoader
from data import Data, DataCollate
from train import update_params, parse_data_from_batch
from common import get_mask_from_lengths

from tqdm import tqdm


CHECKPOINT_DIRS = [
    "/home/dcg-adlr-rafaelvalle-output.cosmos356/radtts++/github/radtts++ljs-decoder-defaults-amp",
    "/home/dcg-adlr-rafaelvalle-output.cosmos356/radtts++/github/radtts++ljs-decoder-defaults-amp-melnz0p1",
    "/home/dcg-adlr-rafaelvalle-output.cosmos356/radtts++/github/radtts++ljs-decoder-defaults-amp-melnz0p05",
]


def save_grid_to_image(grid_img, save_path):
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.imshow(grid_img.permute(1, 2, 0))
    fig.savefig(save_path)
    plt.close("all")


def get_configs(config_path, params):
    # Parse configs.  Globals nicer in this case
    with open(config_path) as f:
        data = f.read()

    config = json.loads(data)
    update_params(config, params)

    data_config = config["data_config"]
    model_config = config["model_config"]

    return model_config, data_config


def infer(vocoder_path, vocoder_config_path, batch_size, n_batches, sigma,
          use_amp, seed, output_dir_base, denoising_strength, params, shuffle,
          takes, save_mels, no_audio, predict_features, sigma_f0=1.0,
          sigma_energy=1.0, save_features=False, plot_residuals=False,
          f0_mean=0.0, f0_std=0.0, energy_mean=0.0, energy_std=0.0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    ignore_keys = ['training_files', 'validation_files']

    for i in tqdm(range(len(CHECKPOINT_DIRS))):
        checkpoint_dir = CHECKPOINT_DIRS[i]
        if checkpoint_dir.startswith('#'):
            continue

        if os.path.isfile(checkpoint_dir):
            checkpoint_path = checkpoint_dir
        else:
            checkpoint_paths = glob.glob(os.path.join(checkpoint_dir, "*model*"))
            if len(checkpoint_paths):
                checkpoint_paths = sorted([
                    int(model_name.split('/')[-1].split('_')[-1])
                    for model_name in checkpoint_paths])
                checkpoint_path = os.path.join(
                    checkpoint_dir, "model_" + str(checkpoint_paths[-1]))
            else:
                print("No model found in {}".format(checkpoint_dir))
                break

        config_path = os.path.join(
            os.path.dirname(checkpoint_path), 'config.json')
        output_dir = os.path.join(
            output_dir_base,
            os.path.basename(os.path.dirname(checkpoint_path)))

        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        model_config, data_config = get_configs(config_path, params)

        radtts = RADTTS(**model_config)
        radtts.enable_inverse_cache() # cache inverse matrix for 1x1 invertible convs
        print("Loading checkpoint '{}'" .format(checkpoint_path))

        checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
        state_dict = checkpoint_dict['state_dict']

        radtts.load_state_dict(state_dict)
        radtts.remove_norms()
        radtts.eval()
        radtts.cuda()
        print("Loaded checkpoint '{}'" .format(checkpoint_path))

        trainset = Data(
            data_config['training_files'],
            **dict((k, v) for k, v in data_config.items()
                   if k not in ignore_keys))

        data_config['aug_probabilities'] = None
        data_config['dur_max'] = 60
        valset = Data(data_config['validation_files'],
                      **dict((k, v) for k, v in data_config.items()
                      if k not in ignore_keys),
                      speaker_ids=trainset.speaker_ids)
        collate_fn = DataCollate()

        for name, dataset in (("training", trainset), ("validation", valset)):
            dataloader = DataLoader(dataset, num_workers=1, shuffle=shuffle,
                                    sampler=None, batch_size=batch_size,
                                    pin_memory=False, drop_last=False,
                                    collate_fn=collate_fn)

            for k, batch in enumerate(dataloader):
                (mel, speaker_ids, text, in_lens, out_lens, attn_prior,
                 f0, voiced_mask, p_voiced, energy_avg,
                 audiopaths) = parse_data_from_batch(batch)

                suffix_path = f"sigma{sigma}"

                with amp.autocast(use_amp):
                    # extract duration from attention using ground truth mel
                    outputs = radtts(
                        mel, speaker_ids, text, in_lens, out_lens, True,
                        attn_prior=attn_prior, f0=f0, energy_avg=energy_avg,
                        voiced_mask=voiced_mask, p_voiced=p_voiced)
                    dur_target = outputs['attn'][:, 0].sum(1)

                with amp.autocast(use_amp):
                    model_output = radtts.infer(
                        speaker_ids, text, sigma, dur=dur_target, f0=f0,
                        energy_avg=energy_avg, voiced_mask=voiced_mask,
                        f0_mean=f0_mean, f0_std=f0_std,
                        energy_mean=energy_mean, energy_std=energy_std)

                    mask = get_mask_from_lengths(out_lens).bool().cuda()
                    mask = mask[:, None]
                    mel_hat = model_output['mel']
                    max_len = min(mel.shape[2], mel_hat.shape[2])
                    mel = mel[..., :max_len] * mask[..., :max_len]
                    mel_hat = mel_hat[..., :max_len] * mask[..., :max_len]
                    mel_residuals = (mel - mel_hat).pow(2)
                    mel_mse = (mel_residuals.sum() / mask.sum()).item()
                    print(f"mse loss: {name} {mel_mse}")

                    if True or plot_residuals:
                        fig, axes = plt.subplots(3, 1, figsize=(16, 8))
                        for l in range(mel.shape[0]):
                            mel_res_l = mel_residuals[l].sum() / mask[l].sum()
                            save_path = "{}/data_{}_{}_{}.png".format(output_dir, name, k, l, suffix_path)
                            axes[0].imshow(mel[l].cpu(),aspect='auto', origin='lower', interpolation='none')
                            axes[1].imshow(mel_hat[l].cpu(), aspect='auto', origin='lower', interpolation='none')
                            axes[2].imshow(mel_residuals[l].cpu(), aspect='auto', origin='lower', interpolation='none')
                            axes[0].set_title(f"mel mse {mel_res_l.item()}")
                            fig.savefig(save_path)
                    plt.close("all")

                if k + 1 == n_batches:
                    break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--vocoder_path', type=str)
    parser.add_argument('-k', '--vocoder_config_path', type=str, help='vocoder JSON file config',
                        default="/home/dcg-adlr-rafaelvalle-source.cosmos597/repos/hifi-gan/config_44khz_new.json")
    parser.add_argument('-p', '--params', nargs='+', default=[])
    parser.add_argument('-b', '--batch_size', default=4, type=int)
    parser.add_argument('-n', '--n_batches', default=2, type=int)
    parser.add_argument("-s", "--sigma", default=1.0, type=float)
    parser.add_argument("--sigma_f0", default=1.0, type=float)
    parser.add_argument("--sigma_energy", default=1.0, type=float)
    parser.add_argument("--f0_mean", default=0.0, type=float)
    parser.add_argument("--f0_std", default=0.0, type=float)
    parser.add_argument("--energy_mean", default=0.0, type=float)
    parser.add_argument("--energy_std", default=0.0, type=float)
    parser.add_argument("--seed", default=1234, type=int)
    parser.add_argument("--use_amp", action="store_true")
    parser.add_argument("-o", '--output_dir_base', type=str)
    parser.add_argument("-d", "--denoising_strength", default=0.01, type=float)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--save_mels", action="store_true")
    parser.add_argument("--no_audio", action="store_true")
    parser.add_argument("--predict_features", action="store_true")
    parser.add_argument("--save_features", action="store_true")
    parser.add_argument("--plot_residuals", action="store_true")
    parser.add_argument('-t', '--takes', default=1, type=int)

    args = parser.parse_args()

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False

    if not os.path.exists(args.output_dir_base):
        os.makedirs(args.output_dir_base)

    with torch.no_grad():
        infer(args.vocoder_path, args.vocoder_config_path, args.batch_size,
              args.n_batches, args.sigma, args.use_amp, args.seed,
              args.output_dir_base, args.denoising_strength, args.params,
              args.shuffle, args.takes, args.save_mels, args.no_audio,
              args.predict_features, args.sigma_f0, args.sigma_energy,
              args.save_features, args.plot_residuals, args.f0_mean,
              args.f0_std, args.energy_mean, args.energy_std)
