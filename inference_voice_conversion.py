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
import argparse
import json
import numpy as np

from scipy.io.wavfile import write

import torch
from torch.cuda import amp

from radtts import RADTTS
from torch.utils.data import DataLoader
from data import Data, DataCollate
from train import update_params, parse_data_from_batch

from hifigan_models import Generator
from hifigan_env import AttrDict
from hifigan_denoiser import Denoiser
from tqdm import tqdm


def is_feature_invalid(x, max_val):
    return (torch.isnan(x).any().item() or
            x.sum() == 0 or
            (x.max().item() > max_val))


def get_configs(config_path, params):
    # Parse configs.  Globals nicer in this case
    with open(config_path) as f:
        data = f.read()

    config = json.loads(data)
    update_params(config, params)

    data_config = config["data_config"]
    model_config = config["model_config"]

    return model_config, data_config


def load_vocoder(vocoder_path, config_path, to_cuda=True):
    with open(config_path) as f:
        data_vocoder = f.read()
    config_vocoder = json.loads(data_vocoder)
    h = AttrDict(config_vocoder)
    if 'blur' in vocoder_path:
        config_vocoder['gaussian_blur']['p_blurring'] = 0.5
    else:
        if 'gaussian_blur' in config_vocoder:
            config_vocoder['gaussian_blur']['p_blurring'] = 0.0
        else:
            config_vocoder['gaussian_blur'] = {'p_blurring': 0.0}
            h['gaussian_blur'] = {'p_blurring': 0.0}

    state_dict_g = torch.load(vocoder_path, map_location='cpu')['generator']

    # load hifigan
    vocoder = Generator(h)
    vocoder.load_state_dict(state_dict_g)
    denoiser = Denoiser(vocoder)
    if to_cuda:
        vocoder.cuda()
        denoiser.cuda()
    vocoder.eval()
    denoiser.eval()

    return vocoder, denoiser


def infer(radtts_path, radtts_config_path, vocoder_path,
          vocoder_config_path, n_samples, sigma, use_amp, seed, output_dir,
          denoising_strength, params, shuffle, takes, save_mels, no_audio,
          predict_features, sigma_f0=1.0, sigma_energy=0.8,
          save_features=False, plot_features=False, f0_mean=0.0, f0_std=0.0,
          energy_mean=0.0, energy_std=0.0, filter_invalid=False):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    ignore_keys = ['training_files', 'validation_files']
    vocoder, denoiser = load_vocoder(vocoder_path, vocoder_config_path)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    model_config, data_config = get_configs(radtts_config_path, params)

    radtts = RADTTS(**model_config)
    radtts.enable_inverse_cache() # cache inverse matrix for 1x1 invertible convs
    print("Loading checkpoint '{}'" .format(radtts_path))

    checkpoint_dict = torch.load(radtts_path, map_location='cpu')
    state_dict = checkpoint_dict['state_dict']

    radtts.load_state_dict(state_dict)
    radtts.remove_norms()
    radtts.eval()
    radtts.cuda()
    print("Loaded checkpoint '{}'" .format(radtts_path))

    trainset = Data(
        data_config['training_files'],
        **dict((k, v) for k, v in data_config.items() if k not in ignore_keys))

    data_config['aug_probabilities'] = None
    data_config['dur_max'] = 60
    valset = Data(data_config['validation_files'],
                  **dict((k, v) for k, v in data_config.items()
                  if k not in ignore_keys),
                  speaker_ids=trainset.speaker_ids)
    collate_fn = DataCollate()
    dataloader = DataLoader(valset, num_workers=1, shuffle=shuffle,
                            sampler=None, batch_size=1,
                            pin_memory=False, drop_last=False,
                            collate_fn=collate_fn)

    f0_max = trainset.f0_max
    energy_max = 1.0
    for k, batch in enumerate(dataloader):
        (mel, speaker_ids, text, in_lens, out_lens, attn_prior,
            f0, voiced_mask, p_voiced, energy_avg,
            audiopaths) = parse_data_from_batch(batch)
        filename = os.path.splitext(
            os.path.basename(batch['audiopaths'][0]))[0]
        f0_gt, energy_avg_gt = f0.clone(), energy_avg.clone()

        suffix_path = "sid{}_sigma{}".format(speaker_ids.item(), sigma)

        print("sample", k, filename)
        with amp.autocast(use_amp):
            # extract duration from attention using ground truth mel
            outputs = radtts(
                mel, speaker_ids, text, in_lens, out_lens, True,
                attn_prior=attn_prior, f0=f0, energy_avg=energy_avg,
                voiced_mask=voiced_mask, p_voiced=p_voiced)

            dur_target = outputs['attn'][0, 0].sum(0, keepdim=True)

        dur_target = (dur_target + 0.5).floor().int()

        with amp.autocast(use_amp):
            for j in tqdm(range(takes)):
                audio_path = "{}/{}_{}_{}_denoised.wav".format(
                        output_dir, filename, j, suffix_path)

                if os.path.exists(audio_path):
                    print("skipping", audio_path)
                    continue

                if predict_features:
                    f0_is_invalid, energy_is_invalid = True, True
                    while f0_is_invalid or energy_is_invalid:
                        model_output = radtts.infer(
                            speaker_ids, text, sigma, None, sigma_f0,
                            sigma_energy, dur=dur_target)
                        f0 = model_output['f0']
                        energy_avg = model_output['energy_avg']
                        if filter_invalid:
                            f0_is_invalid = is_feature_invalid(f0, f0_max)
                            energy_is_invalid = is_feature_invalid(
                                energy_avg, energy_max)
                        else:
                            f0_is_invalid, energy_is_invalid = False, False
                else:
                    model_output = radtts.infer(
                        speaker_ids, text, sigma, dur=dur_target, f0=f0,
                        energy_avg=energy_avg, voiced_mask=voiced_mask,
                        f0_mean=f0_mean, f0_std=f0_std,
                        energy_mean=energy_mean, energy_std=energy_std)

                mel = model_output['mel']

                if save_mels:
                    np.save("{}/{}_{}_{}_mel".format(
                        output_dir, filename, j, suffix_path),
                        mel.cpu().numpy())

                if not no_audio:
                    audio = vocoder(mel).float()[0]
                    audio_denoised = denoiser(
                        audio, strength=denoising_strength)[0].float()
                    audio = audio[0].cpu().numpy()
                    audio_denoised = audio_denoised[0].cpu().numpy()

                    write("{}/{}_{}_{}.wav".format(
                        output_dir, filename, j, suffix_path),
                        data_config['sampling_rate'], audio_denoised)

                if plot_features:
                    fig, axes = plt.subplots(2, 1, figsize=(8, 3))
                    axes[0].plot(f0_gt[0].cpu(), label='gt')
                    axes[0].plot(f0[0].cpu(), label='pred')
                    axes[1].plot(energy_avg_gt[0].cpu(), label='gt')
                    axes[1].plot(energy_avg[0].cpu(), label='pred')
                    plt.savefig("{}/{}_{}_{}.png".format(
                        output_dir, filename, j, suffix_path))
                    plt.close("all")

                if save_features:
                    mask = f0 < data_config['f0_min']
                    f0[mask] = 0.0
                    np.save("{}/{}_{}_{}_f0".format(
                        output_dir, filename, j, suffix_path),
                        f0.cpu().numpy())

                    np.save("{}/{}_{}_{}_energy".format(
                        output_dir, filename, j, suffix_path),
                        energy_avg.cpu().numpy())

        if k + 1 == n_samples:
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--radtts_path', type=str)
    parser.add_argument('-c', '--radtts_config_path', type=str, help='vocoder JSON file config')
    parser.add_argument('-v', '--vocoder_path', type=str)
    parser.add_argument('-k', '--vocoder_config_path', type=str, help='vocoder JSON file config')
    parser.add_argument('-p', '--params', nargs='+', default=[])
    parser.add_argument('-n', '--n_samples', default=5, type=int)
    parser.add_argument("-s", "--sigma", default=0.8, type=float)
    parser.add_argument("--sigma_f0", default=1.0, type=float)
    parser.add_argument("--sigma_energy", default=1.0, type=float)
    parser.add_argument("--f0_mean", default=0.0, type=float)
    parser.add_argument("--f0_std", default=0.0, type=float)
    parser.add_argument("--energy_mean", default=0.0, type=float)
    parser.add_argument("--energy_std", default=0.0, type=float)
    parser.add_argument("--seed", default=1234, type=int)
    parser.add_argument("--use_amp", action="store_true")
    parser.add_argument("-o", '--output_dir', type=str)
    parser.add_argument("-d", "--denoising_strength", default=0.01, type=float)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--save_mels", action="store_true")
    parser.add_argument("--no_audio", action="store_true")
    parser.add_argument("--predict_features", action="store_true")
    parser.add_argument("--save_features", action="store_true")
    parser.add_argument("--plot_features", action="store_true")
    parser.add_argument("--filter_invalid", action="store_true")
    parser.add_argument('-t', '--takes', default=1, type=int)

    args = parser.parse_args()

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    with torch.no_grad():
        infer(args.radtts_path, args.radtts_config_path, args.vocoder_path,
              args.vocoder_config_path, args.n_samples, args.sigma,
              args.use_amp, args.seed, args.output_dir,
              args.denoising_strength, args.params, args.shuffle, args.takes,
              args.save_mels, args.no_audio, args.predict_features,
              args.sigma_f0, args.sigma_energy, args.save_features,
              args.plot_features, args.f0_mean, args.f0_std, args.energy_mean,
              args.energy_std, args.filter_invalid)
