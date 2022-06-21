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
import argparse
import os
import json
import numpy as np
import torch
from torch.cuda import amp
from scipy.io.wavfile import write

from radtts import RADTTS
from data import Data
from common import update_params

from hifigan_models import Generator
from hifigan_env import AttrDict
from hifigan_denoiser import Denoiser


def lines_to_list(filename):
    """
    Takes a text file of filenames and makes a list of filenames
    """
    with open(filename, encoding='utf-8') as f:
        files = f.readlines()

    files = [f.rstrip() for f in files]
    return files


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

def infer(radtts_path, vocoder_path, vocoder_config_path, text_path, speaker,
          speaker_text, speaker_attributes, sigma, sigma_tkndur, sigma_f0,
          sigma_energy, f0_mean, f0_std, energy_mean, energy_std,
          token_dur_scaling, denoising_strength, n_takes, output_dir, use_amp,
          plot, seed):

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    vocoder, denoiser = load_vocoder(vocoder_path, vocoder_config_path)
    radtts = RADTTS(**model_config).cuda()
    radtts.enable_inverse_cache() # cache inverse matrix for 1x1 invertible convs

    checkpoint_dict = torch.load(radtts_path, map_location='cpu')
    state_dict = checkpoint_dict['state_dict']
    radtts.load_state_dict(state_dict, strict=False)
    radtts.eval()
    print("Loaded checkpoint '{}')" .format(radtts_path))

    ignore_keys = ['training_files', 'validation_files']
    trainset = Data(
        data_config['training_files'],
        **dict((k, v) for k, v in data_config.items() if k not in ignore_keys))

    speaker_id = trainset.get_speaker_id(speaker).cuda()
    speaker_id_text, speaker_id_attributes = speaker_id, speaker_id
    if speaker_text is not None:
        speaker_id_text = trainset.get_speaker_id(speaker_text).cuda()
    if speaker_attributes is not None:
        speaker_id_attributes = trainset.get_speaker_id(
            speaker_attributes).cuda()

    text_list = lines_to_list(text_path)

    os.makedirs(output_dir, exist_ok=True)
    for i, text in enumerate(text_list):
        if text.startswith("#"):
            continue
        print("{}/{}: {}".format(i, len(text_list), text))
        text = trainset.get_text(text).cuda()[None]
        for take in range(n_takes):
            with amp.autocast(use_amp):
                with torch.no_grad():
                    outputs = radtts.infer(
                        speaker_id, text, sigma, sigma_tkndur, sigma_f0,
                        sigma_energy, token_dur_scaling, token_duration_max=100,
                        speaker_id_text=speaker_id_text,
                        speaker_id_attributes=speaker_id_attributes,
                        f0_mean=f0_mean, f0_std=f0_std, energy_mean=energy_mean,
                        energy_std=energy_std)

                    mel = outputs['mel']
                    audio = vocoder(mel).float()[0]
                    audio_denoised = denoiser(
                        audio, strength=denoising_strength)[0].float()

                    audio = audio[0].cpu().numpy()
                    audio_denoised = audio_denoised[0].cpu().numpy()
                    audio_denoised = audio_denoised / np.max(np.abs(audio_denoised))

                    suffix_path = "{}_{}_{}_durscaling{}_sigma{}_sigmatext{}_sigmaf0{}_sigmaenergy{}".format(
                    i, take, speaker, token_dur_scaling, sigma, sigma_tkndur, sigma_f0,
                    sigma_energy)

                    write("{}/{}_denoised_{}.wav".format(
                        output_dir, suffix_path, denoising_strength),
                        data_config['sampling_rate'], audio_denoised)

            if plot:
                fig, axes = plt.subplots(2, 1, figsize=(10, 6))
                axes[0].plot(outputs['f0'].cpu().numpy()[0], label='f0')
                axes[1].plot(outputs['energy_avg'].cpu().numpy()[0], label='energy_avg')
                for ax in axes:
                    ax.legend(loc='best')
                plt.tight_layout()
                fig.savefig("{}/{}_features.png".format(output_dir, suffix_path))
                plt.close('all')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, help='JSON file config')
    parser.add_argument('-k', '--config_vocoder', type=str, help='vocoder JSON file config')
    parser.add_argument('-p', '--params', nargs='+', default=[])
    parser.add_argument('-r', '--radtts_path', type=str)
    parser.add_argument('-v', '--vocoder_path', type=str)
    parser.add_argument('-t', '--text_path', type=str)
    parser.add_argument('-s', '--speaker', type=str)
    parser.add_argument('--speaker_text', type=str, default=None)
    parser.add_argument('--speaker_attributes', type=str, default=None)
    parser.add_argument('-d', '--denoising_strength', type=float, default=0.0)
    parser.add_argument('-o', "--output_dir", default="results")
    parser.add_argument("--sigma", default=0.8, type=float, help="sampling sigma for decoder")
    parser.add_argument("--sigma_tkndur", default=0.666, type=float, help="sampling sigma for duration")
    parser.add_argument("--sigma_f0", default=1.0, type=float, help="sampling sigma for f0")
    parser.add_argument("--sigma_energy", default=1.0, type=float, help="sampling sigma for energy avg")
    parser.add_argument("--f0_mean", default=0.0, type=float)
    parser.add_argument("--f0_std", default=0.0, type=float)
    parser.add_argument("--energy_mean", default=0.0, type=float)
    parser.add_argument("--energy_std", default=0.0, type=float)
    parser.add_argument("--token_dur_scaling", default=1.00, type=float)
    parser.add_argument("--n_takes", default=1, type=int)
    parser.add_argument("--use_amp", action="store_true")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--seed", default=1234, type=int)
    args = parser.parse_args()

    # Parse configs.  Globals nicer in this case
    with open(args.config) as f:
        data = f.read()

    global config
    config = json.loads(data)
    update_params(config, args.params)

    data_config = config["data_config"]
    global model_config
    model_config = config["model_config"]

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    infer(args.radtts_path, args.vocoder_path, args.config_vocoder,
          args.text_path, args.speaker, args.speaker_text,
          args.speaker_attributes, args.sigma, args.sigma_tkndur, args.sigma_f0,
          args.sigma_energy, args.f0_mean, args.f0_std, args.energy_mean,
          args.energy_std, args.token_dur_scaling, args.denoising_strength,
          args.n_takes, args.output_dir, args.use_amp, args.plot, args.seed)
