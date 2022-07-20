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

# Based on https://github.com/NVIDIA/flowtron/blob/master/data.py
# Original license text:
###############################################################################
#
#  Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
###############################################################################

import os
import argparse
import json
import numpy as np
import lmdb
import pickle as pkl
import torch
import torch.utils.data
from scipy.io.wavfile import read
from audio_processing import TacotronSTFT
from tts_text_processing.text_processing import TextProcessing
from scipy.stats import betabinom
from librosa import pyin
from common import update_params
from scipy.ndimage import distance_transform_edt as distance_transform


def beta_binomial_prior_distribution(phoneme_count, mel_count,
                                     scaling_factor=0.05):
    P = phoneme_count
    M = mel_count
    x = np.arange(0, P)
    mel_text_probs = []
    for i in range(1, M+1):
        a, b = scaling_factor*i, scaling_factor*(M+1-i)
        rv = betabinom(P, a, b)
        mel_i_prob = rv.pmf(x)
        mel_text_probs.append(mel_i_prob)
    return torch.tensor(np.array(mel_text_probs))


def load_wav_to_torch(full_path):
    """ Loads wavdata into torch array """
    sampling_rate, data = read(full_path)
    return torch.from_numpy(np.array(data)).float(), sampling_rate


class Data(torch.utils.data.Dataset):
    def __init__(self, datasets, filter_length, hop_length, win_length,
                 sampling_rate, n_mel_channels, mel_fmin, mel_fmax, f0_min,
                 f0_max, max_wav_value, use_f0, use_energy_avg, use_log_f0,
                 use_scaled_energy, symbol_set, cleaner_names, heteronyms_path,
                 phoneme_dict_path, p_phoneme, handle_phoneme='word',
                 handle_phoneme_ambiguous='ignore', speaker_ids=None,
                 include_speakers=None, n_frames=-1,
                 use_attn_prior_masking=True, prepend_space_to_text=True,
                 append_space_to_text=True, add_bos_eos_to_text=False,
                 betabinom_cache_path="", betabinom_scaling_factor=0.05,
                 lmdb_cache_path="", dur_min=None, dur_max=None,
                 combine_speaker_and_emotion=False, **kwargs):

        self.combine_speaker_and_emotion = combine_speaker_and_emotion
        self.max_wav_value = max_wav_value
        self.audio_lmdb_dict = {}  # dictionary of lmdbs for audio data
        self.data = self.load_data(datasets)
        self.distance_tx_unvoiced = False
        if 'distance_tx_unvoiced' in kwargs.keys():
            self.distance_tx_unvoiced = kwargs['distance_tx_unvoiced']
        self.stft = TacotronSTFT(filter_length=filter_length,
                                 hop_length=hop_length,
                                 win_length=win_length,
                                 sampling_rate=sampling_rate,
                                 n_mel_channels=n_mel_channels,
                                 mel_fmin=mel_fmin, mel_fmax=mel_fmax)

        self.do_mel_scaling = kwargs.get('do_mel_scaling', True)
        self.mel_noise_scale = kwargs.get('mel_noise_scale', 0.0)
        self.filter_length = filter_length
        self.hop_length = hop_length
        self.win_length = win_length
        self.mel_fmin = mel_fmin
        self.mel_fmax = mel_fmax
        self.f0_min = f0_min
        self.f0_max = f0_max
        self.use_f0 = use_f0
        self.use_log_f0 = use_log_f0
        self.use_energy_avg = use_energy_avg
        self.use_scaled_energy = use_scaled_energy
        self.sampling_rate = sampling_rate
        self.tp = TextProcessing(
            symbol_set, cleaner_names, heteronyms_path, phoneme_dict_path,
            p_phoneme=p_phoneme, handle_phoneme=handle_phoneme,
            handle_phoneme_ambiguous=handle_phoneme_ambiguous,
            prepend_space_to_text=prepend_space_to_text,
            append_space_to_text=append_space_to_text,
            add_bos_eos_to_text=add_bos_eos_to_text)

        self.dur_min = dur_min
        self.dur_max = dur_max
        if speaker_ids is None or speaker_ids == '':
            self.speaker_ids = self.create_speaker_lookup_table(self.data)
        else:
            self.speaker_ids = speaker_ids

        print("Number of files", len(self.data))
        if include_speakers is not None:
            for (speaker_set, include) in include_speakers:
                self.filter_by_speakers_(speaker_set, include)
            print("Number of files after speaker filtering", len(self.data))

        if dur_min is not None and dur_max is not None:
            self.filter_by_duration_(dur_min, dur_max)
            print("Number of files after duration filtering", len(self.data))

        self.use_attn_prior_masking = bool(use_attn_prior_masking)
        self.prepend_space_to_text = bool(prepend_space_to_text)
        self.append_space_to_text = bool(append_space_to_text)
        self.betabinom_cache_path = betabinom_cache_path
        self.betabinom_scaling_factor = betabinom_scaling_factor
        self.lmdb_cache_path = lmdb_cache_path
        if self.lmdb_cache_path != "":
            self.cache_data_lmdb = lmdb.open(
                self.lmdb_cache_path, readonly=True, max_readers=1024,
                lock=False).begin()

        # make sure caching path exists
        if not os.path.exists(self.betabinom_cache_path):
            os.makedirs(self.betabinom_cache_path)

        print("Dataloader initialized with no augmentations")
        self.speaker_map = None
        if 'speaker_map' in kwargs:
            self.speaker_map = kwargs['speaker_map']

    def load_data(self, datasets, split='|'):
        dataset = []
        for dset_name, dset_dict in datasets.items():
            folder_path = dset_dict['basedir']
            audiodir = dset_dict['audiodir']
            filename = dset_dict['filelist']
            audio_lmdb_key = None
            if 'lmdbpath' in dset_dict.keys() and len(dset_dict['lmdbpath']) > 0:
                self.audio_lmdb_dict[dset_name] = lmdb.open(
                    dset_dict['lmdbpath'], readonly=True, max_readers=256,
                    lock=False).begin()
                audio_lmdb_key = dset_name

            wav_folder_prefix = os.path.join(folder_path, audiodir)
            filelist_path = os.path.join(folder_path, filename)
            with open(filelist_path, encoding='utf-8') as f:
                data = [line.strip().split(split) for line in f]

            for d in data:
                emotion = 'other' if len(d) == 3 else d[3]
                duration = -1 if len(d) == 3 else d[4]
                dataset.append(
                    {'audiopath': os.path.join(wav_folder_prefix, d[0]),
                     'text': d[1],
                     'speaker': d[2] + '-' + emotion if self.combine_speaker_and_emotion else d[2],
                     'emotion': emotion,
                     'duration': float(duration),
                     'lmdb_key': audio_lmdb_key
                     })
        return dataset

    def filter_by_speakers_(self, speakers, include=True):
        print("Include spaker {}: {}".format(speakers, include))
        if include:
            self.data = [x for x in self.data if x['speaker'] in speakers]
        else:
            self.data = [x for x in self.data if x['speaker'] not in speakers]

    def filter_by_duration_(self, dur_min, dur_max):
        self.data = [
            x for x in self.data
            if x['duration'] == -1 or (
                x['duration'] >= dur_min and x['duration'] <= dur_max)]

    def create_speaker_lookup_table(self, data):
        speaker_ids = np.sort(np.unique([x['speaker'] for x in data]))
        d = {speaker_ids[i]: i for i in range(len(speaker_ids))}
        print("Number of speakers:", len(d))
        print("Speaker IDS", d)
        return d

    def f0_normalize(self, x):
        if self.use_log_f0:
            mask = x >= self.f0_min
            x[mask] = torch.log(x[mask])
            x[~mask] = 0.0

        return x

    def f0_denormalize(self, x):
        if self.use_log_f0:
            log_f0_min = np.log(self.f0_min)
            mask = x >= log_f0_min
            x[mask] = torch.exp(x[mask])
            x[~mask] = 0.0
        x[x <= 0.0] = 0.0

        return x

    def energy_avg_normalize(self, x):
        if self.use_scaled_energy:
            x = (x + 20.0) / 20.0
        return x

    def energy_avg_denormalize(self, x):
        if self.use_scaled_energy:
            x = x * 20.0 - 20.0
        return x

    def get_f0_pvoiced(self, audio, sampling_rate=22050, frame_length=1024,
                       hop_length=256, f0_min=100, f0_max=300):

        audio_norm = audio / self.max_wav_value
        f0, voiced_mask, p_voiced = pyin(
            audio_norm, f0_min, f0_max, sampling_rate,
            frame_length=frame_length, win_length=frame_length // 2,
            hop_length=hop_length)
        f0[~voiced_mask] = 0.0
        f0 = torch.FloatTensor(f0)
        p_voiced = torch.FloatTensor(p_voiced)
        voiced_mask = torch.FloatTensor(voiced_mask)
        return f0, voiced_mask, p_voiced

    def get_energy_average(self, mel):
        energy_avg = mel.mean(0)
        energy_avg = self.energy_avg_normalize(energy_avg)
        return energy_avg

    def get_mel(self, audio):
        audio_norm = audio / self.max_wav_value
        audio_norm = audio_norm.unsqueeze(0)
        audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)
        melspec = self.stft.mel_spectrogram(audio_norm)
        melspec = torch.squeeze(melspec, 0)
        if self.do_mel_scaling:
            melspec = (melspec + 5.5) / 2
        if self.mel_noise_scale > 0:
            melspec += torch.randn_like(melspec) * self.mel_noise_scale
        return melspec

    def get_speaker_id(self, speaker):
        if self.speaker_map is not None and speaker in self.speaker_map:
            speaker = self.speaker_map[speaker]

        return torch.LongTensor([self.speaker_ids[speaker]])

    def get_text(self, text):
        text = self.tp.encode_text(text)
        text = torch.LongTensor(text)
        return text

    def get_attention_prior(self, n_tokens, n_frames):
        # cache the entire attn_prior by filename
        if self.use_attn_prior_masking:
            filename = "{}_{}".format(n_tokens, n_frames)
            prior_path = os.path.join(self.betabinom_cache_path, filename)
            prior_path += "_prior.pth"
            if self.lmdb_cache_path != "":
                attn_prior = pkl.loads(
                    self.cache_data_lmdb.get(prior_path.encode('ascii')))
            elif os.path.exists(prior_path):
                attn_prior = torch.load(prior_path)
            else:
                attn_prior = beta_binomial_prior_distribution(
                    n_tokens, n_frames, self.betabinom_scaling_factor)
                torch.save(attn_prior, prior_path)
        else:
            attn_prior = torch.ones(n_frames, n_tokens)  # all ones baseline

        return attn_prior

    def __getitem__(self, index):
        data = self.data[index]
        audiopath, text = data['audiopath'], data['text']
        speaker_id = data['speaker']

        if data['lmdb_key'] is not None:
            data_dict = pkl.loads(
                self.audio_lmdb_dict[data['lmdb_key']].get(
                    audiopath.encode('ascii')))
            audio = data_dict['audio']
            sampling_rate = data_dict['sampling_rate']
        else:
            audio, sampling_rate = load_wav_to_torch(audiopath)

        if sampling_rate != self.sampling_rate:
            raise ValueError("{} SR doesn't match target {} SR".format(
                sampling_rate, self.sampling_rate))

        mel = self.get_mel(audio)
        f0 = None
        p_voiced = None
        voiced_mask = None
        if self.use_f0:
            filename = '_'.join(audiopath.split('/')[-3:])
            f0_path = os.path.join(self.betabinom_cache_path, filename)
            f0_path += "_f0_sr{}_fl{}_hl{}_f0min{}_f0max{}_log{}.pt".format(
                self.sampling_rate, self.filter_length, self.hop_length,
                self.f0_min, self.f0_max, self.use_log_f0)

            dikt = None
            if len(self.lmdb_cache_path) > 0:
                dikt = pkl.loads(
                    self.cache_data_lmdb.get(f0_path.encode('ascii')))
                f0 = dikt['f0']
                p_voiced = dikt['p_voiced']
                voiced_mask = dikt['voiced_mask']
            elif os.path.exists(f0_path):
                try:
                    dikt = torch.load(f0_path)
                except:
                    print(f"f0 loading from {f0_path} is broken, recomputing.")

            if dikt is not None:
                f0 = dikt['f0']
                p_voiced = dikt['p_voiced']
                voiced_mask = dikt['voiced_mask']
            else:
                f0, voiced_mask, p_voiced = self.get_f0_pvoiced(
                    audio.cpu().numpy(), self.sampling_rate,
                    self.filter_length, self.hop_length, self.f0_min,
                    self.f0_max)
                print("saving f0 to {}".format(f0_path))
                torch.save({'f0': f0,
                            'voiced_mask': voiced_mask,
                            'p_voiced': p_voiced}, f0_path)
            if f0 is None:
                raise Exception("STOP, BROKEN F0 {}".format(audiopath))

            f0 = self.f0_normalize(f0)
            if self.distance_tx_unvoiced:
                mask = f0 <= 0.0
                distance_map = np.log(distance_transform(mask))
                distance_map[distance_map <= 0] = 0.0
                f0 = f0 - distance_map

        energy_avg = None
        if self.use_energy_avg:
            energy_avg = self.get_energy_average(mel)
            if self.use_scaled_energy and energy_avg.min() < 0.0:
                print(audiopath, "has scaled energy avg smaller than 0")

        speaker_id = self.get_speaker_id(speaker_id)
        text_encoded = self.get_text(text)

        attn_prior = self.get_attention_prior(
                text_encoded.shape[0], mel.shape[1])

        if not self.use_attn_prior_masking:
            attn_prior = None

        return {'mel': mel,
                'speaker_id': speaker_id,
                'text_encoded': text_encoded,
                'audiopath': audiopath,
                'attn_prior': attn_prior,
                'f0': f0,
                'p_voiced': p_voiced,
                'voiced_mask': voiced_mask,
                'energy_avg': energy_avg,
                }

    def __len__(self):
        return len(self.data)


class DataCollate():
    """ Zero-pads model inputs and targets given number of steps """
    def __init__(self, n_frames_per_step=1):
        self.n_frames_per_step = n_frames_per_step

    def __call__(self, batch):
        """Collate from normalized data """
        # Right zero-pad all one-hot text sequences to max input length
        input_lengths, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([len(x['text_encoded']) for x in batch]),
            dim=0, descending=True)

        max_input_len = input_lengths[0]
        text_padded = torch.LongTensor(len(batch), max_input_len)
        text_padded.zero_()

        for i in range(len(ids_sorted_decreasing)):
            text = batch[ids_sorted_decreasing[i]]['text_encoded']
            text_padded[i, :text.size(0)] = text

        # Right zero-pad mel-spec
        num_mel_channels = batch[0]['mel'].size(0)
        max_target_len = max([x['mel'].size(1) for x in batch])

        # include mel padded, gate padded and speaker ids
        mel_padded = torch.FloatTensor(len(batch), num_mel_channels, max_target_len)
        mel_padded.zero_()
        f0_padded = None
        p_voiced_padded = None
        voiced_mask_padded = None
        energy_avg_padded = None
        if batch[0]['f0'] is not None:
            f0_padded = torch.FloatTensor(len(batch), max_target_len)
            f0_padded.zero_()

        if batch[0]['p_voiced'] is not None:
            p_voiced_padded = torch.FloatTensor(len(batch), max_target_len)
            p_voiced_padded.zero_()

        if batch[0]['voiced_mask'] is not None:
            voiced_mask_padded = torch.FloatTensor(len(batch), max_target_len)
            voiced_mask_padded.zero_()

        if batch[0]['energy_avg'] is not None:
            energy_avg_padded = torch.FloatTensor(len(batch), max_target_len)
            energy_avg_padded.zero_()

        attn_prior_padded = torch.FloatTensor(len(batch), max_target_len, max_input_len)
        attn_prior_padded.zero_()

        output_lengths = torch.LongTensor(len(batch))
        speaker_ids = torch.LongTensor(len(batch))
        audiopaths = []
        for i in range(len(ids_sorted_decreasing)):
            mel = batch[ids_sorted_decreasing[i]]['mel']
            mel_padded[i, :, :mel.size(1)] = mel
            if batch[ids_sorted_decreasing[i]]['f0'] is not None:
                f0 = batch[ids_sorted_decreasing[i]]['f0']
                f0_padded[i, :len(f0)] = f0

            if batch[ids_sorted_decreasing[i]]['voiced_mask'] is not None:
                voiced_mask = batch[ids_sorted_decreasing[i]]['voiced_mask']
                voiced_mask_padded[i, :len(f0)] = voiced_mask

            if batch[ids_sorted_decreasing[i]]['p_voiced'] is not None:
                p_voiced = batch[ids_sorted_decreasing[i]]['p_voiced']
                p_voiced_padded[i, :len(f0)] = p_voiced

            if batch[ids_sorted_decreasing[i]]['energy_avg'] is not None:
                energy_avg = batch[ids_sorted_decreasing[i]]['energy_avg']
                energy_avg_padded[i, :len(energy_avg)] = energy_avg

            output_lengths[i] = mel.size(1)
            speaker_ids[i] = batch[ids_sorted_decreasing[i]]['speaker_id']
            audiopath = batch[ids_sorted_decreasing[i]]['audiopath']
            audiopaths.append(audiopath)
            cur_attn_prior = batch[ids_sorted_decreasing[i]]['attn_prior']
            if cur_attn_prior is None:
                attn_prior_padded = None
            else:
                attn_prior_padded[i, :cur_attn_prior.size(0), :cur_attn_prior.size(1)] = cur_attn_prior

        return {'mel': mel_padded,
                'speaker_ids': speaker_ids,
                'text': text_padded,
                'input_lengths': input_lengths,
                'output_lengths': output_lengths,
                'audiopaths': audiopaths,
                'attn_prior': attn_prior_padded,
                'f0': f0_padded,
                'p_voiced': p_voiced_padded,
                'voiced_mask': voiced_mask_padded,
                'energy_avg': energy_avg_padded
                }


# ===================================================================
# Takes directory of clean audio and makes directory of spectrograms
# Useful for making test sets
# ===================================================================
if __name__ == "__main__":
    # Get defaults so it can work with no Sacred
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str,
                        help='JSON file for configuration')
    parser.add_argument('-p', '--params', nargs='+', default=[])
    args = parser.parse_args()
    args.rank = 0

    # Parse configs.  Globals nicer in this case
    with open(args.config) as f:
        data = f.read()

    config = json.loads(data)
    update_params(config, args.params)
    print(config)

    data_config = config["data_config"]

    ignore_keys = ['training_files', 'validation_files']
    trainset = Data(data_config['training_files'],
                    **dict((k, v) for k, v in data_config.items()
                    if k not in ignore_keys))

    valset = Data(data_config['validation_files'],
                  **dict((k, v) for k, v in data_config.items()
                  if k not in ignore_keys), speaker_ids=trainset.speaker_ids)

    collate_fn = DataCollate()

    for dataset in (trainset, valset):
        for i, batch in enumerate(dataset):
            out = batch
            print("{}/{}".format(i, len(dataset)))
