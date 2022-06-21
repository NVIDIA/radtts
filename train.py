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
import argparse
import json
import os
import hashlib
import torch
from timeit import default_timer as timer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.cuda import amp
from radam import RAdam
from loss import RADTTSLoss, AttentionBinarizationLoss
from radtts import RADTTS
from data import Data, DataCollate
from plotting_utils import plot_alignment_to_numpy
from common import update_params
import numpy as np
from distributed import (init_distributed, apply_gradient_allreduce,
                         reduce_tensor)
from torch.utils.data.distributed import DistributedSampler
from inference import load_vocoder


def freeze(model):
    for p in model.parameters():
        p.requires_grad = False


def unfreeze(model):
    for p in model.parameters():
        p.requires_grad = True


def prepare_output_folders_and_logger(output_directory):
    # Get shared output_directory ready
    if not os.path.isdir(output_directory):
        os.makedirs(output_directory)
        os.chmod(output_directory, 0o775)
        print("output directory", output_directory)

    output_config_path = os.path.join(output_directory, 'config.json')
    print("saving current configuration in output dir")
    config_fp = open(output_config_path, 'w')
    json.dump(config, config_fp, indent=4)
    config_fp.close()
    output_code_path = os.path.join(output_directory, 'code.tar.gz')
    os.system('tar -czvf %s *.py' % (output_code_path))

    tboard_out_path = os.path.join(output_directory, 'logs')
    print("setting up tboard log in %s" % (tboard_out_path))
    logger = SummaryWriter(tboard_out_path)
    return logger


def prepare_model_weights(model, unfreeze_modules):
    if unfreeze_modules != 'all':
        freeze(model) # freeze everything
        if 'dur' in unfreeze_modules and hasattr(model, 'dur_pred_layer'):
            print("Training duration prediction")
            unfreeze(model.dur_pred_layer)
        if 'f0' in unfreeze_modules and hasattr(model, 'f0_pred_module'):
            print("Training F0 prediction")
            unfreeze(model.f0_pred_module)
        if 'energy' in unfreeze_modules and hasattr(model, 'energy_pred_module'):
            print("Training energy prediction")
            unfreeze(model.energy_pred_module)
        if 'vpred' in unfreeze_modules and hasattr(model, 'v_pred_module'):
            print("Training voiced prediction")
            unfreeze(model.v_pred_module)
            if hasattr(model, 'v_embeddings'):
                print("Training voiced embeddings")
                unfreeze(model.v_embeddings)
        if 'unvbias' in unfreeze_modules and hasattr(model, 'unvoiced_bias_module'):
            print("Training unvoiced bias")
            unfreeze(model.unvoiced_bias_module)
    else:
        print("Training everything")


def parse_data_from_batch(batch):
    mel = batch['mel']
    speaker_ids = batch['speaker_ids']
    text = batch['text']
    in_lens = batch['input_lengths']
    out_lens = batch['output_lengths']
    attn_prior = batch['attn_prior']
    f0 = batch['f0']
    voiced_mask = batch['voiced_mask']
    p_voiced = batch['p_voiced']
    energy_avg = batch['energy_avg']
    audiopaths = batch['audiopaths']
    if attn_prior is not None:
        attn_prior = attn_prior.cuda()
    if f0 is not None:
        f0 = f0.cuda()
    if voiced_mask is not None:
        voiced_mask = voiced_mask.cuda()
    if p_voiced is not None:
        p_voiced = p_voiced.cuda()
    if energy_avg is not None:
        energy_avg = energy_avg.cuda()

    mel, speaker_ids = mel.cuda(), speaker_ids.cuda()
    text = text.cuda()
    in_lens, out_lens = in_lens.cuda(), out_lens.cuda()

    return (mel, speaker_ids, text, in_lens, out_lens, attn_prior, f0,
            voiced_mask, p_voiced, energy_avg, audiopaths)


def prepare_dataloaders(data_config, n_gpus, batch_size):
    # Get data, data loaders and collate function ready
    ignore_keys = ['training_files', 'validation_files']
    print("initializing training dataloader")
    trainset = Data(data_config['training_files'],
                    **dict((k, v) for k, v in data_config.items()
                    if k not in ignore_keys))

    print("initializing validation dataloader")
    data_config_val = data_config.copy()
    data_config_val['aug_probabilities'] = None  # no aug in val set
    valset = Data(data_config['validation_files'],
                  **dict((k, v) for k, v in data_config_val.items()
                  if k not in ignore_keys), speaker_ids=trainset.speaker_ids)

    collate_fn = DataCollate()

    train_sampler, shuffle = None, True
    if n_gpus > 1:
        train_sampler, shuffle = DistributedSampler(trainset), False

    train_loader = DataLoader(trainset, num_workers=8, shuffle=shuffle,
                              sampler=train_sampler, batch_size=batch_size,
                              pin_memory=False, drop_last=True,
                              collate_fn=collate_fn)

    return train_loader, valset, collate_fn


def warmstart(checkpoint_path, model, include_layers=[],
              ignore_layers_warmstart=[]):
    pretrained_dict = torch.load(checkpoint_path, map_location='cpu')
    pretrained_dict = pretrained_dict['state_dict']

    if len(include_layers):
        pretrained_dict = {k: v for k, v in pretrained_dict.items()
                           if any(l in k for l in include_layers)}

    if len(ignore_layers_warmstart):
        pretrained_dict = {k: v for k, v in pretrained_dict.items()
                           if all(l not in k for l in ignore_layers_warmstart)}

    model_dict = model.state_dict()
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    print("Warm started from {}".format(checkpoint_path))
    return model


def load_checkpoint(checkpoint_path, model, optimizer, ignore_layers=[]):
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    iteration = checkpoint_dict['iteration']
    model_dict = checkpoint_dict['state_dict']
    optimizer.load_state_dict(checkpoint_dict['optimizer'])
    model.load_state_dict(model_dict)
    print("Loaded checkpoint '{}' (iteration {})" .format(
        checkpoint_path, iteration))
    return model, optimizer, iteration


def save_checkpoint(model, optimizer, learning_rate, iteration, filepath):
    print("Saving model and optimizer state at iteration {} to {}".format(
          iteration, filepath))

    torch.save({'state_dict': model.state_dict(),
                'iteration': iteration,
                'optimizer': optimizer.state_dict(),
                'learning_rate': learning_rate}, filepath)


def compute_validation_loss(iteration, model, criterion, valset, collate_fn,
                            batch_size, n_gpus, logger=None, train_config=None):

    model.eval()
    with torch.no_grad():
        val_sampler = DistributedSampler(valset) if n_gpus > 1 else None
        val_loader = DataLoader(valset, sampler=val_sampler, num_workers=8,
                                shuffle=False, batch_size=batch_size,
                                pin_memory=False, collate_fn=collate_fn)

        loss_outputs_full = {}
        n_batches = len(val_loader)
        for i, batch in enumerate(val_loader):
            (mel, speaker_ids, text, in_lens, out_lens, attn_prior,
             f0, voiced_mask, p_voiced, energy_avg,
             audiopaths) = parse_data_from_batch(batch)

            outputs = model(
                mel, speaker_ids, text, in_lens, out_lens,
                binarize_attention=True, attn_prior=attn_prior, f0=f0,
                energy_avg=energy_avg, voiced_mask=voiced_mask,
                p_voiced=p_voiced)
            loss_outputs = criterion(outputs, in_lens, out_lens)
            for k, (v, w) in loss_outputs.items():
                reduced_v = reduce_tensor(v, n_gpus, 0).item()
                if k in loss_outputs_full.keys():
                    loss_outputs_full[k] += (reduced_v / n_batches)
                else:
                    loss_outputs_full[k] = (reduced_v / n_batches)

    if logger is not None:
        for k, v in loss_outputs_full.items():
            logger.add_scalar('val/'+k, v, iteration)
        attn_used = outputs['attn']
        attn_soft = outputs['attn_soft']
        audioname = os.path.basename(audiopaths[0])
        if attn_used is not None:
            logger.add_image(
                'attention_weights',
                plot_alignment_to_numpy(
                    attn_soft[0, 0].data.cpu().numpy().T, title=audioname),
                iteration, dataformats='HWC')
            logger.add_image(
                'attention_weights_mas',
                plot_alignment_to_numpy(
                    attn_used[0, 0].data.cpu().numpy().T, title=audioname),
                iteration, dataformats='HWC')
            attribute_sigmas = []
            """ NOTE: if training vanilla radtts (no attributes involved),
            use log_attribute_samples only, as there will be no ground truth
            features available. The infer function in this case will work with
            f0=None, energy_avg=None, and voiced_mask=None
            """
            if train_config['log_decoder_samples']: # decoder with gt features
                attribute_sigmas.append(-1)
            if train_config['log_attribute_samples']: # attribute prediction
                if model.is_attribute_unconditional():
                    attribute_sigmas.extend([1.0])
                else:
                    attribute_sigmas.extend([0.1, 0.5, 0.8, 1.0])
            if len(attribute_sigmas) > 0:
                durations = attn_used[0, 0].sum(0, keepdim=True)
                durations = (durations + 0.5).floor().int()
                # load vocoder to CPU to avoid taking up valuable GPU vRAM
                vocoder_checkpoint_path = train_config['vocoder_checkpoint_path']
                vocoder_config_path = train_config['vocoder_config_path']
                vocoder, denoiser = load_vocoder(
                    vocoder_checkpoint_path, vocoder_config_path, to_cuda=False)
                for attribute_sigma in attribute_sigmas:
                    try:
                        if attribute_sigma > 0.0:
                            model_output = model.infer(
                                speaker_ids[0:1], text[0:1], 0.8,
                                dur=durations, f0=None, energy_avg=None,
                                voiced_mask=None, sigma_f0=attribute_sigma,
                                sigma_energy=attribute_sigma)
                        else:
                            model_output = model.infer(
                                speaker_ids[0:1], text[0:1], 0.8,
                                dur=durations, f0=f0[0:1, :durations.sum()],
                                energy_avg=energy_avg[0:1, :durations.sum()],
                                voiced_mask=voiced_mask[0:1, :durations.sum()])
                    except:
                        print("Instability or issue occured during inference, skipping sample generation for TB logger")
                        continue
                    mels = model_output['mel']
                    audio = vocoder(mels.cpu()).float()[0]
                    audio_denoised = denoiser(
                        audio, strength=0.00001)[0].float()
                    audio_denoised = audio_denoised[0].detach().cpu().numpy()
                    audio_denoised = audio_denoised / np.abs(audio_denoised).max()
                    if attribute_sigma < 0:
                        sample_tag = "decoder_sample_gt_attributes"
                    else:
                        sample_tag = f"sample_attribute_sigma_{attribute_sigma}"
                    logger.add_audio(sample_tag, audio_denoised, iteration, data_config['sampling_rate'])
    model.train()
    return loss_outputs_full


def train(n_gpus, rank, output_directory, epochs, optim_algo, learning_rate,
          weight_decay, sigma, iters_per_checkpoint, batch_size, seed,
          checkpoint_path, ignore_layers, ignore_layers_warmstart,
          include_layers, finetune_layers, warmstart_checkpoint_path,
          use_amp, grad_clip_val, loss_weights,
          binarization_start_iter=-1, kl_loss_start_iter=-1,
          unfreeze_modules="all", **kwargs):

    if seed is None:
        # convert output directory to seed using a hash
        print(output_directory)
        seed = hashlib.md5(output_directory.encode()).hexdigest()
        seed = int(seed, 16) % 2000
    print('Using seed {}'.format(seed))
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    if n_gpus > 1:
        init_distributed(rank, n_gpus, **dist_config)

    criterion = RADTTSLoss(
        sigma,
        model_config['n_group_size'],
        model_config['dur_model_config'],
        model_config['f0_model_config'],
        model_config['energy_model_config'],
        vpred_model_config=model_config['v_model_config'],
        loss_weights=loss_weights
    )
    attention_kl_loss = AttentionBinarizationLoss()
    model = RADTTS(**model_config).cuda()

    print("Initializing {} optimizer".format(optim_algo))
    if len(finetune_layers):
        for name, param in model.named_parameters():
            if any([l in name for l in finetune_layers]):  # short list hack
                print("Fine-tuning parameter", name)
                param.requires_grad = True
            else:
                param.requires_grad = False

    if optim_algo == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                                     weight_decay=weight_decay)
    elif optim_algo == 'RAdam':
        optimizer = RAdam(model.parameters(), lr=learning_rate,
                          weight_decay=weight_decay)
    else:
        print("Unrecognized optimizer {}!".format(optim_algo))
        exit(1)

    # Load checkpoint if one exists
    iteration = 0
    if warmstart_checkpoint_path != "":
        model = warmstart(warmstart_checkpoint_path, model, include_layers,
                          ignore_layers_warmstart)

    if checkpoint_path != "":
        model, optimizer, iteration = load_checkpoint(
            checkpoint_path, model, optimizer, ignore_layers)
        iteration += 1  # next iteration is iteration + 1

    if n_gpus > 1:
        model = apply_gradient_allreduce(model)
    print(model)
    scaler = amp.GradScaler(enabled=use_amp)

    for param_group in optimizer.param_groups:
        param_group['lr'] = learning_rate

    train_loader, valset, collate_fn = prepare_dataloaders(
        data_config, n_gpus, batch_size)

    if rank == 0:
        logger = prepare_output_folders_and_logger(output_directory)

    prepare_model_weights(model, unfreeze_modules)
    model.train()

    epoch_offset = max(0, int(iteration / len(train_loader)))
    # ================ MAIN TRAINNIG LOOP! ===================
    for epoch in range(epoch_offset, epochs):
        print("Epoch: {}".format(epoch))
        for batch in train_loader:
            tic = timer()
            model.zero_grad()
            (mel, speaker_ids, text, in_lens, out_lens, attn_prior,
             f0, voiced_mask, p_voiced, energy_avg,
             audiopaths) = parse_data_from_batch(batch)

            if iteration >= binarization_start_iter:
                binarize = True   # binarization training phase
            else:
                binarize = False  # no binarization, soft alignments only

            with amp.autocast(use_amp):
                outputs = model(
                    mel, speaker_ids, text, in_lens, out_lens,
                    binarize_attention=binarize, attn_prior=attn_prior,
                    f0=f0, energy_avg=energy_avg,
                    voiced_mask=voiced_mask, p_voiced=p_voiced)
                loss_outputs = criterion(outputs, in_lens, out_lens)

                loss = None
                for k, (v, w) in loss_outputs.items():
                    if w > 0:
                        loss = v * w if loss is None else loss + v * w

                w_bin = criterion.loss_weights.get('binarization_loss_weight', 1.0)
                if binarize and iteration >= kl_loss_start_iter:
                    binarization_loss = attention_kl_loss(
                        outputs['attn'], outputs['attn_soft'])
                    loss += binarization_loss * w_bin
                else:
                    binarization_loss = torch.zeros_like(loss)
                loss_outputs['binarization_loss'] = (binarization_loss, w_bin)

            scaler.scale(loss).backward()
            if grad_clip_val > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), grad_clip_val)
            scaler.step(optimizer)
            scaler.update()

            toc = timer()
            current_lr = optimizer.param_groups[0]['lr']
            print_list = ["iter: {}  ({:.2f} s)  |  lr: {}".format(
                iteration, toc-tic, current_lr)]

            for k, (v, w) in loss_outputs.items():
                reduced_v = reduce_tensor(v, n_gpus, 0).item()
                loss_outputs[k] = reduced_v
                if rank == 0:
                    print_list.append('  |  {}: {:.3f}'.format(k, v))
                    logger.add_scalar('train/'+k, reduced_v, iteration)

            if rank == 0:
                print(''.join(print_list), flush=True)

            if iteration > -1 and iteration % iters_per_checkpoint == 0:
                if rank == 0:
                    val_loss_outputs = compute_validation_loss(
                        iteration, model, criterion, valset, collate_fn,
                        batch_size, n_gpus, logger=logger,
                        train_config=train_config)
                    checkpoint_path = "{}/model_{}".format(
                        output_directory, iteration)
                    save_checkpoint(model, optimizer, learning_rate, iteration,
                                    checkpoint_path)
                    print('Validation loss:', val_loss_outputs)
                else:
                    val_loss_outputs = compute_validation_loss(
                        iteration, model, criterion, valset, collate_fn,
                        batch_size, n_gpus)

            iteration += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str,
                        help='JSON file for configuration')
    parser.add_argument('-p', '--params', nargs='+', default=[])
    args = parser.parse_args()
    args.rank = 0

    # Parse configs.  Globals nicer in this case
    with open(args.config) as f:
        data = f.read()

    global config
    config = json.loads(data)
    update_params(config, args.params)
    print(config)

    train_config = config["train_config"]
    global data_config
    data_config = config["data_config"]
    global dist_config
    dist_config = config["dist_config"]
    global model_config
    model_config = config["model_config"]


    # make sure we have enough augmentation dimensions
    if 'n_aug_dims' in model_config.keys() and \
        'aug_probabilities' in data_config.keys():
        assert(model_config['n_aug_dims'] >= len(data_config['aug_probabilities']))
    # Make sure the launcher sets `RANK` and `WORLD_SIZE`.
    rank = int(os.getenv('RANK', '0'))
    n_gpus = int(os.getenv("WORLD_SIZE", '1'))
    print('> got rank {} and world size {} ...'.format(rank, n_gpus))

    if n_gpus == 1 and rank != 0:
        raise Exception("Doing single GPU training on rank > 0")

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    train(n_gpus, rank, **train_config)
