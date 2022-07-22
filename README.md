# Flow-based TTS with Robust Alignment Learning, Diverse Synthesis, and Generative Modeling and Fine-Grained Control over of Low Dimensional (F0 and Energy) Speech Attributes.
This repository contains the source code and several checkpoints for our work based on RADTTS. RADTTS is a normalizing-flow-based TTS framework with state of the art acoustic fidelity and a highly robust audio-transcription alignment module. Our project page and some samples can be found [here](https://nv-adlr.github.io/RADTTS), with relevant works listed [here](#relevant-papers).

This repository can be used to train the following models:

- A normalizing-flow bipartite architecture for mapping text to mel spectrograms
- A variant of the above, conditioned on F0 and Energy
- Normalizing flow models for explicitly modeling text-conditional phoneme duration, fundamental frequency (F0), and energy
- A standalone alignment module for learning unspervised text-audio alignments necessary for TTS training

## HiFi-GAN vocoder pre-trained models
We provide a [checkpoint](https://drive.google.com/file/d/1lD62jl5hF6T5AkGoWKOcgMZuMR4Ir76d/view?usp=sharing) and [config](https://drive.google.com/file/d/1WRtyvkmQxlYShkeTwWmlj7_WiS70R7Jb/view?usp=sharing) for a HiFi-GAN vocoder trained on LibriTTS 100 and 360.<br>
For a HiFi-GAN vocoder trained on LJS, please download the v1 model provided by the HiFi-GAN authors [here](https://github.com/jik876/hifi-gan), .

## RADTTS pre-trained models
| Model name                | Description                                             | Dataset                                      | 
|---------------------------|---------------------------------------------------------|----------------------------------------------|
| [RADTTS++DAP-LJS](https://drive.google.com/file/d/1Rb2VMUwQahGrnpFSlAhCPh7OpDN3xgOr/view?usp=sharing) | RADTTTS model conditioned on F0 and Energy with deterministic attribute predictors | LJSpeech Dataset 


We will soon provide more pre-trained RADTTS models with generative attribute predictors trained on LJS and LibriTTS. Stay tuned!


## Setup
1. Clone this repo: `git clone https://github.com/NVIDIA/RADTTS.git`
2. Install python requirements or build docker image
    - Install python requirements: `pip install -r requirements.txt`
3. Update the filelists inside the filelists folder and json configs to point to your data
    - `basedir` – the folder containing the filelists and the audiodir
    - `audiodir` – name of the audiodir
    - `filelist` – <mark>|</mark> (pipe) separated text file with relative audiopath, text, speaker, and optionally categorical label and audio duration in seconds
## Training RADTTS (without pitch and energy conditioning)
1. Train the decoder <br> 
	`python train.py -c config_ljs_radtts.json -p train_config.output_directory=outdir`
2. Further train with the duration predictor
	`python train.py -c config_ljs_radtts.json -p train_config.output_directory=outdir_dir train_config.warmstart_checkpoint_path=model_path.pt model_config.include_modules="decatndur"`


## Training RADTTS++ (with pitch and energy conditioning)
1. Train the decoder<br> 
	`python train.py -c config_ljs_decoder.json -p train_config.output_directory=outdir`
2. Train the attribute predictor: autoregressive flow (agap), bi-partite flow (bgap) or deterministic (dap)<br>
    `python train.py -c config_ljs_{agap,bgap,dap}.json -p train_config.output_directory=outdir_wattr train_config.warmstart_checkpoint_path=model_path.pt`


## Training starting from a pre-trained model, ignoring the speaker embedding table
1. Download our pre-trained model
2. `python train.py -c config.json -p train_config.ignore_layers_warmstart=["speaker_embedding.weight"] train_config.warmstart_checkpoint_path=model_path.pt`

## Multi-GPU (distributed)
1. `python -m torch.distributed.launch --use_env --nproc_per_node=NUM_GPUS_YOU_HAVE train.py -c config.json -p train_config.output_directory=outdir`

## Inference demo
1. `python inference.py -c CONFIG_PATH -r RADTTS_PATH -v HG_PATH -k HG_CONFIG_PATH -t TEXT_PATH -s ljs --speaker_attributes ljs --speaker_text ljs -o results/`


## Inference Voice Conversion demo 
1. `python inference_voice_conversion.py --radtts_path RADTTS_PATH --radtts_config_path RADTTS_CONFIG_PATH --vocoder_path HG_PATH --vocoder_config_path HG_CONFIG_PATH --f0_mean=211.413 --f0_std=46.6595 --energy_mean=0.724884 --energy_std=0.0564605 --output_dir=results/ -p data_config.validation_files="{'Dummy': {'basedir': 'data/', 'audiodir':'22khz', 'filelist': 'vc_audiopath_txt_speaker_emotion_duration_filelist.txt'}}"`

## Config Files
| Filename                 | Description                                             | Nota bene                                      |
|--------------------------|---------------------------------------------------------|------------------------------------------------|
| [config\_ljs_decoder.json](https://github.com/NVIDIA/radtts/blob/main/configs/config_ljs_decoder.json) | Config for the decoder conditioned on F0 and Energy     |                                                |
| [config\_ljs_radtts.json](https://github.com/NVIDIA/radtts/blob/main/configs/config_ljs_radtts.json)   | Config for the decoder not conditioned on F0 and Energy |                                                |
| [config\_ljs_agap.json](https://github.com/NVIDIA/radtts/blob/main/configs/config_ljs_agap.json)       | Config for the Autoregressive Flow Attribute Predictors | Requires at least pre-trained alignment module |
| [config\_ljs_bgap.json](https://github.com/NVIDIA/radtts/blob/main/configs/config_ljs_bgap.json)       | Config for the Bi-Partite Flow Attribute Predictors     | Requires at least pre-trained alignment module |
| [config\_ljs_dap.json](https://github.com/NVIDIA/radtts/blob/main/configs/config_ljs_dap.json)         | Config for the Deterministic Attribute Predictors       | Requires at least pre-trained alignment module |


## LICENSE
Unless otherwise specified, the source code within this repository is provided under the
[MIT License](LICENSE)

## Acknowledgements
The code in this repository is heavily inspired by or makes use of source code from the following works:

- Tacotron implementation from [Keith Ito](https://github.com/keithito/tacotron/)
- STFT code from [Prem Seetharaman](https://github.com/pseeth/pytorch-stft)
- [Masked Autoregressive Flows](https://arxiv.org/abs/1705.07057)
- [Flowtron](https://arxiv.org/abs/2005.05957)
- Source for neural spline functions used in this work: https://github.com/ndeutschmann/zunis 
- Original Source for neural spline functions: https://github.com/bayesiains/nsf 
- Bipartite Architecture based on code from [WaveGlow](https://github.com/NVIDIA/waveglow) 
- [HiFi-GAN](https://github.com/jik876/hifi-gan) 
- [Glow-TTS](https://github.com/jaywalnut310/glow-tts) 

## Relevant Papers

Rohan Badlani, Adrian Łańcucki, Kevin J. Shih, Rafael Valle, Wei Ping, Bryan Catanzaro. <br/>[One TTS Alignment to Rule Them All.](https://ieeexplore.ieee.org/abstract/document/9747707) ICASSP 2022
<br/><br/>
Kevin J Shih, Rafael Valle, Rohan Badlani, Adrian Lancucki, Wei Ping, Bryan Catanzaro. <br/>[RAD-TTS: Parallel flow-based TTS with robust alignment learning and diverse synthesis.](https://openreview.net/pdf?id=0NQwnnwAORi)<br/> ICML Workshop on Invertible Neural Networks, Normalizing Flows, and Explicit Likelihood Models 2021
<br/><br/>
Kevin J Shih, Rafael Valle, Rohan Badlani, João Felipe Santos, Bryan Catanzaro.<br/>[Generative Modeling for Low Dimensional Speech Attributes with Neural Spline Flows.](https://arxiv.org/pdf/2203.01786) Technical Report
