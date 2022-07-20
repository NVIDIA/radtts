FROM nvcr.io/nvidia/pytorch:21.03-py3
RUN apt-get update -y
ENV TZ=America/Los_Angeles
RUN DEBIAN_FRONTEND="noninteractive" apt-get -y install tzdata
RUN apt-get install -y ffmpeg libsndfile1 sox locales vim
RUN pip install --upgrade pip
RUN pip install -U numpy
RUN pip install librosa==0.8.0 soundfile audioread sox matplotlib Pillow inflect unidecode natsort pandas jupyter tgt srt peakutils webrtcvad torchaudio lmdb
