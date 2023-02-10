This is a research project for a buinsess partner specializing on AI/chatbot application to explore the potentials of open source algorithms/pretrained model in voice analytics, especially speech recognition and diarization to achive outstanding WER, DER in a cost-effective way.

# Business objective: Investigate and assess state-of-the-art Speech-to-text (STT) technologies that work best in the chatbot context.

1.Identify relevant open-source datasets

2.Identify relevant performance metrics (e.g., F1 Score)

3.Investigate and compare existing COTS solutions
E.g., IBM Watson, Azure, Deepgram, AssemblyAI, Amazon Transcribe based on price, features, quality, etc.

4.Identify SOTA open-source algorithms/models
E.g., wav2vec, Speech2Text, Transformers to assess performance on open source and proprietary datasets, transcription quality, diarisation quality, etc.

5.Zero-shot and few-shot

# Technically, it is a quite productive project by leveraging different pre-trained models Wave2Vec2 and toolkits such as Mozila Deepspeech, SpeechBrain. 

Audio with Speaker Labels Identified
![image](https://user-images.githubusercontent.com/10097632/217982094-4a72ae50-bad8-4758-99fe-7b10e403b0d9.png)

![image](https://user-images.githubusercontent.com/10097632/217981960-06ab3ff6-29ff-4b36-b0e0-a0656f1b06c7.png)

Transcription along with speaker labels
![image](https://user-images.githubusercontent.com/10097632/217982003-2dfb9420-4094-4b52-a408-4b52572f0cc2.png)

![image](https://user-images.githubusercontent.com/10097632/217982041-668fd4d3-1148-4520-b4ec-29afe774de44.png)

![image](https://user-images.githubusercontent.com/10097632/217982159-08f7c64d-c14b-45a3-b8c7-791b0193e2c2.png)

![image](https://user-images.githubusercontent.com/10097632/217982191-51ac47c5-c2c0-401c-b50a-287a783f6ccb.png)

Approach:

We used Common Voice to compare models for STT accuracy, and we used AMI to compare models for diarization accuracy.

Dataset for Speech to Text (STT): Common Voice
![image](https://user-images.githubusercontent.com/10097632/217982806-111d2545-f43f-47e1-ade4-934d82860ad3.png)

![image](https://user-images.githubusercontent.com/10097632/217982842-a83ab2c8-994e-4264-8676-b29af95c5c24.png)

Dataset for Diarization: 
AMI Meeting Corpus
![image](https://user-images.githubusercontent.com/10097632/217982874-3e239748-7f36-4406-a555-a91b85797003.png)

![image](https://user-images.githubusercontent.com/10097632/217982894-f6ce5fe4-33c6-44c4-9f40-9ef0fb868d3f.png)


Challenges we encountered:
Size of the full datasets is large; computationally heavy on CPU / GPU and memory
Requires very extensive pipeline for data ETL
As data is managed separately by each team member, difficult to align consistent sampling/use of same data samples
Only a subset of the data was leveraged for this analysis due to computational constraints (see 1 and 2)
![image](https://user-images.githubusercontent.com/10097632/217982468-9a3a8818-61d9-49e1-90ed-a2e3951e3088.png)


The experiment result suggested that Speechbrain outperformed other open-source algorithms so only Speechbrain will be elaborated further and which is an **open-source** and **all-in-one** conversational AI toolkit based on PyTorch. *SpeechBrain is currently in beta

# Key features

SpeechBrain provides various useful tools to speed up and facilitate research on speech and language technologies:
- Various pretrained models nicely integrated with <img src="https://huggingface.co/front/assets/huggingface_logo.svg" alt="drawing" width="40"/> <sub>(HuggingFace)</sub> in our official [organization account](https://huggingface.co/speechbrain). These models are coupled with easy-inference interfaces that facilitate their use.  To help everyone replicate our results, we also provide all the experimental results and folders (including logs, tng curves, etc.) in a shared Google Drive folder.
- The `Brain` class is a fully-customizable tool for managing training and evaluation loops over data. The annoying details of training loops are handled for you while retaining complete flexibility to override any part of the process when needed.
- A YAML-based hyperparameter file that specifies all the hyperparameters, from individual numbers (e.g., learning rate) to complete objects (e.g., custom models). This elegant solution dramatically simplifies the training script.
- Multi-GPU training and inference with PyTorch Data-Parallel or Distributed Data-Parallel.
- Mixed-precision for faster training.
- A transparent and entirely customizable data input and output pipeline. SpeechBrain follows the PyTorch data loading style and enables users to customize the I/O pipelines (e.g., adding on-the-fly downsampling, BPE tokenization, sorting, threshold ...).
- On-the-fly dynamic batching
- Efficient reading of large datasets from a shared  Network File System (NFS) via [WebDataset](https://github.com/webdataset/webdataset).
- Interface with [HuggingFace](https://huggingface.co/speechbrain) for popular models such as wav2vec2  and Hubert.
- Interface with [Orion](https://github.com/Epistimio/orion) for hyperparameter tuning.

### Speaker recognition, identification and diarization

SpeechBrain provides different models for speaker recognition, identification, and diarization on different datasets:
- State-of-the-art performance on speaker recognition and diarization based on ECAPA-TDNN models.
- Original Xvectors implementation (inspired by Kaldi) with PLDA.
- Spectral clustering for speaker diarization (combined with speakers embeddings).
- Libraries to extract speaker embeddings with a pre-trained model on your data.

### Speech recognition

SpeechBrain supports state-of-the-art methods for end-to-end speech recognition:
- Support of wav2vec 2.0 pretrained model with finetuning.
- State-of-the-art performance or comparable with other existing toolkits in several ASR benchmarks.
- Easily customizable neural language models, including RNNLM and TransformerLM. We also share several pre-trained models that you can easily use (more to come!). We support the Hugging Face `dataset` to facilitate the training over a large text dataset.
- Hybrid CTC/Attention end-to-end ASR:
    - Many available encoders: CRDNN (VGG + {LSTM,GRU,LiGRU} + DNN), ResNet, SincNet, vanilla transformers, context net-based transformers or conformers. Thanks to the flexibility of SpeechBrain, any fully customized encoder could be connected to the CTC/attention decoder and trained in a few hours of work. The decoder is fully customizable: LSTM, GRU, LiGRU, transformer, or your neural network!
    - Optimised and fast beam search on both CPUs and GPUs.
- Transducer end-to-end ASR with both a custom Numba loss and the torchaudio one. Any encoder or decoder can be plugged into the transducer ranging from VGG+RNN+DNN to conformers.
- Pre-trained ASR models for transcribing an audio file or extracting features for a downstream task.

### Performance
The recipes released with speechbrain implement speech processing systems with competitive or state-of-the-art performance. In the following, we report the best performance achieved on some popular benchmarks:

| Dataset        | Task           | System  | Performance  |
| ------------- |:-------------:| -----:|-----:|
| CommonVoice (English) | Speech Recognition | wav2vec2 + CTC | WER=15.69% (test) |
| AMI      | Speaker Diarization | ECAPA-TDNN | DER=3.01% (eval)|

For more details, take a look at the corresponding implementation in recipes/dataset/.

### Pretrained Models

Beyond providing recipes for training the models from scratch, SpeechBrain shares several pre-trained models (coupled with easy-inference functions) on [HuggingFace](https://huggingface.co/speechbrain). In the following, we report some of them:

| Task        | Dataset | Model |
| ------------- |:-------------:| -----:|
| Speech Recognition | CommonVoice(English) | [wav2vec + CTC](https://huggingface.co/speechbrain/asr-wav2vec2-commonvoice-en) |
| Speaker Recognition | Voxceleb | [ECAPA-TDNN](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb) |

The full list of pre-trained models can be found on [HuggingFace](https://huggingface.co/speechbrain)

## Install via PyPI

Once you have created your Python environment (Python 3.8+) you can simply type:

```
pip install speechbrain
```

Then you can access SpeechBrain with:

```
import speechbrain as sb
```

## Install with GitHub

Once you have created your Python environment (Python 3.8+) you can simply type:

```
git clone https://github.com/speechbrain/speechbrain.git
cd speechbrain
pip install -r requirements.txt
pip install --editable.
```

Then you can access SpeechBrain with:

```
import speechbrain as sb
```

Any modification made to the `speechbrain` package will be automatically interpreted as we installed it with the `--editable` flag.

## Test Installation
Please, run the following script to make sure your installation is working:
```
pytest tests
pytest --doctest-modules speechbrain
```

# Running an experiment
In SpeechBrain, you can run experiments in this way:

```
> cd recipes/<dataset>/<task>/
> python experiment.py params.yaml
```

The results will be saved in the `output_folder` specified in the yaml file. The folder is created by calling `sb.core.create_experiment_directory()` in `experiment.py`. Both detailed logs and experiment outputs are saved there. Furthermore, less verbose logs are output to stdout.

# License
SpeechBrain is released under the Apache License, version 2.0. The Apache license is a popular BSD-like license. SpeechBrain can be redistributed for free, even for commercial purposes, although you can not take off the license headers (and under some circumstances, you may have to distribute a license document). Apache is not a viral license like the GPL, which forces you to release your modifications to the source code. Note that this project has no connection to the Apache Foundation, other than that we use the same license terms.

# Citing SpeechBrain
Please, cite SpeechBrain if you use it for your research or business.

```bibtex
@misc{speechbrain,
  title={{SpeechBrain}: A General-Purpose Speech Toolkit},
  author={Mirco Ravanelli and Titouan Parcollet and Peter Plantinga and Aku Rouhe and Samuele Cornell and Loren Lugosch and Cem Subakan and Nauman Dawalatabad and Abdelwahab Heba and Jianyuan Zhong and Ju-Chieh Chou and Sung-Lin Yeh and Szu-Wei Fu and Chien-Feng Liao and Elena Rastorgueva and François Grondin and William Aris and Hwidong Na and Yan Gao and Renato De Mori and Yoshua Bengio},
  year={2021},
  eprint={2106.04624},
  archivePrefix={arXiv},
  primaryClass={eess.AS},
  note={arXiv:2106.04624}
}
```

