
<div align="center">

🌍 MLMA

Multilingual Automatic Speech Recognition with Mamba-Based Architectures

Scaling efficient state-space models to multilingual speech recognition


<br>

<img src="assets/Mamba%20arc.png" alt="MLMA architecture" width="850">

<br>

MLMA explores Mamba-based sequence modeling for multilingual ASR, combining a convolutional front end, a ConMamba-style encoder with bidirectional Mamba, and Connectionist Temporal Classification (CTC).

Overview · Architecture · Results · Installation · Training · Citation

</div>

🔎 Overview

Automatic speech recognition systems must model long, variable-length speech sequences while learning acoustic patterns shared across languages. Transformer-based encoders can be effective, but self-attention becomes increasingly expensive as sequence length grows.

MLMA investigates selective state-space modeling as an efficient alternative for multilingual ASR. The model is trained jointly across six European languages using a shared multilingual character vocabulary and a CTC objective.

Highlights

🌐 Six languages: English, Italian, French, Spanish, German, and Dutch.

🎙️ 11,840 hours of multilingual transcribed speech used for training.

🧩 Mamba-based encoder: combines local convolutional processing with bidirectional state-space sequence modeling.

🔤 End-to-end CTC recognition: shared character-level output vocabulary across languages.

⚡ Long-context efficiency: evaluates inference latency and encoder memory as audio duration increases.

🔬 Matched baseline: comparison with a CTC-Conformer of similar model size.

Main result: In the paper's long-context efficiency experiment, encoder inference time grows from approximately 25 ms to 75 ms for MLMA as audio duration increases from 5 s to 256 s, compared with 23 ms to 608 ms for the CTC-Conformer baseline. At 256 s, the paper reports encoder memory below 500 MB for MLMA, versus more than 14 GB for Conformer.

🧠 Architecture

<p align="center">
  <img src="assets/Mamba%20arc.png" alt="MLMA model architecture" width="850">
</p>

The MLMA pipeline follows these stages:

Acoustic features: input audio is converted to 80-dimensional log-Mel filterbank features using a 25 ms analysis window and a 10 ms frame shift. Utterance-level mean and variance normalization is applied.

CNN feature extractor: two convolutional blocks extract local acoustic features and downsample the time dimension.

ConMamba encoder: an 18-layer encoder uses a Conformer-style block in which bidirectional Mamba replaces multi-head self-attention, while the convolution module models local patterns.

CTC output layer: a linear projection and LogSoftmax produce frame-level character probabilities, optimized with CTC loss.

Reported model configuration

Component

Configuration

Input features

80-dimensional log-Mel filterbanks

Feature window / shift

25 ms / 10 ms

CNN front end

2 convolutional blocks

Encoder depth

18 layers

Encoder hidden dimension

256

Feed-forward dimension

1024

Dropout

0.1

Activation

GELU

Bi-Mamba state dimension

16

Bi-Mamba expansion

2

Bi-Mamba convolution width

4

Output

Shared multilingual character vocabulary

Objective

Connectionist Temporal Classification (CTC)

Parameters

31.6M for the reported MLMA model

These values describe the configuration reported in the paper. Check the selected training configuration in the repository before assuming every setting is identical.

🌍 Languages and datasets

MLMA is trained jointly on six languages:

Language

Code

Training hours reported in the paper

English

en

2,760

Italian

it

574

French

fr

2,112

Spanish

es

1,569

German

de

3,178

Dutch

nl

1,646

Total

—

11,840

The training mixture draws on:

LibriSpeech (LS) — English

Common Voice v20.0 (CV)

VoxPopuli-ASR (VP-ASR)

Multilingual LibriSpeech (MLS)

FLEURS is reserved for evaluation and is not used for training. It is used to assess generalization to an unseen benchmark.

Dataset access, licensing, and preparation requirements are governed by the individual corpus providers. Please obtain each dataset from its official source and follow its terms of use.

📊 Results

Monolingual LibriSpeech-960h

MLMA and the CTC-Conformer baseline are trained from scratch on LibriSpeech-960h under matched conditions.

Model

Parameters

Test-clean WER ↓

Test-other WER ↓

CTC-Conformer

28.8M

4.27%

11.29%

MLMA

31.6M

4.05%

10.50%

Multilingual comparison with CTC-Conformer

Selected results reported in the paper:

Evaluation set

Language

CTC-Conformer WER ↓

MLMA WER ↓

LibriSpeech

English

8.7%

7.2%

Common Voice

English

24.7%

23.2%

Common Voice

Italian

16.8%

13.0%

Common Voice

French

17.8%

15.0%

Common Voice

Spanish

14.2%

11.2%

Common Voice

German

15.9%

12.9%

Common Voice

Dutch

18.9%

16.8%

MLS

Italian

14.8%

13.3%

MLS

French

10.7%

9.1%

MLS

Spanish

7.3%

6.5%

MLS

German

10.6%

9.5%

MLS

Dutch

17.3%

14.8%

FLEURS (unseen)

English

21.6%

19.2%

FLEURS (unseen)

Italian

14.8%

12.5%

FLEURS (unseen)

French

21.1%

19.6%

FLEURS (unseen)

Spanish

12.6%

10.6%

FLEURS (unseen)

German

18.2%

15.4%

FLEURS (unseen)

Dutch

30.6%

27.9%

WER values are transcribed from the paper's multilingual results table. FLEURS is an unseen evaluation benchmark for MLMA; comparisons with systems trained on larger corpora are not necessarily matched-data comparisons.

Long-context efficiency

The paper evaluates encoder inference as audio duration increases from 5 to 256 seconds.

Metric

MLMA

CTC-Conformer

Encoder inference time, 5 s

~25 ms

~23 ms

Encoder inference time, 256 s

~75 ms

~608 ms

Time growth, 5 s → 256 s

~3.0×

~26.7×

Encoder peak memory at 256 s

<500 MB (reported in abstract/conclusion)

>14 GB

These measurements are from the paper's reported experimental setup; actual speed and memory depend on hardware, precision, batch size, and software versions.

🛠️ Installation

MLMA builds on Mamba-ASR/ConMamba and uses PyTorch-based speech training components. Mamba-related packages can be sensitive to CUDA, PyTorch, compiler, and GPU compatibility.

1. Clone the repository

git clone https://github.com/mnabihali/MLMA.git
cd MLMA

2. Create an environment

The upstream Mamba-ASR project documents a Python 3.9 environment. Treat this as a starting point, and adapt it to the versions required by your target machine.

conda create -n mlma python=3.9
conda activate mlma

3. Install dependencies

If this repository contains a requirements file, install it with:

pip install -r requirements.txt

If the file is named requirement.txt instead, use:

pip install -r requirement.txt

Before training, verify that your installed PyTorch, CUDA, mamba-ssm, and causal-conv1d versions are mutually compatible. Follow the installation instructions for the exact Mamba implementation used by your checkout.

🚀 Training

MLMA uses the CTC training entry point, train_CTC.py. The upstream Mamba-ASR recipe launches CTC training with torchrun, a YAML hyperparameter file, a data directory, and a precision setting.

Multi-GPU training

From the repository root, the upstream-style command is:

torchrun --nproc-per-node 4 train_CTC.py <PATH_TO_CTC_CONFIG.yaml> --data_folder <PATH_TO_DATA> --precision bf16

Replace:

<PATH_TO_CTC_CONFIG.yaml> with the MLMA CTC configuration you intend to train.

<PATH_TO_DATA> with the directory expected by the dataset preparation pipeline.

4 with the number of GPUs available to the run, if supported by your distributed setup.

bf16 with a supported precision setting for your GPU and training configuration.

For example, if you are using the upstream configuration layout unchanged, the original Mamba-ASR recipe is:

torchrun --nproc-per-node 4 train_CTC.py hparams/CTC/conmamba_large.yaml --data_folder <PATH_TO_LIBRISPEECH> --precision bf16

This example is the upstream LibriSpeech recipe, not a guarantee that the same YAML file or data layout is appropriate for multilingual MLMA training. Use the MLMA-specific configuration and dataset manifest expected by your version of the code.

Training checklist

Download and prepare the required corpora.

Verify the dataset manifests and paths used by the selected configuration.

Confirm that the output vocabulary covers the languages and transcripts in the training mixture.

Check GPU, CUDA, PyTorch, and Mamba package compatibility.

Choose a precision supported by your hardware.

Save the training configuration and software versions alongside each experiment for reproducibility.

📦 Checkpoints and inference

Pretrained model checkpoints are referenced in the paper as available through the SpeechTek Hugging Face organization.

This README does not assume a specific checkpoint filename or inference command. Once the checkpoint and inference interface are published for this repository, add the exact download command and a minimal end-to-end transcription example here.

🗂️ Repository structure

The repository includes an assets/ directory for README figures. The training entry point used for CTC experiments is train_CTC.py; the exact configuration and dataset directory layout should be documented alongside the corresponding code and training recipe.

A useful structure for a reproducible release is:

MLMA/
├── assets/
│   └── Mamba arc.png
├── hparams/
│   └── CTC/
│       └── <MLMA CTC configuration>.yaml
├── train_CTC.py
├── requirement.txt  # or requirements.txt
└── README.md

The tree above is illustrative, not a claim that every listed file is currently present in the repository.

📄 Citation

If you use MLMA in your research, please cite the paper:

@article{ali2025mlma,
  title   = {MLMA: Towards Multilingual ASR with Mamba-Based Architectures},
  author  = {Ali, Mohamed Nabih and Falavigna, Daniele and Brutti, Alessio},
  year    = {2025},
  eprint  = {2510.18684},
  archivePrefix = {arXiv},
  primaryClass  = {eess.AS},
  url     = {https://arxiv.org/abs/2510.18684}
}

Please check the paper's current publication status and final bibliographic details before using this citation in a camera-ready publication.

🙏 Acknowledgements

MLMA builds on ideas and code from the Mamba and ConMamba research communities. The training recipe is inspired by the Mamba-ASR repository, and the experimental work uses publicly available speech corpora. Please also consult the upstream projects and dataset providers for their respective licenses and acknowledgements.

📜 License

Please refer to the LICENSE file in this repository. If no license file is present, the code should not be assumed to be released under a particular open-source license.

<div align="center">

MLMA — efficient sequence modeling for multilingual speech recognition

Developed at SpeechTek · Fondazione Bruno Kessler (FBK)

</div>
