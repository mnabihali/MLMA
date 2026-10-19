🌍 MLMA

From Mono- to Multilingual ASR with Mamba-Based Architectures

Efficient multilingual speech recognition with bidirectional state-space models






Authors: Mohamed Nabih Ali, Daniele Falavigna, Alessio Brutti
Affiliation: SpeechTek, Fondazione Bruno Kessler (FBK), Trento, Italy
Contact: mnabih@fbk.eu

<br>

<img src="assets/Mamba%20arc.png" alt="MLMA architecture" width="820"/>

<br>

Overview ·
Architecture ·
Datasets ·
Results ·
Installation ·
Training ·
Citation

</div>

📖 Overview

MLMA explores Mamba-based architectures for multilingual automatic speech recognition (ASR). It replaces the self-attention component of a Conformer-style encoder with bidirectional Mamba blocks, combining local acoustic modeling with efficient sequence processing.

The model is trained jointly on six European languages with a shared character-level output vocabulary and a Connectionist Temporal Classification (CTC) objective.

Key contributions

🌐 Multilingual scale: joint training across English, Italian, French, Spanish, German, and Dutch.

🎙️ 11,840 hours of speech: a multilingual training mixture assembled from public speech corpora.

🧠 Mamba-based sequence modeling: bidirectional state-space blocks integrated into a Conformer-style encoder.

🔤 Shared CTC output: one multilingual character vocabulary across the six languages.

⚡ Long-audio efficiency: evaluation of encoder latency and memory as audio duration increases.

⚖️ Comparable model size: MLMA has 31.6M parameters, compared with 28.8M for the CTC-Conformer baseline.

Headline result: In the reported long-context experiment, MLMA encoder inference time increases from approximately 25 ms for 5-second audio to 75 ms for 256-second audio. The CTC-Conformer baseline increases from approximately 23 ms to 608 ms. The paper also reports less than 500 MB encoder memory for MLMA at 256 seconds, compared with more than 14 GB for Conformer.

🏗️ Architecture

The MLMA pipeline consists of four main stages:

Acoustic features — 80-dimensional log-Mel filterbanks, computed with a 25 ms window and 10 ms frame shift, followed by utterance-level mean and variance normalization.

CNN front end — two convolutional blocks capture local acoustic patterns and downsample the input sequence.

ConMamba encoder — an 18-layer encoder uses bidirectional Mamba for sequence modeling alongside convolutional modules for local context.

CTC prediction head — a linear projection and LogSoftmax produce frame-level character probabilities, trained with CTC loss.

Model configuration reported in the paper

<table>
<thead>
<tr><th align="left">Component</th><th align="left">Setting</th><th align="left">Component</th><th align="left">Setting</th></tr>
</thead>
<tbody>
<tr><td>Input features</td><td>80-dim log-Mel</td><td>Feature window / shift</td><td>25 ms / 10 ms</td></tr>
<tr><td>CNN front end</td><td>2 convolutional blocks</td><td>Encoder depth</td><td>18 layers</td></tr>
<tr><td>Hidden dimension</td><td>256</td><td>Feed-forward dimension</td><td>1,024</td></tr>
<tr><td>Dropout</td><td>0.1</td><td>Activation</td><td>GELU</td></tr>
<tr><td>Mamba state dimension</td><td>16</td><td>Mamba expansion</td><td>2</td></tr>
<tr><td>Mamba convolution width</td><td>4</td><td>Prediction objective</td><td>CTC</td></tr>
<tr><td>Output vocabulary</td><td colspan="3">Shared multilingual character vocabulary</td></tr>
<tr><td>Model size</td><td colspan="3">31.6M parameters (MLMA); 28.8M parameters (CTC-Conformer baseline)</td></tr>
</tbody>
</table>

These are the values reported in the paper. Check the selected configuration file in the code before assuming that every setting matches the reported experiment.

🌍 Datasets

MLMA is trained jointly on six languages. The training-hour totals below are reported in the paper.

<table>
<thead>
<tr><th align="left">Language</th><th align="center">Code</th><th align="right">Training hours</th><th align="left">Corpus sources</th></tr>
</thead>
<tbody>
<tr><td>English</td><td align="center"><code>en</code></td><td align="right">2,760</td><td>LibriSpeech, Common Voice, VoxPopuli-ASR</td></tr>
<tr><td>Italian</td><td align="center"><code>it</code></td><td align="right">574</td><td>Common Voice, MLS</td></tr>
<tr><td>French</td><td align="center"><code>fr</code></td><td align="right">2,112</td><td>Common Voice, VoxPopuli-ASR, MLS</td></tr>
<tr><td>Spanish</td><td align="center"><code>es</code></td><td align="right">1,569</td><td>Common Voice, MLS</td></tr>
<tr><td>German</td><td align="center"><code>de</code></td><td align="right">3,178</td><td>Common Voice, VoxPopuli-ASR, MLS</td></tr>
<tr><td>Dutch</td><td align="center"><code>nl</code></td><td align="right">1,646</td><td>Common Voice, MLS</td></tr>
<tr><th align="left">Total</th><td align="center">—</td><th align="right">11,840</th><td>Six-language training mixture</td></tr>
</tbody>
</table>

The training mixture draws on:

LibriSpeech (LS)

Common Voice v20.0 (CV)

VoxPopuli-ASR (VP-ASR)

Multilingual LibriSpeech (MLS)

Evaluation note: FLEURS is used as an unseen evaluation benchmark and is not included in the MLMA training mixture.

Please obtain each corpus from its official source and comply with its licensing and usage conditions.

📊 Results

WER is reported as a percentage; lower is better. Bold values indicate the better result within each comparison.

1. Monolingual ASR — LibriSpeech-960h

<table>
<thead>
<tr><th align="left">Model</th><th align="right">Parameters</th><th align="right">test-clean WER ↓</th><th align="right">test-other WER ↓</th></tr>
</thead>
<tbody>
<tr><td>CTC-Conformer</td><td align="right">28.8M</td><td align="right">4.27%</td><td align="right">11.29%</td></tr>
<tr><td><strong>MLMA</strong></td><td align="right"><strong>31.6M</strong></td><td align="right"><strong>4.05%</strong></td><td align="right"><strong>10.50%</strong></td></tr>
</tbody>
</table>

2. Multilingual ASR — Common Voice

<table>
<thead>
<tr><th align="left">Language</th><th align="center">Code</th><th align="right">CTC-Conformer WER ↓</th><th align="right">MLMA WER ↓</th><th align="right">Relative WER reduction</th></tr>
</thead>
<tbody>
<tr><td>English</td><td align="center"><code>en</code></td><td align="right">24.7%</td><td align="right"><strong>23.2%</strong></td><td align="right">6.1%</td></tr>
<tr><td>Italian</td><td align="center"><code>it</code></td><td align="right">16.8%</td><td align="right"><strong>13.0%</strong></td><td align="right">22.6%</td></tr>
<tr><td>French</td><td align="center"><code>fr</code></td><td align="right">17.8%</td><td align="right"><strong>15.0%</strong></td><td align="right">15.7%</td></tr>
<tr><td>Spanish</td><td align="center"><code>es</code></td><td align="right">14.2%</td><td align="right"><strong>11.2%</strong></td><td align="right">21.1%</td></tr>
<tr><td>German</td><td align="center"><code>de</code></td><td align="right">15.9%</td><td align="right"><strong>12.9%</strong></td><td align="right">18.9%</td></tr>
<tr><td>Dutch</td><td align="center"><code>nl</code></td><td align="right">18.9%</td><td align="right"><strong>16.8%</strong></td><td align="right">11.1%</td></tr>
</tbody>
</table>

3. Multilingual ASR — MLS

<table>
<thead>
<tr><th align="left">Language</th><th align="center">Code</th><th align="right">CTC-Conformer WER ↓</th><th align="right">MLMA WER ↓</th><th align="right">Relative WER reduction</th></tr>
</thead>
<tbody>
<tr><td>Italian</td><td align="center"><code>it</code></td><td align="right">14.8%</td><td align="right"><strong>13.3%</strong></td><td align="right">10.1%</td></tr>
<tr><td>French</td><td align="center"><code>fr</code></td><td align="right">10.7%</td><td align="right"><strong>9.1%</strong></td><td align="right">15.0%</td></tr>
<tr><td>Spanish</td><td align="center"><code>es</code></td><td align="right">7.3%</td><td align="right"><strong>6.5%</strong></td><td align="right">11.0%</td></tr>
<tr><td>German</td><td align="center"><code>de</code></td><td align="right">10.6%</td><td align="right"><strong>9.5%</strong></td><td align="right">10.4%</td></tr>
<tr><td>Dutch</td><td align="center"><code>nl</code></td><td align="right">17.3%</td><td align="right"><strong>14.8%</strong></td><td align="right">14.5%</td></tr>
</tbody>
</table>

4. Generalization — FLEURS (unseen benchmark)

<table>
<thead>
<tr><th align="left">Language</th><th align="center">Code</th><th align="right">CTC-Conformer WER ↓</th><th align="right">MLMA WER ↓</th><th align="right">Relative WER reduction</th></tr>
</thead>
<tbody>
<tr><td>English</td><td align="center"><code>en</code></td><td align="right">21.6%</td><td align="right"><strong>19.2%</strong></td><td align="right">11.1%</td></tr>
<tr><td>Italian</td><td align="center"><code>it</code></td><td align="right">14.8%</td><td align="right"><strong>12.5%</strong></td><td align="right">15.5%</td></tr>
<tr><td>French</td><td align="center"><code>fr</code></td><td align="right">21.1%</td><td align="right"><strong>19.6%</strong></td><td align="right">7.1%</td></tr>
<tr><td>Spanish</td><td align="center"><code>es</code></td><td align="right">12.6%</td><td align="right"><strong>10.6%</strong></td><td align="right">15.9%</td></tr>
<tr><td>German</td><td align="center"><code>de</code></td><td align="right">18.2%</td><td align="right"><strong>15.4%</strong></td><td align="right">15.4%</td></tr>
<tr><td>Dutch</td><td align="center"><code>nl</code></td><td align="right">30.6%</td><td align="right"><strong>27.9%</strong></td><td align="right">8.8%</td></tr>
</tbody>
</table>

The relative WER reductions above are calculated from the displayed WER values. Results are transcribed from the paper's reported tables; evaluation setups and training data may differ across external systems.

5. Long-context efficiency

<table>
<thead>
<tr><th align="left">Metric</th><th align="right">MLMA</th><th align="right">CTC-Conformer</th></tr>
</thead>
<tbody>
<tr><td>Encoder latency — 5 s audio</td><td align="right">≈ 25 ms</td><td align="right">≈ 23 ms</td></tr>
<tr><td>Encoder latency — 256 s audio</td><td align="right">≈ 75 ms</td><td align="right">≈ 608 ms</td></tr>
<tr><td>Latency growth — 5 s to 256 s</td><td align="right"><strong>≈ 3.0×</strong></td><td align="right">≈ 26.7×</td></tr>
<tr><td>Encoder memory — 256 s audio</td><td align="right"><strong>&lt; 500 MB</strong></td><td align="right">&gt; 14 GB</td></tr>
</tbody>
</table>

Actual latency and memory depend on the hardware, precision, batch size, and software environment used.

⚙️ Installation

1. Clone the repository

git clone https://github.com/mnabihali/MLMA.git
cd MLMA

2. Create the environment

The upstream Mamba-ASR project documents a Python 3.9 environment. Use this as a starting point and adjust versions to match your CUDA and GPU environment.

conda create --name mlma python=3.9 --yes
conda activate mlma

3. Install dependencies

Run the command that matches the requirements filename included in your checkout:

<details>
<summary><strong>📦 If the repository contains <code>requirements.txt</code></strong></summary>

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

</details>

<details>
<summary><strong>📦 If the repository contains <code>requirement.txt</code></strong></summary>

python -m pip install --upgrade pip
python -m pip install -r requirement.txt

</details>

Compatibility tip: Mamba-related dependencies can be sensitive to the exact PyTorch, CUDA, compiler, mamba-ssm, and causal-conv1d versions. Install compatible versions for your system before launching a long training run.

🚀 Training

The CTC training entry point is train_CTC.py. The command below follows the torchrun style used by the upstream Mamba-ASR project. Replace the configuration and data paths with the ones supplied by your MLMA checkout.

1. Configure your experiment

Before training, identify the YAML configuration file for the MLMA experiment and the dataset directory expected by that configuration.

<MLMA_REPOSITORY>/
├── train_CTC.py
├── <PATH_TO_CTC_CONFIG.yaml>
└── <PATH_TO_DATA>/

2. Launch multi-GPU training

torchrun \
  --standalone \
  --nnodes=1 \
  --nproc_per_node=4 \
  train_CTC.py \
  <PATH_TO_CTC_CONFIG.yaml> \
  --data_folder <PATH_TO_DATA> \
  --precision bf16

Before running, replace these placeholders:

Placeholder

Replace with

<PATH_TO_CTC_CONFIG.yaml>

Path to the YAML configuration for the MLMA experiment

<PATH_TO_DATA>

Dataset root expected by the training recipe

4

Number of GPUs assigned to this run

bf16

Precision supported by your GPU and codebase

Important: The exact configuration filename and data layout must match the files and argument parser in your checkout. The command above is a launch template, not a claim that a particular MLMA configuration is already present in the public repository.

<details>
<summary><strong>🔧 Upstream Mamba-ASR example (LibriSpeech / ConMamba)</strong></summary>

The upstream project documents the following CTC recipe. Use it only if you are intentionally running that upstream configuration and its expected LibriSpeech layout:

torchrun --nproc-per-node 4 train_CTC.py \
  hparams/CTC/conmamba_large.yaml \
  --data_folder <PATH_TO_LIBRISPEECH> \
  --precision bf16

</details>

3. Pre-flight checklist

Confirm the configuration file exists and matches the model variant you want to train.

Prepare the required datasets and verify all paths/manifests.

Confirm the multilingual character vocabulary covers all training transcripts.

Verify GPU count, CUDA, PyTorch, and Mamba package compatibility.

Use a precision mode supported by the hardware and training code.

Save the configuration and software versions with the experiment for reproducibility.

📦 Checkpoints and inference

The paper refers to checkpoints being available through the SpeechTek Hugging Face organization.

The exact checkpoint identifier and inference command are not specified here. Add the verified model ID, download instructions, and a runnable transcription example once those interfaces are published for this repository.

🗂️ Repository structure

The current public repository should be treated as the source of truth for the available scripts and configurations. The following is an illustrative layout for documenting the training components; it does not assert that every path already exists.

MLMA/
├── assets/
│   └── Mamba arc.png
├── train_CTC.py
├── <MLMA_CTC_config>.yaml
├── <dataset_root>/
└── README.md

📄 Citation

If you use MLMA in your research, please cite the paper:

@article{ali2025mlma,
  title   = {MLMA: Towards Multilingual ASR with Mamba-Based Architectures},
  author  = {Ali, Mohamed Nabih and Falavigna, Daniele and Brutti, Alessio},
  journal = {arXiv preprint arXiv:2510.18684},
  year    = {2025},
  eprint  = {2510.18684},
  archivePrefix = {arXiv}
}

🙏 Acknowledgements

This work was carried out at the SpeechTek unit, Fondazione Bruno Kessler (FBK), Trento, Italy.

📬 Contact

For questions or issues, please open a GitHub issue or contact Mohamed Nabih Ali.

📜 License

Please refer to the repository's LICENSE file for the applicable license. If no license file is present, clarify the license before redistributing the code.
