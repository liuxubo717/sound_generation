# Conditional Sound Generation Using Neural Discrete Time-Frequency Representation Learning

This repository contains the code and generated sound samples of our paper *"Conditional Sound Generation Using Neural Discrete Time-Frequency Representation Learning"*, which was accepted for MLSP 2021. 

## Set up environment

* Clone the repository: `git clone https://github.com/liuxubo717/sound_generation.git`
* Create conda environment with dependencies: `conda create -f environment.yml -n sound_generation`
* Activate conda environment:  `conda activate sound_generation`

## Prepare dataset

## Usage

1: (Stage 1) train a multi-scale VQ-VAE to extract the Discrete T-F Representation (DTFR) of sound: 

`python train_vqvae.py --epoch 800`

2: Extract DTFR for stage 2 training: 

`python extract_code.py --ckpt checkpoint/[VQ-VAE CHECKPOINT] `

3: (Stage 3) train a PixelSNAIL model on the extracted DTFR of sound: 

`python train_pixelsnail.py --epoch 2000`

4: Sample mel-spectrogram of sound from the trained PixelSNAIL model:

`python mel_sample.py --vqvae checkpoint/[VQ-VAE CHECKPOINT] --bottom checkpoint/[PixelSNAIL CHECKPOINT] --label [Class ID: 0-9]`

5: Synthesize waveform of sound using HiFi-GAN vocoder:

`python mel2audio.py --input_mels_dor [INPUT MEL-SPECTROGRAM PATH] --output_dir [OUTPUT WAVEFORM PATH]`

The trained HiFi-GAN checkpoint is provided in `/hifi_gan/cp_hifigan/g_00335000`

## Generated samples

The generated sound samples are available at  `/generated_sounds`

## Cite

If you use our code, please kindly cite following: 

````
```
@article{liu2021conditional,
  title={Conditional Sound Generation Using Neural Discrete Time-Frequency Representation Learning},
  author={Liu, Xubo and Iqbal, Turab and Zhao, Jinzheng and Huang, Qiushi and Plumbley, Mark D and Wang, Wenwu},
  journal={arXiv preprint arXiv:2107.09998},
  year={2021}
}
```
````
