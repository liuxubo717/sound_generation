import sys
sys.path.append("hifi_gan/")

from hifi_gan import inference_e2e as vocoder
import argparse

h = None
device = None

def Mel2Audio():
    print('Initializing Inference Process..')

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_mels_dir', default='/home/lxb/Desktop/audio_generation/sample-results/v2-result-1.0')
    parser.add_argument('--output_dir', default='/home/lxb/Desktop/audio_generation/generated_samples')
    parser.add_argument('--ckpt', default='hifi_gan/cp_hifigan/g_00335000')
    # parser.add_argument('--checkpoint_file', default='hifi_gan/checkpoint-2GPU/g_00125000')
    a = parser.parse_args()

    vocoder.main(a)

if __name__ == '__main__':
    Mel2Audio()






