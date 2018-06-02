# Simple Neuarl Machine Translation Toolkit
This repo contains a simple source code for advanced neural machine translation based on sequence-to-sequence. Most open sources are unnecessarily too complicated, so those have too many features more than people's expected. Therefore, I hope that this repo can be a good solution for people who doesn't want unnecessarily many features.

## Basic Features

1. LSTM sequence-to-seuqnce with attention
2. Beam search with mini-batch in parallel

## Usage

### 1. Training

```
$ python train.py -h
usage: train.py [-h] -model MODEL -train TRAIN -valid VALID -lang LANG
                [-gpu_id GPU_ID] [-batch_size BATCH_SIZE] [-n_epochs N_EPOCHS]
                [-print_every PRINT_EVERY] [-early_stop EARLY_STOP]
                [-max_length MAX_LENGTH] [-dropout DROPOUT]
                [-word_vec_dim WORD_VEC_DIM] [-hidden_size HIDDEN_SIZE]
                [-n_layers N_LAYERS] [-max_grad_norm MAX_GRAD_NORM] [-adam]
                [-lr LR] [-min_lr MIN_LR]
                [-lr_decay_start_at LR_DECAY_START_AT] [-lr_slow_decay]
```

example usage:
```
$ python train.py -model ./model/enko.pth -train ./data/train -valid ./data/valid -lang enko -gpu_id 0 -batch_size 32 -n_epochs 13 -word_vec_dim 256 -hidden_size 768
```

You may need to change the argument parameters.

### 2. Inference

```
$ python translate.py -h
usage: translate.py [-h] -model MODEL [-gpu_id GPU_ID]
                    [-batch_size BATCH_SIZE] [-max_length MAX_LENGTH]
                    [-n_best N_BEST] [-beam_size BEAM_SIZE]
```

example usage:
```
$ python translate.py -model ./model/enko.12.1.18-3.24.1.37-3.92.pth -gpu_id 0 -batch_size 128 -beam_size 5
```

You may also need to change the argument parameters.