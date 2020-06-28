## Finetuning GPT-2 models

Finetuning from the command line.


### Setup

`cd` into this folder.

Install packages from `requirements.txt`. Make sure `tensorflow` v1.15.x is installed, and `cudnn` is recommended too.


### Downloading pretrained GPT-2 model

`python download_model.py XXXX`, where `XXXX` is `117M` or `345M`. The models larger than these two require too much memory.


### Encoding dataset

Copy the text dataset into this folder then run `python src/encode.py datain.txt dataout.npz --model_name 117M`, replacing datain and dataout as needed.


### Finetuning

`python src/train.py --dataset dataout.npz`. It will keep finetuning until cancelled with ctrl+c. 

Other useful arguments:
- `model_name`: default 117M, change if needed
- `batch_size`: default 1
- `learning_rate`: default 0.00002 
- `run_name`: default run1, recommended to change to a more descriptive name 
- `sample_every`: number of steps between generating samples while training
- `save_every`: number of steps between saving the model
- `optimizer`: default adam, change to sgd to save memory
- `memory_saving_gradients`: default False, change to True to save memory



