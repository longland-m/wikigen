# WikiGen

This is the code for the WikiGen project, Semester 1 2020.

The code uses elements from [OpenAI's GPT-2](https://github.com/openai/gpt-2), [GRAN](https://github.com/lrjconan/GRAN), and [SIF](https://github.com/PrincetonML/SIF/).

## Setup

The code is written in Python 3.7. 

Install dependencies with:
```
$ pip install -r requirements.txt
```

Two other files are also required. 

SIF requires paragram_sl999_small.txt. Run `setup.sh` to download and save it to the SIF folder.

GPT-2 requires a pretrained model checkpoint, available from: https://drive.google.com/file/d/1pDzNpPU3-TnKgMgWgjiM5Cxe4hvg0qeq/view
Unzip it and place the `smallWikiPages` folder in the `gpt2/models` folder. Due to the large file size this must be done manually.


## Run Demo

The `example_run.ipynb` shows a demo of the WikiGen model.
