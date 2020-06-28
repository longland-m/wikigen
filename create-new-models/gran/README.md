# Creating GRAN models


- put graph dataset (as pickle file) into the `data` folder
- edit config file:
  - change `dataset.file_name` to the graph dataset name
  - change parameters in `model` to the desired values for the model (all self-explanatory)
  - change `train` params like `max_epoch`, `lr` (learning rate), `batch_size` etc if desired
- `cd` into this folder
- run `python run_exp.py -c config/config_large.yaml` to train the model (replacing config_large.yaml if it's been renamed)
- run `python run_exp.py -c config/config_large.yaml -t` to test the model

Model will be in the folder `exp/GRAN/XXXX`, the last folder being named based on the date/time the experiment was run
