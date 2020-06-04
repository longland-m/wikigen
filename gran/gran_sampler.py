import os
import time
import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import json
import yaml
from easydict import EasyDict as edict

from gran.granmodel import *
from gran.utils.train_helper import load_model


def get_config(config_file, exp_dir=None):
  config = edict(yaml.load(open(config_file, 'r'), Loader=yaml.SafeLoader))
  return config


def get_graph(adj, undirected):
  """ get a networkx graph from zero-padded adj """
  if undirected:
    # remove all zeros rows and columns
    adj = adj[~np.all(adj == 0, axis=1)]
    adj = adj[:, ~np.all(adj == 0, axis=0)]
    adj = np.asmatrix(adj)
    G = nx.from_numpy_matrix(adj)
  else:
    G = nx.from_numpy_matrix(adj, create_using=nx.DiGraph)
  return G


def graph_sample(model_dir, model_name, config_file, 
                 num_gen=1, batch_size=1, return_as_nx=True):
  
  config_path = os.path.join(model_dir, config_file)
  model_path = os.path.join(model_dir, model_name)
  
  config = edict(yaml.load(open(config_path, 'r'), Loader=yaml.SafeLoader))
  granmodel = GRANMixtureBernoulli(config)

  load_model(granmodel, model_path, device='cuda:0')

  if config.use_gpu:
    granmodel = nn.DataParallel(granmodel, device_ids=config.gpus).to(config.device)

  granmodel.eval()

  ### Generate Graphs
  A_pred = []
  num_nodes_pred = []
  num_batch = int(np.ceil(num_gen / batch_size))
  num_nodes_pmf = np.array(json.loads(config.num_nodes_pmf))

  gen_run_time = []
  for ii in range(num_batch):
    with torch.no_grad():        
      start_time = time.time()
      input_dict = {}
      input_dict['is_sampling'] = True
      input_dict['batch_size'] = batch_size
      input_dict['num_nodes_pmf'] = num_nodes_pmf
      A_tmp = granmodel(input_dict)
      gen_run_time += [time.time() - start_time]
      A_pred += [aa.data.cpu().numpy() for aa in A_tmp]
      num_nodes_pred += [aa.shape[0] for aa in A_tmp]

  if return_as_nx:
    undirected = config.model.is_sym
    graphs_gen = [get_graph(aa, undirected=undirected) for aa in A_pred]
  else:
    graphs_gen = A_pred
    
  return graphs_gen










