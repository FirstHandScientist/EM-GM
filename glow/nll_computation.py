import re
import os
import torch
import torch.nn.functional as F


import datetime
import numpy as np
from tqdm import tqdm
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from .utils import save, load, plot_prob
from .config import JsonConfig
from .models import Glow
from . import thops

class nll_computer(object):
    def __init__(self, graph, devices, data_device,
                 dataset, hparams):
        
        hparams = hparams
        # set members
        # append date info
        self.num_component = hparams.Mixture.num_component
        
        # model relative
        self.graph = graph

                
        self.devices = devices
        self.data_device = data_device
        # number of training batches
        self.batch_size = hparams.Train.batch_size
        
        self.data_loader = dataset
        self.global_step = 0
                
        
        
        # mixture setting
        self.naive = hparams.Mixture.naive
        # set latent posterior
        self.num_component = hparams.Mixture.num_component
        
        self.model_prior = torch.FloatTensor(hparams.Mixture.num_component).to(self.data_device)
        # source mixture setting
        self.regulate_std = hparams.Mixture.regulate_std
        #self.regulator_std = hparams.Mixture.regulator_std
        self.warm_start = True if len(hparams.Train.warm_start)>0 else False
        #self.nlog_joint_prob = []
        
        
    def compute(self):
        # set to training state
        if self.naive:
            for i in range(self.num_component):
                self.graph.get_component(i).eval()
        else:
            self.graph.get_component().eval()
        
        # begin to train
        nll_result = []
        counter = 0
        for epoch in range(1):
            progress = tqdm(self.data_loader)
            # negative log likelyhood
            tmp_nll_result = 0

            for i_batch, batch in enumerate(progress):
                batch = {"x": batch[0], "y":batch[1]}
                for k in batch:
                    batch[k] = batch[k].to(self.data_device)
                
                x = batch["x"] 
                y = None
                y_onehot = None
                
                with torch.no_grad():
                    if self.naive:
                        z, nll = self.graph(x=x, y_onehot=y_onehot)
                        # logp = -nll * thops.pixels(x)
                        logp = -nll.cpu().numpy()
                    else:
                        z, gaussian_nlogp, nlogdet, reg_prior_logp = self.graph(x=x, y_onehot=y_onehot,regulate_std=self.regulate_std)
                        #testing reverse
                        logp = -(gaussian_nlogp + nlogdet) 
                        logp = logp.cpu().numpy()
                    #######################nats/pixels#################
                    # real_p = np.exp(logp) * self.graph.get_prior().numpy()[:, np.newaxis]
                    # tmp_sum = np.sum(real_p, axis=0)
                    # loss = np.mean( - np.log(tmp_sum + 1e-6) )
                    #######################exactly compute#################
                    logp = logp * thops.all_pixels(x)
                    min_logp = logp.mean(axis= 0)
                    delta_logp = logp - min_logp
                    delta_logp = delta_logp.astype(np.float128)
                    summand = np.exp(delta_logp) * self.graph.get_prior().numpy()[:,None].astype(np.float128)
                    log_sum = np.log(np.sum( summand, axis=0) )
                    loss = np.mean(-log_sum - min_logp)/thops.all_pixels(x)
                                    
                tmp_nll_result += loss
                # global step
                self.global_step += 1
            #################### collections results ###############
            nll_result.append(tmp_nll_result/len(self.data_loader))
        print("[NLL computer: NLL evaluation: {}]".format(nll_result[0]))
        return nll_result[0]
