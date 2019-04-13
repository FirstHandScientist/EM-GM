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
    def __init__(self, graph, graph_prior, optim, lrschedule, loaded_step,
                 devices, data_device,
                 dataset, hparams):
        if isinstance(hparams, str):
            hparams = JsonConfig(hparams)
        # set members
        # append date info
        date = str(datetime.datetime.now())
        date = date[:date.rfind(":")].replace("-", "")\
                                     .replace(":", "")\
                                     .replace(" ", "_")
        self.checkpoints_gap = hparams.Train.checkpoints_gap
        
        self.max_checkpoints = hparams.Train.max_checkpoints
        # set the mixture prior
        self.num_component = hparams.Mixture.num_component
        self.mix_prior = torch.ones(self.num_component)
        # model relative
        self.graph = graph

        self.optim = optim
        self.weight_y = hparams.Train.weight_y
        # grad operation
        self.max_grad_clip = hparams.Train.max_grad_clip
        self.max_grad_norm = hparams.Train.max_grad_norm
        # copy devices from built graph
        self.devices = devices
        self.data_device = data_device
        # number of training batches
        self.batch_size = hparams.Train.batch_size
        
        self.data_loader = DataLoader(dataset,
                                      batch_size=self.batch_size,
                                    #   num_workers=8,
                                      shuffle=True,
                                      drop_last=True)
        self.n_epoches = 1
        self.global_step = 0
        # lr schedule
        self.lrschedule = lrschedule
        self.loaded_step = loaded_step
        # data relative
        self.y_classes = hparams.Glow.y_classes
        self.y_condition = hparams.Glow.y_condition
        self.y_criterion = hparams.Criterion.y_condition
        assert self.y_criterion in ["multi-classes", "single-class"]

        # mixture setting
        self.naive = hparams.Mixture.naive
        # set latent posterior
        self.num_component = hparams.Mixture.num_component
        self.graph_prior = torch.ones(hparams.Mixture.num_component)/hparams.Mixture.num_component
        self.model_prior = torch.FloatTensor(hparams.Mixture.num_component).to(self.data_device)
        # source mixture setting
        self.regulate_std = hparams.Mixture.regulate_std
        #self.regulator_std = hparams.Mixture.regulator_std
        self.warm_start = True if len(hparams.Train.warm_start)>0 else False
        #self.nlog_joint_prob = []
        #self.nlog_joint_prob = torch.FloatTensor(hparams.Mixture.num_component,
        #                                hparams.Train.batch_size).to(self.data_device)
        
        #self.em_loss = em_loss(self.graph, self.nlog_gamma, self.nlog_joint_prob, self.batch_size)
        
    def compute(self):
        # set to training state
        if self.naive:
            for i in range(self.num_component):
                self.graph.get_component(i).eval()
        else:
            self.graph.get_component().eval()
        self.global_step = self.loaded_step
        # begin to train
        nll_result = []
        counter = 0
        for epoch in range(self.n_epoches):
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
                    #min_logp = logp.min(axis= 0)
                    min_logp = logp.mean(axis=0)
                    delta_logp = logp - min_logp
                    delta_logp = delta_logp.astype(np.float128)
                    summand = np.exp(delta_logp) * self.graph.get_prior().numpy()[:,None].astype(np.float128)
                    log_sum = np.log(np.sum( summand, axis=0) )
                    loss = np.mean(-log_sum - min_logp)/thops.all_pixels(x)
                    if np.isinf(loss):
                        print("[NLL]: encounter inf.")
                    #######################approximate compute #################
                    # logp = logp * thops.pixels(x)
                    # loss_vec = np.zeros(self.batch_size)
                    # loss= 0
                    # count= 0
                    # for i in range(self.batch_size):
                    #     temp_array = logp[:,i]
                    #     #################### kick out infinite numbers 
                    #     if True in np.isinf(temp_array):
                    #         continue
                    #     #################### for K = 1 case ############
                    #     if temp_array.shape[0]==1:
                    #         loss_vec[i] = -temp_array/thops.pixels(x)
                    #         continue

                    #     ##### for K > 1 cases####################
                        
                    #     delta_logp = temp_array- temp_array.min()
                    #     sorted_delta_logp = np.sort(delta_logp)
                    #     if sorted_delta_logp[1]>20 or sorted_delta_logp[-1]>40:
                    #         vec_p = np.log(self.graph.get_prior().numpy()[delta_logp.argmax()]) + delta_logp.max()
                    #     else:
                    #         vec_p = np.log(np.sum(self.graph.get_prior().numpy() * np.exp(delta_logp)))
                    #     loss_vec[i] =  - (temp_array.min()+ vec_p)/thops.pixels(x)
                    # count = 0
                    # for i in range(self.batch_size):
                    #     if loss_vec[i]-loss_vec.min()<1e6:
                    #         loss = loss + loss_vec[i]
                    #         count = count +1
                            
                    # try:
                    #     loss = loss/count
                    #     #loss = np.median(loss_vec)
                    # except:
                    #     print("[loss_vec:{}]".format(loss_vec))
                    #         # if count>0:
                    # #     loss = loss/count
                    # # else:
                    # #     pass
                    #  #################### sanity check #############
 
                    # if loss < 0:
                    #     print("logp: {}".format(logp))
                    #     print("loss_vec is: {}".format(loss_vec))
                    #     counter +=1
                                    
                tmp_nll_result += loss
                # global step
                self.global_step += 1
            #################### collections results ###############
            nll_result.append(tmp_nll_result/len(self.data_loader))
        print("Prior of this model is:{}".format(self.graph.get_prior()))
        return (nll_result, counter)
