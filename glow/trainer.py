import re
import os
import torch
import torch.nn.functional as F

import datetime
import numpy as np
from tqdm import tqdm
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from .utils import save, load, plot_prob, new2old
from .config import JsonConfig
from .models import Glow
from . import thops
from .nll_computation import nll_computer

class Trainer(object):
    def __init__(self, graph, graph_prior, optim, lrschedule, loaded_step,
                 devices, data_device,
                 dataset, hparams):
        if isinstance(hparams, str):
            hparams = JsonConfig(hparams)
        self.hparams = hparams
            # set members
            # append date info
        date = str(datetime.datetime.now())
        date = date[:date.rfind(":")].replace("-", "")\
                                     .replace(":", "")\
                                     .replace(" ", "_")
        self.log_dir = os.path.join(hparams.Dir.log_root, "log")
        
        self.checkpoints_dir = os.path.join(self.log_dir, "checkpoints")
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
            # write hparams
        hparams.dump(self.log_dir)
        if not os.path.exists(self.checkpoints_dir):
            os.makedirs(self.checkpoints_dir)

        self.checkpoints_gap = hparams.Train.checkpoints_gap
            
        self.em_gap = hparams.Train.em_gap
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
        self.n_epoches = hparams.Train.n_epoches

        self.global_step = 0
        # lr schedule
        self.lrschedule = lrschedule
        self.loaded_step = loaded_step
        # data relative
        self.y_classes = hparams.Glow.y_classes
        self.y_condition = hparams.Glow.y_condition
        self.y_criterion = hparams.Criterion.y_condition
        assert self.y_criterion in ["multi-classes", "single-class"]

        # log relative
        # tensorboard
        self.writer = SummaryWriter(log_dir=self.log_dir)
        self.scalar_log_gaps = hparams.Train.scalar_log_gap
        self.plot_gaps = hparams.Train.plot_gap
        self.inference_gap = hparams.Train.inference_gap
        self.warm_start = True if len(hparams.Train.warm_start)>0 else False
        # mixture setting
        self.naive = hparams.Mixture.naive
        # set latent posterior
        self.num_component = hparams.Mixture.num_component
        self.graph_prior = torch.ones(hparams.Mixture.num_component)/hparams.Mixture.num_component if not self.warm_start else self.graph["new"].get_prior()
        self.model_prior = torch.FloatTensor(hparams.Mixture.num_component).to(self.data_device)
        # source mixture setting
        self.regulate_std = hparams.Mixture.regulate_std
        self.regulate_mulI = hparams.Mixture.regulate_mulI
        #self.regulator_std = hparams.Mixture.regulator_std
 
        self.current_nll = np.inf

    def _posterior_computation(self, x, y_onehot):
        
        if self.naive:
            with torch.no_grad():
                z, nll = self.graph["old"](x=x, y_onehot=y_onehot)

                nlog_joint_prob =nll - torch.log(self.graph_prior.unsqueeze(1).expand_as(nll)+1e-6).to(self.data_device)
                tmp_sum = torch.log( torch.sum( torch.exp(-nlog_joint_prob), dim=[0]) ).to(self.data_device)                
                nlog_gamma = nlog_joint_prob + tmp_sum.expand_as(nlog_joint_prob)

        else:
            with torch.no_grad():
                z, gaussian_nlogp, nlogdet, reg_prior_logp = self.graph["old"](x=x, y_onehot=y_onehot,regulate_std=self.regulate_std)
                gaussian_nlogp = gaussian_nlogp * thops.all_pixels(x)
                nlogdet = nlogdet * thops.all_pixels(x)

                nlog_joint_prob = gaussian_nlogp - torch.log(self.graph_prior.unsqueeze(1)+1e-8).to(self.data_device)

                min_nlog_joint_prob, _ = nlog_joint_prob.min(dim=0)
                delta_nlog_joint_prob = nlog_joint_prob - min_nlog_joint_prob

                tmp_sum = torch.log( torch.sum( torch.exp(-delta_nlog_joint_prob), dim=[0]) ).to(self.data_device)                
                nlog_gamma = delta_nlog_joint_prob + tmp_sum.expand_as(delta_nlog_joint_prob)
        return nlog_gamma

    def train(self):
        # set old and new to have the same parameters
        def myRange(start,end,step):
            i = start
            while i < end:
                yield i
                i += step
            
            yield end

        self.graph["new"].update_prior(self.graph_prior)
        self.graph["old"].update_prior(self.graph_prior)
        start_epoch = self.loaded_step
        
        # set to training state
        
        for key, graph in self.graph.items():
            graph.train()

        # begin to train

        for epoch in myRange(start_epoch, start_epoch + self.n_epoches, 1):    
            try:
                print("{} at epoch: {}, loss {}, prior {}, prior_in_graph {}".format(os.path.basename(self.hparams.Dir.log_root), epoch, loss.data, self.graph_prior, self.graph["new"].get_prior()))
            except NameError:
                print("epoch {}, loss {}, prior {}".format(epoch, 0, self.graph_prior))
                
            progress = tqdm(self.data_loader)
            
            for i_batch, batch in enumerate(progress):
                # update learning rate
                lr = self.lrschedule["func"](global_step=self.global_step,
                                             **self.lrschedule["args"])
                
                for param_group in self.optim.param_groups:
                    param_group['lr'] = lr
                    
                self.optim.zero_grad()
                if self.global_step % self.scalar_log_gaps == 0:
                    self.writer.add_scalar("lr/lr", lr, self.global_step)
                    # get batch data
                batch = {"x": batch[0], "y":batch[1]}
                for k in batch:
                    batch[k] = batch[k].to(self.data_device)
                    
                x = batch["x"]
                y = None
                y_onehot = None
                if self.y_condition:# not entered at this stage
                    if self.y_criterion == "multi-classes":
                        assert "y_onehot" in batch, "multi-classes ask for `y_onehot` (torch.FloatTensor onehot)"
                        y_onehot = batch["y_onehot"]
                    elif self.y_criterion == "single-class":
                        assert "y" in batch, "single-class ask for `y` (torch.LongTensor indexes)"
                        y = batch["y"]
                        y_onehot = thops.onehot(y, num_classes=self.y_classes)
                        
                # at first time, initialize ActNorm
                if self.global_step == 0 and self.warm_start is False:

                    self.graph["new"](x[:self.batch_size // len(self.devices), ...],
                                      y_onehot[:self.batch_size // len(self.devices), ...] if y_onehot is not None else None)
                    self.graph["old"](x[:self.batch_size // len(self.devices), ...],
                                      y_onehot[:self.batch_size // len(self.devices), ...] if y_onehot is not None else None)
                    
                    new2old(global_step=epoch,
                            path=self.log_dir,
                            graph=self.graph,
                            optim=self.optim,
                            max_checkpoints=self.max_checkpoints,
                            is_best=False)
                    self.graph["old"].eval()
        
                    
                
                # forward phase and loss calculate
                #assert self.graph_prior.sum()==1, ("prior should sum to 1")
                if self.naive:
                    nlog_gamma = self._posterior_computation(x=x, y_onehot=y_onehot)
                    z, nll = self.graph["new"](x=x, y_onehot=y_onehot)
                    nll = nll * thops.all_pixels(x)
                    nlog_joint_prob =nll - torch.log(self.graph_prior.unsqueeze(1).expand_as(nll)+1e-6).to(self.data_device)

                    loss_generative = (torch.sum( torch.exp(-nlog_gamma) * nlog_joint_prob) /self.batch_size) / thops.all_pixels(x)
                    loss = loss_generative

                else:
                    ## posterior computation
                    nlog_gamma = self._posterior_computation(x=x, y_onehot=y_onehot)
                    ##### likelihood computation
                    new_z, new_gaussian_nlogp, new_nlogdet, new_reg_prior_logp = self.graph["new"](x=x, y_onehot=y_onehot,regulate_std=self.regulate_std)
                    new_gaussian_nlogp = new_gaussian_nlogp * thops.all_pixels(x)
                    new_nlogdet = new_nlogdet * thops.all_pixels(x)

                    new_nlog_joint_prob = new_gaussian_nlogp - torch.log(self.graph_prior.unsqueeze(1)+1e-8).to(self.data_device)
                    
                    loss_generative = (torch.mean(torch.sum( torch.exp(-nlog_gamma) * new_nlog_joint_prob, dim=0)) + torch.mean(new_nlogdet))/thops.all_pixels(x)
                    ## Conditional entropy computation
                    if self.regulate_mulI>0:
                        # mutual information regulation to be considered in future
                        new_nlog_gamma = nlog_gamma
                        total_joint_nlog = new_gaussian_nlogp + new_nlogdet - torch.log(self.graph_prior.unsqueeze(1)+1e-8).to(self.data_device)
                        conditional_entropy = torch.sum((-nlog_gamma) * torch.exp(-total_joint_nlog/thops.all_pixels(x)) )
                        # conditional_entropy = torch.sum(fix_point_conditionalH * px)
                        if conditional_entropy == float('inf') or conditional_entropy == float('-inf'):
                            print("[Encounter inf entropy]...\n")

                        loss_generative = loss_generative + conditional_entropy*(self.regulate_mulI)/self.batch_size
                        
                    loss_std = 0
                    if self.regulate_std:
                        loss_std = -new_reg_prior_logp.mean()
                    
                    loss = loss_generative + loss_std

                if self.global_step % self.scalar_log_gaps == 0:
                    self.writer.add_scalar("loss/loss_generative", loss_generative, self.global_step)
                    if self.y_condition:
                        self.writer.add_scalar("loss/loss_classes", loss_classes, self.global_step)
                        
                
                # clear buffers
                self.graph["new"].zero_grad()
                self.optim.zero_grad()

                # backward
                loss.backward()
                
                # operate grad
                
                if self.max_grad_clip is not None and self.max_grad_clip > 0:
                    torch.nn.utils.clip_grad_value_(self.graph["new"].parameters(), self.max_grad_clip)
                if self.max_grad_norm is not None and self.max_grad_norm > 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.graph["new"].parameters(), self.max_grad_norm)
                    if self.global_step % self.scalar_log_gaps == 0:
                        self.writer.add_scalar("grad_norm/grad_norm", grad_norm, self.global_step)
                        
                
                # step
                self.optim.step()
                
                # global step
                self.global_step += 1
                # accumulate the posterior of model
                with torch.no_grad():
                    self.model_prior += torch.exp(-nlog_gamma.data).mean(dim=1)

                progress.set_description("{} at epoch {} on {}:".format(os.path.basename(self.hparams.Dir.log_root), epoch, self.data_device) + "loss: {0:.4f}".format(loss.cpu().data.numpy().round(4)))

            if epoch>0 and epoch%self.em_gap == 0:
                with torch.no_grad():
                    self.model_prior = self.model_prior/torch.sum(self.model_prior)
                    self.graph_prior = self.model_prior.cpu().data
                    self.graph["new"].update_prior(self.graph_prior)
                    print("[Trainer]: Update prior: {}]".format(self.graph_prior))
                    #### evaluate the current nll value
                    my_nll_computer = nll_computer(graph=self.graph["new"],
                                                   devices=self.devices,
                                                   data_device=self.data_device,
                                                   dataset=self.data_loader,
                                                   hparams=self.hparams)
                    my_new_solution_nll = my_nll_computer.compute()
                if my_new_solution_nll < self.current_nll and not np.isinf(my_new_solution_nll):
                    self.current_nll = my_new_solution_nll
                    is_best = True
                    print("[{}, {}, NLL computer: Best update.]".format(os.path.basename(self.hparams.Dir.log_root), epoch))

                else:
                    is_best = False
                    print("[{}, {}, NLL computer: No Best update.]".format(os.path.basename(self.hparams.Dir.log_root), epoch))
                
                self.writer.add_scalar("nll_value/step", my_new_solution_nll, epoch)
                
                new2old(global_step=epoch,
                        path=self.log_dir,
                        graph=self.graph,
                        optim=self.optim,
                        max_checkpoints=self.max_checkpoints,
                        is_best=is_best)

            self.model_prior.zero_()
            
        self.writer.export_scalars_to_json(os.path.join(self.log_dir, "all_scalars.json"))
        self.writer.close()
