"""Train script.

Usage:
    train.py <hparams>
"""
import os
from itertools import cycle
import torch
import torchvision.datasets as dset
import torchvision.utils as vutils
import datetime
from docopt import docopt
from torchvision import transforms
from glow.builder import build
from glow.trainer import Trainer
from glow.config import JsonConfig
from joblib import Parallel, delayed


if __name__ == "__main__":
    args = docopt(__doc__)
    hparams_dir = args["<hparams>"]
    

    assert os.path.exists(hparams_dir), ("Failed to find hparams josn `{}`".format(hparams_dir))

    hparams = JsonConfig(hparams_dir)

    dataset_root = hparams.Data.dataset_root

    dataset = hparams.Data.dataset
    assert os.path.exists(dataset_root), (
        "Failed to find root dir `{}` of dataset.".format(dataset_root))
    
    if dataset == "mnist":
        transform = transforms.Compose([
            transforms.Resize(hparams.Data.resize),
            transforms.ToTensor(),
            transforms.Normalize((0.5,),(1,))
        ])
        dataset_ins = dset.MNIST(root=dataset_root, train=False,
                         download=True,
                         transform=transform)
    elif dataset == "fashion-mnist":
        transform = transforms.Compose([
            transforms.Resize(hparams.Data.resize),
            transforms.ToTensor(),transforms.Normalize((0.5,),(1,))])
        dataset_ins = dset.FashionMNIST(root=dataset_root, train=False,
                         download=True,
                         transform=transform)
    # build graph and dataset
    
    # set the available devices
    dic_cuda = ["cuda:0", "cuda:1"]
    device_iter = iter(cycle(dic_cuda))
            
    def worker(mix_k):
        local_hparams = JsonConfig(hparams_dir)
        local_hparams.Mixture.num_component = mix_k
        model = "GenMM-K{}" if local_hparams.Mixture.naive else "LatMM-K{}"
        local_hparams.Dir.log_root = os.path.join(local_hparams.Dir.log_root,
                                                  model.format(mix_k))
        this_device = next(device_iter)
        local_hparams.Device.glow[0] = this_device
        local_hparams.Device.data = this_device
        print("Dir: {} and device: {}".format(local_hparams.Dir.log_root,this_device))
        peeked = False
        if not peeked:
            tmp_dataloader = torch.utils.data.DataLoader(dataset_ins, batch_size=64,
                                         shuffle=True, num_workers=int(2))
            img = next(iter(tmp_dataloader))[0]
            
            if not os.path.exists(local_hparams.Dir.log_root):
                os.makedirs(local_hparams.Dir.log_root)
            # peek the training data set
            vutils.save_image(img.add(0.5), os.path.join(local_hparams.Dir.log_root, "img_under_evaluation.png"))
            peeked = True
        
        built = build(local_hparams, True)
        trainer = Trainer(**built, dataset=dataset_ins, hparams=local_hparams)
        trainer.train()

    Parallel(n_jobs=2, pre_dispatch="all", backend="threading")(map(delayed(worker), [1,3,5,7]))
