"""evaluate nll script.

Usage:
    nll_evaluate.py <hparams> <dataset> <dataset_root>
"""
import os
import vision
import torchvision.datasets as dset
from docopt import docopt
from torchvision import transforms
from glow.builder import build
from glow.toy_builder import toy_build
from glow.trainer import Trainer

from glow.config import JsonConfig
from models.toy_data import inf_train_gen

#import ipython.core.debuger.set_tracer as set_tracer

if __name__ == "__main__":
    args = docopt(__doc__)
    hparams = args["<hparams>"]
    dataset = args["<dataset>"]

    # assert dataset in vision.Datasets, (
    #     "`{}` is not supported, use `{}`".format(dataset, vision.Datasets.keys()))
    assert os.path.exists(hparams), (
        "Failed to find hparams josn `{}`".format(hparams))

    dataset_root = args["<dataset_root>"]
    assert os.path.exists(dataset_root), (
        "Failed to find root dir `{}` of dataset.".format(dataset_root))
    hparams = JsonConfig(hparams)

    # set transform of dataset remove normalize 2018-12-26
    if dataset == "mnist":
        transform = transforms.Compose([
            transforms.CenterCrop(hparams.Data.center_crop),
            transforms.Resize(hparams.Data.resize),
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
        ])
        dataset = dset.MNIST(root=dataset_root,
                         download=True,
                         transform=transform)
    elif dataset == "fashion-mnist":
        transform = transforms.Compose([
            transforms.Resize(hparams.Data.resize),
            transforms.ToTensor(),transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
        dataset = dset.FashionMNIST(root=dataset_root,
                         download=True,
                         transform=transform)
    # build graph and dataset
    built = build(hparams, True)
    trainer = Trainer(**built, dataset=dataset, hparams=hparams)
    trainer.train()
