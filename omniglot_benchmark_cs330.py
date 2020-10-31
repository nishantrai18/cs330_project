#!/usr/bin/env python3

import random
import learn2learn as l2l

from torchvision import transforms
from PIL.Image import LANCZOS
from full_omniglot import FullOmniglot

def omniglot_tasksets(
    train_ways,
    train_samples,
    test_ways,
    test_samples,
    root,
    **kwargs
):

    print("Loading Omniglot taskset")
    """
    Benchmark definition for Omniglot.
    """
    data_transforms = transforms.Compose([
        transforms.Resize(28, interpolation=LANCZOS),
        transforms.ToTensor(),
        lambda x: 1.0 - x,
    ])
    
    # this gives the omniglot datasets
    #omniglot = l2l.vision.datasets.FullOmniglot(
    #    root=root,
    #    transform=data_transforms,
    #    download=True,
    #)
    
    omniglot = FullOmniglot(
        root=root,
        transform=data_transforms,
        download=True,
    )
    
    dataset = l2l.data.MetaDataset(omniglot)
    train_dataset = dataset # redo the labels in the train set here. We can append an extra column to train_dataset with the weak or not
    validation_datatset = dataset
    test_dataset = dataset
    
    
    classes = list(range(1623))
    random.shuffle(classes)
    train_transforms = [
        l2l.data.transforms.FusedNWaysKShots(dataset,
                                             n=train_ways,
                                             k=train_samples,
                                             filter_labels=classes[:1100]),
        l2l.data.transforms.LoadData(dataset),
        l2l.data.transforms.RemapLabels(dataset),
        l2l.data.transforms.ConsecutiveLabels(dataset),
        l2l.vision.transforms.RandomClassRotation(dataset, [0.0, 90.0, 180.0, 270.0])
    ]
    validation_transforms = [
        l2l.data.transforms.FusedNWaysKShots(dataset,
                                             n=test_ways,
                                             k=test_samples,
                                             filter_labels=classes[1100:1200]),
        l2l.data.transforms.LoadData(dataset),
        l2l.data.transforms.RemapLabels(dataset),
        l2l.data.transforms.ConsecutiveLabels(dataset),
        l2l.vision.transforms.RandomClassRotation(dataset, [0.0, 90.0, 180.0, 270.0])
    ]
    test_transforms = [
        l2l.data.transforms.FusedNWaysKShots(dataset,
                                             n=test_ways,
                                             k=test_samples,
                                             filter_labels=classes[1200:]),
        l2l.data.transforms.LoadData(dataset),
        l2l.data.transforms.RemapLabels(dataset),
        l2l.data.transforms.ConsecutiveLabels(dataset),
        l2l.vision.transforms.RandomClassRotation(dataset, [0.0, 90.0, 180.0, 270.0])
    ]

    _datasets = (train_dataset, validation_datatset, test_dataset)
    _transforms = (train_transforms, validation_transforms, test_transforms)
    return _datasets, _transforms