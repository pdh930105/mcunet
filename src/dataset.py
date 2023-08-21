# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import torchvision
import torchvision.transforms as transforms
import pathlib
logger = logging.getLogger(__name__)


def get_loader(args, img_resize):
    if args.db.name.lower() == 'cifar10':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.Resize(img_resize),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])

        transform_test = transforms.Compose([
            transforms.Resize(img_resize),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
        trainset = torchvision.datasets.CIFAR10(
            root=args.db.root, train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR10(
            root=args.db.root, train=False, download=True, transform=transform_test)
        num_classes = 10
    elif args.db.name.lower() == 'cifar100':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.Resize(img_resize),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2671)),
            ])

        transform_test = transforms.Compose([
            transforms.Resize(img_resize),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2671)),
            ])
        trainset = torchvision.datasets.CIFAR100(
            root=args.db.root, train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR100(
            root=args.db.root, train=False, download=True, transform=transform_test)
        num_classes = 100
    elif args.db.name.lower() == 'imagenet':
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.Resize(img_resize),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        transform_test = transforms.Compose([
            transforms.CenterCrop(224),
            transforms.Resize(img_resize),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        train_path = pathlib.Path(args.db.root) / 'train'
        test_path = pathlib.Path(args.db.root) / 'val'
        num_classes = 1000
        trainset = torchvision.datasets.ImageFolder(root=train_path, transform=transform_train)
        testset = torchvision.datasets.ImageFolder(root=test_path, transform=transform_test)
    else:
        logger.log("Unknown dataset: {}".format(args.db.name))    
        assert False

    return trainset, testset, num_classes
