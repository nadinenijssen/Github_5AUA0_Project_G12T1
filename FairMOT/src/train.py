from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import os
import json

import torch
import torch.utils.data
from torchvision.transforms import transforms as T

from opts import opts
from models.model import create_model, load_model, save_model
from models.data_parallel import DataParallel
from logger import Logger
from datasets.dataset.jde import JointDataset
from trains.mot import MotTrainer


# our custom modifications to pose_hrnet and MOTtrainer
from models.networks.pose_hrnet_ours import get_pose_net as get_pose_net_hrnet_ours
from models.networks.pose_hrnet_ours import freeze, freeze_module, print_layers_with_gradients
from trains.mot_softtriple import MotTrainer as MotTrainer_softtriple
from trains.mot_ours import MotTrainer as MotTrainer_ours

def main(opt):
    torch.manual_seed(opt.seed)
    torch.backends.cudnn.benchmark = not opt.not_cuda_benchmark and not opt.test

    print("Setting up data...")
    trainset_paths = opt.train_data  #{"mot17": "./data/mot17.training"}
    dataset_root = opt.data_dir
    transforms = T.Compose([T.ToTensor()])
    dataset = JointDataset(
        opt, dataset_root, trainset_paths, (1088, 608), augment=True, transforms=transforms
    )
    opt = opts().update_dataset_info_and_set_heads(opt, dataset)
    print(opt)

    logger = Logger(opt)

    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpus_str
    opt.device = torch.device("cuda" if opt.gpus[0] >= 0 else "cpu")

    print("Creating model...")
    # model = create_model(opt.arch, opt.heads, opt.head_conv)
    model = get_pose_net_hrnet_ours(num_layers=18, heads=opt.heads, head_conv=opt.head_conv)


    start_epoch = 0
    optimizer = torch.optim.Adam(model.parameters(), opt.lr)
    if opt.load_model != "":
        model, optimizer, start_epoch = load_model(
            model, opt.load_model, optimizer, opt.resume, opt.lr, opt.lr_step
        )

    if opt.freeze:
        freeze(model)
    # Get dataloader
    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    print("Starting training...")
    #trainer = MotTrainer(opt, model, optimizer)
    # trainer = MotTrainer_softtriple(opt, model, optimizer)
    trainer = MotTrainer_ours(opt, model, optimizer)
    trainer.set_device(opt.gpus, opt.chunk_sizes, opt.device)

    for epoch in range(start_epoch + 1, opt.num_epochs + 1):
        mark = epoch if opt.save_all else "last"
        # print_layers_with_gradients(model)
        log_dict_train, _ = trainer.train(epoch, train_loader)
        logger.write("epoch: {} |".format(epoch))
        for k, v in log_dict_train.items():
            logger.scalar_summary("train_{}".format(k), v, epoch)
            logger.write("{} {:8f} | ".format(k, v))

        if opt.val_intervals > 0 and epoch % opt.val_intervals == 0:
            save_model(
                os.path.join(opt.save_dir, "model_{}.pth".format(mark)), epoch, model, optimizer
            )
        else:
            save_model(os.path.join(opt.save_dir, "model_last.pth"), epoch, model, optimizer)
        logger.write("\n")
        if epoch in opt.lr_step:
            save_model(
                os.path.join(opt.save_dir, "model_{}.pth".format(epoch)), epoch, model, optimizer
            )
            lr = opt.lr * (0.1 ** (opt.lr_step.index(epoch) + 1))
            print("Drop LR to", lr)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr
        if epoch % 5 == 0:
            save_model(
                os.path.join(opt.save_dir, "model_{}.pth".format(epoch)), epoch, model, optimizer
            )
    logger.close()


if __name__ == "__main__":
    opt = opts().parse()
    print("gpus", ",".join(map(str, opt.gpus)))
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, opt.gpus))

    main(opt)
