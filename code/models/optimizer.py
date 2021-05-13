"""Optimizer."""

import math
import torch

import utils.lr_policy as lr_policy


def construct_optimizer(model, cfg):
    """
    Construct the optimizer.

    Args:
        model (model): model to perform optimization.
        cfg (config): configs of hyper-parameters of the optimizer, including
        base learning rate, weight_decay and etc.
    """
    # Finetune batchnorm parameters.
    finetune_bn_params = []
    # Finetune parameters.
    finetune_rest_params = []
    # Task batchnorm parameters
    bn_params = []
    # Task parameter
    rest_params = []
    if cfg.MODEL.USE_TRANSFORMER:
        for name, p in model.named_parameters():
            if 'bn' in name and 'head' not in name:
                finetune_bn_params.append(p)
            elif 'bn' in name:
                bn_params.append(p)
            elif 'avbert' in name and 'head' not in name:
                finetune_rest_params.append(p)
            else:
                rest_params.append(p)
    else:
        for name, p in model.named_parameters():
            if 'bn' in name and 'head' not in name:
                finetune_bn_params.append(p)
            elif 'bn' in name:
                bn_params.append(p)
            elif 'conv' in name and 'head' not in name:
                finetune_rest_params.append(p)
            else:
                rest_params.append(p)

    optim_params = []
    if cfg.SOLVER.PROTOCOL == "finetune":
        if finetune_bn_params:
            optim_params.append(
                {
                    "name": "finetune_bn",
                    "params": finetune_bn_params,
                    "lr": cfg.SOLVER.FINETUNE_LR,
                    "weight_decay": cfg.BN.WEIGHT_DECAY,
                }
            )
        if finetune_rest_params:
            optim_params.append(
                {
                    "name": "finetune_rest",
                    "params": finetune_rest_params,
                    "lr": cfg.SOLVER.FINETUNE_LR,
                    "weight_decay": cfg.SOLVER.FINETUNE_WEIGHT_DECAY,
                }
            )
    if bn_params:
        optim_params.append(
            {
                "name": "task_bn",
                "params": bn_params,
                "lr": cfg.SOLVER.BASE_LR,
                "weight_decay": cfg.BN.WEIGHT_DECAY,
            }
        )
    if rest_params:
        optim_params.append(
            {
                "name": "task_rest",
                "params": rest_params,
                "lr": cfg.SOLVER.BASE_LR,
                "weight_decay": cfg.SOLVER.WEIGHT_DECAY,
            }
        )
    # Check all parameters will be passed into optimizer.
    assert len(list(model.parameters())) == len(rest_params) + len(bn_params) + len(finetune_rest_params) + len(
        finetune_bn_params
    ), "parameter size does not match: {} + {} + {} + {} != {}".format(
        len(rest_params), len(bn_params), len(finetune_rest_params), len(finetune_bn_params), len(list(model.parameters()))
    )

    if cfg.SOLVER.OPTIMIZING_METHOD == "sgd":
        optimizer = torch.optim.SGD(
            optim_params,
            momentum=cfg.SOLVER.MOMENTUM,
            dampening=cfg.SOLVER.DAMPENING,
            nesterov=cfg.SOLVER.NESTEROV,
        )
    elif cfg.SOLVER.OPTIMIZING_METHOD == "adam":
        optimizer = torch.optim.Adam(
            optim_params,
            betas=(0.9, 0.999),
            eps=1e-6,
        )
    elif cfg.SOLVER.OPTIMIZING_METHOD == "adamw":
        optimizer = torch.optim.AdamW(
            optim_params,
            betas=(0.9, 0.999),
            eps=1e-6,
            amsgrad=cfg.SOLVER.USE_AMSGRAD,
        )
    else:
        raise NotImplementedError(
            "Does not support {} optimizer".format(cfg.SOLVER.OPTIMIZING_METHOD)
        )
    return optimizer


def get_lr(
    policy,
    base_lr,
    warmup_start_lr,
    global_step,
    num_optimizer_steps,
    num_warmup_steps,
):
    return lr_policy.get_lr(
        policy,
        base_lr,
        warmup_start_lr,
        global_step,
        num_optimizer_steps,
        num_warmup_steps,
    )


def set_lr(optimizer, new_lr, name=None):
    """
    Sets the optimizer lr to the specified value.
    Args:
        optimizer (optim): the optimizer using to optimize the current network.
        new_lr (float): the new learning rate to set.
        name (str): parameter group name.
    """
    if name is not None:
        for param_group in optimizer.param_groups:
            if name in param_group["name"]:
                param_group["lr"] = new_lr
    else:
        for param_group in optimizer.param_groups:
            param_group["lr"] = new_lr
