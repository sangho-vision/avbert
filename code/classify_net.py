import os
import math
import random
import pprint
from datetime import datetime

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

import models.optimizer as optim
import models.losses as losses
import utils.distributed as du
import utils.checkpoint as cu
import utils.logging as logging
import utils.metrics as metrics
import utils.misc as misc
from data import loader
from models import build_model, ClassifyHead
from utils.meters import ClassifyTrainMeter, ClassifyValMeter, ClassifyTestMeter


logger = logging.get_logger(__name__)


def to_cuda(inputs):
    assert len(inputs) == 3 or len(inputs) == 4, \
        "inputs must be of length 3 or 4 (i.e., clips, labels, index)"
    if len(inputs) == 3:
        clip, labels, index = inputs
        if isinstance(clip, (list, )):
            for i in range(len(clip)):
                clip[i] = clip[i].cuda(non_blocking=True)
        else:
            clip = clip.cuda(non_blocking=True)
        labels = labels.cuda()
        index = index.cuda()
        return (clip,), labels, index
    else:
        visual_clip, audio_clip, labels, index = inputs
        for i in range(len(visual_clip)):
            visual_clip[i] = visual_clip[i].cuda(non_blocking=True)
        audio_clip = audio_clip.cuda(non_blocking=True)
        labels = labels.cuda()
        index = index.cuda()
        return (visual_clip, audio_clip), labels, index


def classify(cfg):
    if cfg.TRAIN.ENABLE:
        train(cfg)
    if cfg.TEST.ENABLE:
        test(cfg)


def train(cfg):
    # Set up  environment.
    du.init_distributed_training(cfg)
    if cfg.RNG_SEED != -1:
        random.seed(cfg.RNG_SEED)
        np.random.seed(cfg.RNG_SEED)
        torch.manual_seed(cfg.RNG_SEED)
        torch.cuda.manual_seed(cfg.RNG_SEED)
        torch.cuda.manual_seed_all(cfg.RNG_SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True

    # Setup logging format.
    timestamp = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    logging.setup_logging(os.path.join(cfg.LOG_DIR, f"log-{timestamp}.txt"))

    # Print config.
    logger.info("Classification Task Train with config:")
    logger.info(pprint.pformat(cfg))

    # Classification model and optimizer.
    model = build_model(cfg)
    if cfg.SOLVER.PROTOCOL == "linear_eval":
        head = ClassifyHead(cfg)
        cur_device = torch.cuda.current_device()
        head = head.cuda(device=cur_device)
        if cfg.NUM_GPUS > 1:
            head = torch.nn.parallel.DistributedDataParallel(
                module=head,
                device_ids=[cur_device],
                output_device=cur_device,
                broadcast_buffers=False,
            )
        optimizer = optim.construct_optimizer(head, cfg)
    else:
        head = None
        optimizer = optim.construct_optimizer(model, cfg)
    logger.info(f"Train Protocol: {cfg.SOLVER.PROTOCOL}" )

    # Create dataloaders.
    train_loader = loader.construct_loader(cfg, 'train')

    num_batches_per_epoch = len(train_loader)
    # Priority : MAX_EPOCH > NUM_STEPS
    # Priority : WARMUP_STEPS > WARMUP_EPOCHS > WARMUP_PROPORTION
    if cfg.SOLVER.MAX_EPOCH != -1:
        num_optimizer_epochs = cfg.SOLVER.MAX_EPOCH
        num_optimizer_steps = (
            num_optimizer_epochs * num_batches_per_epoch
        )
        if cfg.SOLVER.WARMUP_STEPS != -1:
            num_warmup_steps = cfg.SOLVER.WARMUP_STEPS
            num_warmup_epochs = num_warmup_steps / num_batches_per_epoch
        elif cfg.SOLVER.WARMUP_EPOCHS != -1:
            num_warmup_epochs = cfg.SOLVER.WARMUP_EPOCHS
            num_warmup_steps = (
                num_warmup_epochs * num_batches_per_epoch
            )
        else:
            num_warmup_steps = (
                num_optimizer_steps * cfg.SOLVER.WARMUP_PROPORTION
            )
            num_warmup_epochs = num_warmup_steps / num_batches_per_epoch
        num_epochs = cfg.SOLVER.MAX_EPOCH
        num_steps = num_epochs * num_batches_per_epoch
    else:
        num_optimizer_steps = cfg.SOLVER.NUM_STEPS
        num_optimizer_epochs = num_optimizer_steps / num_batches_per_epoch
        if cfg.SOLVER.WARMUP_STEPS != -1:
            num_warmup_steps = cfg.SOLVER.WARMUP_STEPS
            num_warmup_epochs = num_warmup_steps / num_batches_per_epoch
        elif cfg.SOLVER.WARMUP_EPOCHS != -1:
            num_warmup_epochs = cfg.SOLVER.WARMUP_EPOCHS
            num_warmup_steps = (
                num_warmup_epochs * num_batches_per_epoch
            )
        else:
            num_warmup_steps = (
                num_optimizer_steps * cfg.SOLVER.WARMUP_PROPORTION
            )
            num_warmup_epochs = num_warmup_steps / num_batches_per_epoch
        num_steps = cfg.SOLVER.NUM_STEPS
        num_epochs = math.ceil(num_steps / num_batches_per_epoch)


    if cfg.VAL.ENABLE and cfg.TRAIN.EVAL_PERIOD < num_epochs:
        val_loader = loader.construct_loader(cfg, 'val')
    else:
        val_loader = None

    start_epoch = 0
    global_step = 0

    best_epoch, best_top1_acc, top5_acc = 0, 0.0, 0.0

    if cfg.TRAIN.PREEMPTIBLE:
        train_checkpoint_file_path = os.path.join(
            cfg.SAVE_DIR,
            "epoch_latest.pyth",
        )
    else:
        train_checkpoint_file_path = cfg.TRAIN.CHECKPOINT_FILE_PATH

    if os.path.isfile(train_checkpoint_file_path):
        logger.info(
            "=> loading checkpoint '{}'".format(
                train_checkpoint_file_path
            )
        )
        checkpoint = torch.load(
            train_checkpoint_file_path, map_location='cpu'
        )
        cu.load_checkpoint(
            model,
            checkpoint['model_state_dict'],
            cfg.NUM_GPUS > 1,
        )
        if cfg.SOLVER.PROTOCOL == "linear_eval":
            cu.load_checkpoint(
                head,
                checkpoint['head_state_dict'],
                cfg.NUM_GPUS > 1,
            )
        start_epoch = checkpoint['epoch']
        optimizer.load_state_dict(checkpoint['optimizer'])
        global_step = checkpoint['epoch'] * len(train_loader)
        logger.info(
            "=> loaded checkpoint '{}'".format(
                train_checkpoint_file_path,
            )
        )
        if 'best_epoch' in checkpoint:
            best_epoch = checkpoint['best_epoch']
        if 'best_top1_acc' in checkpoint:
            best_top1_acc = checkpoint['best_top1_acc']
        if 'top5_acc' in checkpoint:
            top5_acc = checkpoint['top5_acc']

    elif cfg.PRETRAIN.CHECKPOINT_FILE_PATH:
        if os.path.isfile(cfg.PRETRAIN.CHECKPOINT_FILE_PATH):
            logger.info(
                "=> loading pretrain checkpoint '{}'".format(
                    cfg.PRETRAIN.CHECKPOINT_FILE_PATH
                )
            )
            checkpoint = torch.load(
                cfg.PRETRAIN.CHECKPOINT_FILE_PATH, map_location='cpu'
            )
            cu.load_finetune_checkpoint(
                model,
                checkpoint['state_dict'],
                cfg.NUM_GPUS > 1,
                cfg.MODEL.USE_TRANSFORMER,
            )
            logger.info(
                "=> loaded pretrain checkpoint '{}'".format(
                    cfg.PRETRAIN.CHECKPOINT_FILE_PATH,
                )
            )
    else:
        logger.info("Training with random initialization.")

    writer = None
    if du.is_master_proc():
        writer = SummaryWriter(cfg.LOG_DIR)

    # Create meters
    train_meter = ClassifyTrainMeter(
        writer,
        len(train_loader),
        num_epochs,
        num_steps,
        cfg,
    )
    if cfg.VAL.ENABLE and cfg.TRAIN.EVAL_PERIOD < num_epochs:
        val_meter = ClassifyValMeter(
            writer,
            len(val_loader),
            num_epochs,
            cfg,
        )
    else:
        val_meter = None

    if cfg.TRAIN.TEST_PERIOD < num_epochs:
        test_loader = loader.construct_loader(cfg, "test")
        if cfg.MODEL.TASK in ["VisualClassify", "AudioClassify"]:
            dataset_size = len(test_loader.dataset) // cfg.TEST.NUM_SAMPLES
        elif cfg.MODEL.TASK in ["MultimodalSequenceClassify"]:
            dataset_size = len(test_loader.dataset) // cfg.TEST.NUM_SPATIAL_CROPS
        test_meter = ClassifyTestMeter(
            dataset_size,
            cfg.TEST.NUM_SAMPLES,
            cfg.MODEL.NUM_CLASSES,
            len(test_loader),
            cfg.DATA.ENSEMBLE_METHOD,
            cfg.LOG_PERIOD,
        )
    else:
        test_loader = None
        test_meter = None

    # Perform the training loop.
    logger.info("Start epoch: {}".format(start_epoch+1))

    best_epoch, best_top1_acc, top5_acc = 0, 0.0, 0.0

    for cur_epoch in range(start_epoch, num_epochs):
        is_best_epoch = False
        # Shuffle the dataset.
        loader.shuffle_dataset(train_loader, cur_epoch)

        global_step = train_epoch(
            train_loader,
            model,
            head,
            optimizer,
            train_meter,
            cur_epoch,
            global_step,
            num_steps,
            num_optimizer_steps,
            num_warmup_steps,
            cfg,
        )

        if misc.is_eval_epoch(cfg.TRAIN.EVAL_PERIOD, cur_epoch, num_epochs) and cfg.VAL.ENABLE:
            eval_epoch(
                val_loader,
                model,
                head,
                val_meter,
                cur_epoch,
                cfg,
            )

        if misc.is_eval_epoch(cfg.TRAIN.TEST_PERIOD, cur_epoch, num_epochs):
            stats = perform_test(
                test_loader,
                model,
                head,
                test_meter,
                cfg,
                cur_epoch,
                num_epochs,
                writer,
            )

            if best_top1_acc < float(stats["top1_acc"]):
                best_epoch = cur_epoch + 1
                best_top1_acc = float(stats["top1_acc"])
                top5_acc = float(stats["top5_acc"])
                is_best_epoch = True
            logger.info(
                "BEST: epoch: {}, best_top1_acc: {:.2f}, top5_acc: {:.2f}".format(
                    best_epoch, best_top1_acc, top5_acc
                )
            )

        model_sd = \
            model.module.state_dict() if cfg.NUM_GPUS > 1 else \
            model.state_dict()
        if cfg.SOLVER.PROTOCOL == "linear_eval":
            head_sd = \
                head.module.state_dict() if cfg.NUM_GPUS > 1 else \
                head.state_dict()
        else:
            head_sd = None

        ckpt = {
            'epoch': cur_epoch + 1,
            'model_state_dict': model_sd,
            'head_state_dict': head_sd,
            'optimizer': optimizer.state_dict(),
        }

        if cfg.TRAIN.TEST_PERIOD < num_epochs:
            ckpt['best_epoch'] = best_epoch
            ckpt['best_top1_acc'] = best_top1_acc
            ckpt['top5_acc'] = top5_acc

        if cfg.TRAIN.PREEMPTIBLE and du.get_rank() == 0:
            cu.save_checkpoint(
                ckpt, filename=os.path.join(cfg.SAVE_DIR, "epoch_latest.pyth")
            )

        if (cur_epoch + 1) % cfg.TRAIN.SAVE_EVERY_EPOCH == 0 and du.get_rank() == 0:
            cu.save_checkpoint(
                ckpt,
                filename=os.path.join(cfg.SAVE_DIR, f"epoch{cur_epoch+1}.pyth")
            )

        if is_best_epoch and du.get_rank() == 0:
            cu.save_checkpoint(
                ckpt,
                filename=os.path.join(cfg.SAVE_DIR, f"epoch_best.pyth")
            )

        if global_step == num_steps:
            break


def test(cfg):
    # Set up  environment.
    du.init_distributed_training(cfg)
    if cfg.RNG_SEED != -1:
        random.seed(cfg.RNG_SEED)
        np.random.seed(cfg.RNG_SEED)
        torch.manual_seed(cfg.RNG_SEED)
        torch.cuda.manual_seed(cfg.RNG_SEED)
        torch.cuda.manual_seed_all(cfg.RNG_SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True

    # Setup logging format.
    if not cfg.TRAIN.ENABLE:
        timestamp = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
        logging.setup_logging(os.path.join(cfg.LOG_DIR, f"test_log-{timestamp}.txt"))

    # Print config.
    logger.info("Classification Task Test with config:")
    logger.info(pprint.pformat(cfg))

    # Classification model and optimizer.
    model = build_model(cfg)
    if cfg.SOLVER.PROTOCOL == "linear_eval":
        head = ClassifyHead(cfg)
        cur_device = torch.cuda.current_device()
        head = head.cuda(device=cur_device)
        if cfg.NUM_GPUS > 1:
            head = torch.nn.parallel.DistributedDataParallel(
                module=head,
                device_ids=[cur_device],
                output_device=cur_device,
                broadcast_buffers=False,
            )
    else:
        head = None

    if cfg.TEST.CHECKPOINT_FILE_PATH:
        test_checkpoint_file_path = cfg.TEST.CHECKPOINT_FILE_PATH
    else:
        test_checkpoint_file_path = os.path.join(
            cfg.SAVE_DIR,
            "epoch_best.pyth",
        )

    if os.path.isfile(test_checkpoint_file_path):
        logger.info(
            "=> loading checkpoint '{}'".format(
                test_checkpoint_file_path
            )
        )
        checkpoint = torch.load(
            test_checkpoint_file_path, map_location='cpu'
        )
        cu.load_checkpoint(
            model,
            checkpoint['model_state_dict'],
            cfg.NUM_GPUS > 1,
        )
        if cfg.SOLVER.PROTOCOL == "linear_eval":
            cu.load_checkpoint(
                head,
                checkpoint['head_state_dict'],
                cfg.NUM_GPUS > 1,
            )
        logger.info(
            "=> loaded checkpoint '{}' (epoch {})".format(
                test_checkpoint_file_path,
                checkpoint['epoch']
            )
        )
    else:
        logger.info("Test with random initialization for debugging")

    # Create video testing loaders
    test_loader = loader.construct_loader(cfg, "test")
    logger.info("Testing model for {} iterations".format(len(test_loader)))

    # Create meters for testing.
    if cfg.MODEL.TASK in ["VisualClassify", "AudioClassify"]:
        dataset_size = len(test_loader.dataset) // cfg.TEST.NUM_SAMPLES
    elif cfg.MODEL.TASK in ["MultimodalSequenceClassify"]:
        dataset_size = len(test_loader.dataset) // cfg.TEST.NUM_SPATIAL_CROPS
    test_meter = ClassifyTestMeter(
        dataset_size,
        cfg.TEST.NUM_SAMPLES,
        cfg.MODEL.NUM_CLASSES,
        len(test_loader),
        cfg.DATA.ENSEMBLE_METHOD,
        cfg.LOG_PERIOD,
    )

    perform_test(test_loader, model, head, test_meter, cfg)


def train_epoch(
    train_loader,
    model,
    head,
    optimizer,
    train_meter,
    cur_epoch,
    global_step,
    num_steps,
    num_optimizer_steps,
    num_warmup_steps,
    cfg,
):
    # Switch to train mode.
    if cfg.SOLVER.PROTOCOL == "linear_eval":
        model.eval()
        head.train()
    else:
        model.train()
    train_meter.iter_tic()

    for cur_step, inputs in enumerate(train_loader):
        global_step += 1
        clips, labels, index = to_cuda(inputs)
        clips_protocol = clips + (cfg.SOLVER.PROTOCOL, )
        if cfg.SOLVER.PROTOCOL == "linear_eval":
            with torch.no_grad():
                feature_maps = model(*clips_protocol)
            preds = head(feature_maps)
        else:
            preds = model(*clips_protocol)

        if cfg.MODEL.LOSS_FUNC == 'multi_margin':
            loss_func = losses.get_loss_func(cfg.MODEL.LOSS_FUNC)(margin=cfg.MODEL.MARGIN, reduction='mean')
        else:
            loss_func = losses.get_loss_func(cfg.MODEL.LOSS_FUNC)(reduction='mean')
        loss = [loss_func(p, labels) for p in preds]

        loss = loss[0] + loss[1] if len(loss) == 2 else loss[0]

        # Check Nan Loss.
        misc.check_nan_losses(loss)

        loss.backward()

        lr = optim.get_lr(
            cfg.SOLVER.LR_POLICY,
            cfg.SOLVER.BASE_LR,
            cfg.SOLVER.WARMUP_START_LR,
            global_step,
            num_optimizer_steps,
            num_warmup_steps,
        )

        finetune_lr = optim.get_lr(
            cfg.SOLVER.LR_POLICY,
            cfg.SOLVER.FINETUNE_LR,
            cfg.SOLVER.WARMUP_START_FINETUNE_LR,
            global_step,
            num_optimizer_steps,
            num_warmup_steps,
        )

        if cfg.SOLVER.PROTOCOL == "finetune":
            optim.set_lr(optimizer, finetune_lr, name="finetune")
            optim.set_lr(optimizer, lr, name="task")
            optimizer.step()
            for p in model.parameters():
                p.grad = None
        else:
            optim.set_lr(optimizer, lr)
            optimizer.step()
            for p in head.parameters():
                p.grad = None

        preds = [torch.nn.functional.softmax(p, dim=-1) for p in preds]
        top1_acc, top5_acc = metrics.topk_accuracies(
            torch.stack(preds).sum(dim=0), labels, (1, 5),
        )

        # Gather all the predictions across all the devices.
        if cfg.NUM_GPUS > 1:
            loss, top1_acc, top5_acc = du.all_reduce([loss, top1_acc, top5_acc])

        loss, top1_acc, top5_acc = (
            loss.item(),
            top1_acc.item(),
            top5_acc.item(),
        )

        train_meter.iter_toc()
        train_meter.update_stats(
            top1_acc,
            top5_acc,
            loss,
            lr,
            labels.size(0) * du.get_world_size()
        )

        train_meter.log_iter_stats(
            cur_epoch, cur_step, global_step,
        )

        if global_step % cfg.LOG_PERIOD == 0:
            logger.info("PROGRESS: {}%".format(round(100*(global_step) / num_steps, 4)))
            logger.info("EVALERR: {}%".format(loss))

        if global_step == num_steps and (cur_step + 1) != len(train_loader):
            return global_step

        train_meter.iter_tic()

    # Log epoch stats.
    train_meter.log_epoch_stats(cur_epoch, global_step)
    train_meter.reset()

    return global_step


@torch.no_grad()
def eval_epoch(
    val_loader,
    model,
    head,
    val_meter,
    cur_epoch,
    cfg,
):
    model.eval()
    if cfg.SOLVER.PROTOCOL == "linear_eval":
        head.eval()
    val_meter.iter_tic()

    for cur_step, inputs in enumerate(val_loader):
        clips, labels, idx = to_cuda(inputs)
        clips_protocol = clips + (cfg.SOLVER.PROTOCOL, )
        if cfg.SOLVER.PROTOCOL == "linear_eval":
            feature_maps = model(*clips_protocol)
            preds = head(feature_maps)
        else:
            preds = model(*clips_protocol)

        top1_acc, top5_acc = metrics.topk_accuracies(
            torch.stack(preds).sum(dim=0), labels, (1, 5),
        )

        if cfg.NUM_GPUS > 1:
            top1_acc, top5_acc = du.all_reduce([top1_acc, top5_acc])
        top1_acc, top5_acc = top1_acc.item(), top5_acc.item()

        val_meter.iter_toc()
        val_meter.update_stats(
            top1_acc,
            top5_acc,
            labels.size(0) * du.get_world_size()
        )

        val_meter.log_iter_stats(cur_epoch, cur_step)
        val_meter.iter_tic()

    val_meter.log_epoch_stats(cur_epoch)
    val_meter.reset()


@torch.no_grad()
def perform_test(
    test_loader,
    model,
    head,
    test_meter,
    cfg,
    cur_epoch=None,
    num_epochs=None,
    writer=None,
):
    model.eval()
    if cfg.SOLVER.PROTOCOL == "linear_eval":
        head.eval()
    test_meter.iter_tic()

    for cur_step, inputs in enumerate(test_loader):
        clips, labels, idx = to_cuda(inputs)
        clips_protocol = clips + (cfg.SOLVER.PROTOCOL, )
        if cfg.SOLVER.PROTOCOL == "linear_eval":
            feature_maps = model(*clips_protocol)
            preds = head(feature_maps)
        else:
            preds = model(*clips_protocol)

        preds = torch.stack(preds).sum(dim=0)
        if cfg.NUM_GPUS > 1:
            preds, labels, idx = du.all_gather([preds, labels, idx])
        preds = preds.cpu()
        labels = labels.cpu()
        idx = idx.cpu()
        test_meter.iter_toc()
        test_meter.update_stats(
            preds.detach(), labels.detach(), idx.detach(),
        )
        test_meter.log_iter_stats(cur_step)
        test_meter.iter_tic()

    stats = test_meter.finalize_metrics(
        ks=(1, 5), cur_epoch=cur_epoch, num_epochs=num_epochs, writer=writer
    )
    test_meter.reset()

    return stats
