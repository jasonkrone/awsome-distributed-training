# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

import datetime
import functools
import math
import re
import time

import numpy as np
import torch
from torch import optim
import torch.distributed as dist
import torch.utils.data

import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision
from torch.distributed.fsdp import ShardingStrategy
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy, transformer_auto_wrap_policy
from torch.utils.data import DataLoader

from model_utils.concat_dataset import ConcatTokensDataset
from model_utils.train_utils import (get_model_config,
                                   compute_num_params,
                                   get_transformer_layer,
                                   get_sharding_strategy,
                                   get_backward_fetch_policy,
                                   apply_activation_checkpoint,
                                   get_param_groups_by_weight_decay,
                                   get_logger,
                                   get_learning_rate_scheduler,
                                   create_streaming_dataloader)
from model_utils.checkpoint import save_checkpoint, load_checkpoint
from model_utils.arguments import parse_args

import sys
sys.path.append("/home/ubuntu/jpt")
from model import Decoder, TransformerBlock
from utils import Config


logger = get_logger()

USE_JPK_MODEL = False


def eval_model(model, dataloader, num_batches):
    """Eval step."""
    model = model.eval()
    n_batches = 0
    loss = 0.0

    with torch.no_grad():
        for batch_idx, input_data in enumerate(dataloader):
            if batch_idx >= num_batches:
                break

            if USE_JPK_MODEL:
                _, elem_loss = model(input_data, input_data)
                loss += elem_loss
            else:
                loss += model(input_ids=input_data, attention_mask=None, labels=input_data)["loss"]
            n_batches += 1

    if n_batches > 0:
        detached_loss = loss.detach()
        torch.distributed.all_reduce(detached_loss)
        loss = detached_loss.item() / dist.get_world_size()
        loss /= n_batches
        ppl = math.exp(loss)
    else:
        loss = -1.0
        ppl = -1.0

    return loss, ppl

def train(
        model,
        optimizer,
        train_dataloader,
        val_dataloader,
        lr_scheduler,
        model_config,
        num_params,
        args,
        global_rank,
        world_size,
        total_steps=0,
        start_batch_index=0
    ):
    model.train()
    for index in range(args.epochs):
        for batch_idx, input_data in enumerate(train_dataloader):
            if batch_idx < start_batch_index:
                continue
            optimizer.zero_grad(set_to_none=True)
            step_start = time.time()

            if USE_JPK_MODEL:
                _, loss = model(input_data, input_data)
            else:
                loss = model(input_ids=input_data, attention_mask=None, labels=input_data)["loss"]

            loss.backward()
            model.clip_grad_norm_(args.grad_clip)
            optimizer.step()
            lr_scheduler.step()
            total_steps += 1
            loss_metric = loss.item()
            step_time = time.time() - step_start

            tokens_processed = input_data.shape[0] * input_data.shape[1]
            throughput = tokens_processed / step_time

            loss_scalar = loss.item()
            current_lr = lr_scheduler.get_lr()
            if global_rank==0 and batch_idx % args.logging_freq==0:
                logger.info(
                    "Batch %d Loss: %.5f, Speed: %.2f tokens/sec, lr: %.6f",  # pylint: disable=line-too-long
                    batch_idx,
                    loss_scalar,
                    throughput,
                    current_lr,
                )
            if args.validation_freq and not total_steps % args.validation_freq:
                val_loss, val_ppl = eval_model(
                    model, val_dataloader, args.validation_batches
                )
                model = model.train()
                if global_rank == 0:
                    logger.info(
                            "Batch %d Validation loss: %s",
                            batch_idx,
                            val_loss,
                        )
            if args.checkpoint_dir and not total_steps % args.checkpoint_freq:
                user_content = {
                    "cli_args": args.__dict__,
                    "num_params": num_params,
                    "total_steps": total_steps,
                    "model_config": model_config,
                    "start_batch_index": batch_idx + 1,
                }
                sub_dir = f"{args.model_type}-{total_steps}steps"

                save_checkpoint(
                    model,
                    optimizer,
                    lr_scheduler,
                    user_content,
                    args.checkpoint_dir,
                    sub_dir,
                )
            if total_steps >= args.max_steps:
                break


def main(args):
    dist.init_process_group()
    global_rank = dist.get_rank()
    device = global_rank % torch.cuda.device_count()
    world_size = dist.get_world_size()

    if args.bf16:
        dtype = torch.bfloat16
    else:
        dtype = torch.get_default_dtype()

    model_config = get_model_config(args)
    if global_rank == 0:
        logger.info(
            "Creating Model"
        )

    if USE_JPK_MODEL:
        config = Config.from_yaml("/home/ubuntu/jpt/configs/train/7b_fsdp/train_7b_fsdp_neox_memmap.yaml")
        model = Decoder(config.model)
        transformer_layer = TransformerBlock
    else:
        model = AutoModelForCausalLM.from_config(model_config)
        transformer_layer = get_transformer_layer(args.model_type)

    num_params = compute_num_params(model)
    if global_rank == 0:
        logger.info(
            "Created model with total parameters: %d (%.2f B)", num_params, num_params * 1e-9
        )

    gpt_auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={
            transformer_layer,
        },
    )

    torch.cuda.set_device(device)
    mixed_precision_policy = MixedPrecision(
        param_dtype=dtype, reduce_dtype=dtype, buffer_dtype=dtype
    )

    if args.sharding_strategy=="full":
        sharding_strategy = ShardingStrategy.FULL_SHARD
    elif args.sharding_strategy=="hybrid":
        sharding_strategy = ShardingStrategy.HYBRID_SHARD
    else:
        raise NotImplementedError("Available sharding strategies are full and hybrid")

    model = FSDP(
        model,
        auto_wrap_policy=gpt_auto_wrap_policy,
        mixed_precision=mixed_precision_policy,
        limit_all_gathers=args.limit_all_gathers,
        device_id=torch.cuda.current_device(),
        use_orig_params=False,
        sharding_strategy=sharding_strategy,
    )

    if global_rank == 0:
        logger.info("Wrapped model with FSDP")

    if args.activation_checkpointing > 0:
        apply_activation_checkpoint(args, model=model)

    if args.offload_activations > 0:
        from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import offload_wrapper

        model = offload_wrapper(model)

    param_groups = get_param_groups_by_weight_decay(model)

    optimizer = optim.AdamW(
        param_groups, betas=(args.beta1, args.beta2), lr=args.lr, weight_decay=args.weight_decay
    )

    if global_rank == 0:
        logger.info("Created optimizer")

    lr_scheduler = get_learning_rate_scheduler(optimizer, args)

    if args.resume_from_checkpoint:
        (
            model,
            optimizer,
            lr_scheduler,
            total_steps,
            start_batch_index,
        ) = load_checkpoint(model,
                            optimizer,
                            lr_scheduler,
                            args.resume_from_checkpoint,
                            args.model_type,
                            device)
    else:
        total_steps = 0
        start_batch_index = 0

    train_dataloader = create_streaming_dataloader(args.dataset,
                                                   args.tokenizer,
                                                   name=args.dataset_config_name,
                                                   batch_size=args.train_batch_size,
                                                   split='train')

    val_dataloader = create_streaming_dataloader(args.dataset,
                                                  args.tokenizer,
                                                  name=args.dataset_config_name,
                                                  batch_size=args.train_batch_size,
                                                  split='validation')

    train(model,
          optimizer,
          train_dataloader,
          val_dataloader,
          lr_scheduler,
          model_config,
          num_params,
          args,
          global_rank,
          world_size,
          total_steps,
          start_batch_index)

if __name__ == "__main__":
    args, _ = parse_args()
    main(args)
