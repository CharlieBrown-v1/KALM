import os
import math
import json
import torch
import deepspeed
import numpy as np
import transformers
from tqdm import tqdm
from transformers import AutoTokenizer, LlamaForCausalLM

from t2t.datasets import (
    load_clevr_dataset,
    preprocess_it_dataset,
)
from t2t.trainers import Text2TrajectoryTrainer, SaveCallback
from t2t.datasets import (
    DataCollatorForSingleInstructedTrajectory,
    random_split,
    random_split_by_label,
    fixed_split,
)
from t2t.args import get_train_args_for_clevr
from t2t.models import LlataConfig, Llata2ForTrajectoryGeneration
from t2t.utils import plot_loss
from envs.utils.clevr_utils import DIRECTIONS, GOAL_SRC, GOAL_DST


def train_clevr():
    # get arguments
    model_args, data_args, train_args, custom_args = get_train_args_for_clevr()

    # process available patterns
    available_pattern_list = [i for i in range(len(custom_args.pattern_num) * 4) if str(i) in custom_args.pattern_num]

    # set seed
    transformers.set_seed(train_args.seed)

    # load tokenizer
    config_kwargs = {
        "trust_remote_code": True,
    }
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.pretrained_path,
        **config_kwargs,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = data_args.pad_token_id

    # load data and preprocess (may set missing arguments of data_args) 
    dataset = load_clevr_dataset(data_args, available_pattern_list)
    print("===> dataset loaded")

    dataset = preprocess_it_dataset(dataset, tokenizer, data_args)  # tokenize
    print("===> dataset preprocessed")

    # train test split
    if data_args.split_by == "index":
        _, (test_set, train_set) = random_split(dataset, [data_args.validation_ratio, 1 - data_args.validation_ratio])
    elif data_args.split_by == "goal":
        goals = torch.as_tensor(dataset.goals[:, 0])
        _, (test_set, train_set) = random_split_by_label(
            dataset, goals, [data_args.validation_ratio, 1 - data_args.validation_ratio])
    elif data_args.split_by == "direction":
        goals = dataset.goals
        directions = torch.as_tensor(goals[:, 0, -1])
        num_test_directions = math.ceil(data_args.validation_ratio * len(DIRECTIONS))
        test_directions = torch.randperm(len(DIRECTIONS))[:num_test_directions]

        test_indices = [torch.where(directions == direction)[0] for direction in test_directions]
        test_indices = torch.cat(test_indices, dim=0).flatten().tolist()
        train_indices = list(set(range(len(dataset))) - set(test_indices))

        _, (test_set, train_set) = fixed_split(dataset, [test_indices, train_indices])
    elif data_args.split_by == "source" or data_args.split_by == "target":  # this is used in .sh
        goals = dataset.goals
        flags = goals[:, 0, :-1]
        label = GOAL_SRC if data_args.split_by == "source" else GOAL_DST
        ball_indices = torch.argmax(torch.as_tensor(flags == label, dtype=int), dim=-1)  # record the color of ball num 1 in each trajectory
        num_test_balls = math.ceil(data_args.validation_ratio * dataset.num_objects)
        test_balls = torch.randperm(dataset.num_objects)[:num_test_balls]

        test_indices = [torch.where(ball_indices == ball)[0] for ball in test_balls]
        test_indices = torch.cat(test_indices, dim=0).flatten().tolist()  # list of indexes of text ball is ball num 1 in the trajectory
        train_indices = list(set(range(len(dataset))) - set(test_indices))

        _, (test_set, train_set) = fixed_split(dataset, [test_indices, train_indices])
    else:
        raise NotImplementedError(f"Unknown data split_by: {data_args.split_by}")
    print(f"===> dataset splited into train({len(train_set)}) and test({len(test_set)}) ===")

    # data collator
    data_collator = DataCollatorForSingleInstructedTrajectory(tokenizer)

    # load model
    config = LlataConfig.from_pretrained(model_args.pretrained_path, **config_kwargs)
    config.action_loss_ratio = 0.5
    config.scale_loss_ratio = custom_args.scale_loss_ratio  # add this to control scale_loss
    config.traj_start_layer = custom_args.traj_start_layer
    if not model_args.load_full:
        config.observation_type = data_args.observation_type
        config.action_type = data_args.action_type
        config.observation_dim = data_args.observation_dim
        config.action_dim = data_args.action_dim
        config.max_trajectory_length = data_args.max_traj_len
        config.traj_ln = model_args.traj_ln
    config.traj_ln = False  # manually set to False

    model = Llata2ForTrajectoryGeneration.from_pretrained(model_args.pretrained_path, config=config)
    print("===> model initialized")

    '''
    # for debug
    target_opt_name_list = [
        'model.embed_observation.weight',
        'model.embed_observation.bias',
        'model.embed_action.weight',
        'observation_head.weight', 
        'observation_head.bias',
        'action_head.weight',
        'action_head.bias',
        ]
    for n, p in model.named_parameters():
        if n not in target_opt_name_list:
            p.requires_grad = False
        else:
            print(n)
    '''

    # trainer
    trainer = Text2TrajectoryTrainer(
        model=model,
        args=train_args,  # set tensorboard logger with args
        tokenizer=tokenizer,
        data_collator=data_collator,
        train_dataset=train_set,
        eval_dataset=test_set,
    )
    print("===> trainer initialized")

    # load pretrained observation/action embedding if specified
    '''
    if custom_args.do_load_embedding:
        print('\n--- load pretrained embedding ---\n')
        trainer.accelerator.wait_for_everyone()

        save_name_list = ['embed_observation', 'embed_action']
        for idx, module in enumerate([trainer.model.embed_observation, trainer.model.embed_action]):
            named_parameters = dict(module.named_parameters(recurse=False))
            state_dict = torch.load(custom_args.pretrain_dir + f'/{save_name_list[idx]}.pt')
            params_to_gather = [named_parameters[k] for k in state_dict.keys() if k in named_parameters]
            if len(params_to_gather) > 0:
                with deepspeed.zero.GatheredParameters(params_to_gather, modifier_rank=0):
                    if torch.distributed.get_rank() == 0:
                        args = (state_dict, '', {}, True, [], [], [])
                        module._load_from_state_dict(*args)
                        
        trainer.accelerator.wait_for_everyone()
    '''
    
    # save other informations
    if trainer.args.should_save:
        with open(trainer.args.output_dir + "/data_args.json", "w") as f:
            json.dump(data_args.to_dict(), f, indent=2)
        with open(trainer.args.output_dir + "/model_args.json", "w") as f:
            json.dump(model_args.to_dict(), f, indent=2)
        with open(trainer.args.output_dir + "/data_split.json", "w") as f:
            json.dump(
                {"train_set": train_set.indices, "test_set": test_set.indices}, f, indent=2)

    # training
    if train_args.do_train:
        print("===> training started")
        train_results = trainer.train(resume_from_checkpoint=train_args.resume_from_checkpoint)
        print("===> training finished")

        trainer.log_metrics("train", train_results.metrics)
        trainer.save_metrics("train", train_results.metrics)
        trainer.save_state()
        trainer.save_model()
        if trainer.is_world_process_zero() and train_args.plot_loss:
            plot_loss(train_args.output_dir, keys=["loss", "eval_loss"])

    # evaluation
    if train_args.do_eval:
        print("===> evaluation started")
        metrics = trainer.evaluate(metric_key_prefix="eval")
        print("===> evaluation finished")

        try:
            perplexity = math.exp(metrics["eval_loss"])
        except OverflowError:
            perplexity = float("inf")

        metrics["eval_perplexity"] = perplexity
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    print("===> all finished")


if __name__ == "__main__":
    # torch.autograd.set_detect_anomaly(True)
    train_clevr()
