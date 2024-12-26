import os
import sys
from typing import Optional, Dict, Any, Tuple

import datasets
import transformers
from transformers import HfArgumentParser

from t2t.args import (
    Text2TrajectoryModelArguments,
    ClevrDataArguments,
    InstructedTrajectoryDataArguments,
    Text2TrajectoryTrainingArguments,
)


def _parse_args(parser: HfArgumentParser, args: Optional[Dict[str, Any]] = None) -> Tuple[Any]:
    if args is not None:
        return parser.parse_dict(args)
    elif len(sys.argv) == 2 and (sys.argv[1].endswith(".yaml") or sys.argv[1].endswith(".yml")):
        return parser.parse_yaml_file(os.path.abspath(sys.argv[1]))
    elif len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        return parser.parse_json_file(os.path.abspath(sys.argv[1]))
    else:
        return parser.parse_args_into_dataclasses()


def parse_train_args(args):
    parser = HfArgumentParser((
        Text2TrajectoryModelArguments,
        InstructedTrajectoryDataArguments,
        Text2TrajectoryTrainingArguments,
    ))

    return _parse_args(parser, args)


def parse_train_args_for_clevr(args):
    parser = HfArgumentParser((
        Text2TrajectoryModelArguments,
        ClevrDataArguments,
        Text2TrajectoryTrainingArguments,
    ))

    parser.add_argument(
        "--pretrain_dir",
        type=str,
        default=None,
        help="Dir of the pretrained parameters",
    )
    parser.add_argument(
        "--do_load_embedding",
        type=bool,
        default=False,
        help="if use pretrained embedding layer",
    )
    parser.add_argument(
        "--scale_loss_ratio",
        type=float,
        default=0.001,
        help="add this to control scale_loss",
    )
    parser.add_argument(
        "--traj_start_layer",
        type=int,
        default=0,
        help="add this to set where traj start",
    )
    parser.add_argument(
        "--pattern_num",
        type=str,
        default='0',
        help="available patterns",
    )

    return _parse_args(parser, args)


def get_train_args(args: Optional[Dict[str, Any]] = None):
    model_args, data_args, training_args = parse_train_args(args)

    # setup logging
    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # TODO: some arguments checking and postprocessing

    return model_args, data_args, training_args


def get_train_args_for_clevr(args: Optional[Dict[str, Any]] = None):
    model_args, data_args, training_args, custom_args = parse_train_args_for_clevr(args)

    # setup logging
    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # TODO: some arguments checking and postprocessing

    return model_args, data_args, training_args, custom_args


def parse_infer_args_for_clevr_lky(
        args: Optional[Dict[str, Any]] = None
        ) -> tuple[Text2TrajectoryModelArguments, ClevrDataArguments, Any]:
    '''
    parse infer args
    '''
    parser = HfArgumentParser((
        Text2TrajectoryModelArguments,
        ClevrDataArguments,
    ))

    parser.add_argument(
        "--pretrain_dir",
        type=str,
        default=None,
        help="Dir of the pretrained parameters",
    )
    parser.add_argument(
        "--do_load_embedding",
        type=bool,
        default=False,
        help="if use pretrained embedding layer",
    )
    parser.add_argument(
        "--pattern_num",
        type=str,
        default='0',
        help="available patterns",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--max_inst_len",
        type=int,
        default=30,
        help="The maximum length of generated instructions.",
    )
    parser.add_argument(
        "--infer_num",
        type=int,
        default=1,
        help="infer how many datapoints.",
    )

    return _parse_args(parser, args)


def get_infer_args_for_clevr_lky(
        args: Optional[Dict[str, Any]] = None
        ) -> tuple[Text2TrajectoryModelArguments, ClevrDataArguments, Any]:
    '''
    parse infer args and set possible log level
    '''
    model_args, data_args, custom_args = parse_infer_args_for_clevr_lky(args)

    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    print('\n------')
    print('parse infer args')
    print(model_args, data_args, custom_args, sep='\n')
    print('------\n')

    return model_args, data_args, custom_args