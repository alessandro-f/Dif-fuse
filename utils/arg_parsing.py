import argparse
import json
from collections import namedtuple
import pprint
from pathlib import Path
from utils.storage import load_dict_from_json
import os


def load_arg_from_json(json_file_path):
    parser = argparse.ArgumentParser()
    if json_file_path is not None:
        print(os.path.abspath(Path(json_file_path)))
        with open(os.path.abspath(Path(json_file_path)), "rt") as f:
            t_args = argparse.Namespace()
            t_args.__dict__.update(json.load(f))
            args = parser.parse_args(namespace=t_args)
    return args


def load_arg_only_from_json(json_file_path, arg_dict):
    # print(json_file_path)
    config_dict = load_dict_from_json(json_file_path)
    for key in config_dict.keys():
        # if "num_workers" in key:
        #     pass
        # elif "num_gpus_to_use" in key:
        #     pass
        if key in arg_dict.keys():
            arg_dict[key] = config_dict[key]
        else:
            del arg_dict[key]
    # pprint.pprint(arg_dict, indent=4)
    return arg_dict




class DictWithDotNotation(dict):
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as k:
            raise AttributeError(k)

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, key):
        try:
            del self[key]
        except KeyError as k:
            raise AttributeError(k)

    def __repr__(self):
        return "<DictWithDotNotation " + dict.__repr__(self) + ">"


def parse_args():
    """
    Argument parser
    :return: parsed arguments
    """
    parser = argparse.ArgumentParser()
    #data I/O
    parser.add_argument(
        "-dataset_root_folder",
        "--dataset_root_folder",
        type=str,
        default = "data",
        help="data folder"
    )


    parser.add_argument(
        "-batch",
        "--batch_size",
        type=int,
        default=8,#12, 4
        help="Batch Size"
    )

    parser.add_argument(
        "-evalbatch",
        "--eval_batch_size",
        type=int,
        default=8,
        help="Eval Batch Size"
    )

    parser.add_argument(
        "-x",
        "--max_epochs",
        type=int,
        default=200,#200
        help="How many args.max_epochs to run in total?",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=42,#42
        help="Random seed to use")

    # model
    parser.add_argument(
        "-model",
        "--model",
        type=str,
        default="baseline", # ['image attention', 'SMIC', 'HSM', 'SalClassNet', 'ACAT'],
        help="model to use",
    )


    # data related parameters
    parser.add_argument(
        "-total_slices",
        "--total_slices",
        type=int,
        default=11,
        help="Total slices of the input"
    )
    parser.add_argument(
        "-subset_slices",
        "--subset_slices",
        type=int,
        nargs="+",
        default=None,#[4,5,6],#[4,5,6],  # [5],None
        help="Subset of slices. Should be a list"
    )

    parser.add_argument(
        "-nw",
        "--num_workers",
        type=int,
        default=6,
        help="number of workers to use"
    )

    parser.add_argument(
        "-dropout_prob",
        "--dropout_prob",
        type=float,
        default=0,
        help="dropout probability"
    )
    # logging
    parser.add_argument(
        "-en",
        "--experiment_name",
        type=str,
        default="Debug",
        help="Experiment name for the model to be assessed",
    )

    parser.add_argument(
        "-o",
        "--logs_path",
        type=str,
        default="log",
        help="Directory to save log files, check points, and tensorboard.",
    )
    parser.add_argument(
        "-resume",
        "--resume",
        default=False,
        action="store_true",
        help="Resume training?",
    )
    parser.add_argument(
        "-resume_from_baseline",
        "--resume_from_baseline",
        default=False,
        action="store_true",
        help="Load the weights of the baseline model",
    )
    parser.add_argument(
        "-save", "--save", type=int, default=1, help="Save checkpoint files?"
    )
    parser.add_argument(
        "-saveimgs",
        "--save_images",
        type=int,
        default=0,
        help="Build a folder for saved images?",
    )
    parser.add_argument(
        "-filepath_to_arguments_json_config",
        "--filepath_to_arguments_json_config",
        type=str,
        default=None,
        help="If use an independent json file as the input parameters",
    )

    parser.add_argument(
        "-ngpus",
        "--num_gpus_to_use",
        type=int,
        default=0,
        help="The number of GPUs to use, use 0 to enable CPU",
    )

    # optimization
    parser.add_argument(
        "-optim",
        "--optim",
        type=str,
        default="adam",
        help="Optimizer?"
    )
    parser.add_argument(
        "-l",
        "--learning_rate",
        type=float,
        default=0.001,
        help="Base learning rate"
    )
    parser.add_argument(
        "-sched",
        "--scheduler",
        type=str,
        default="CosineAnnealing",
        help="Scheduler for learning rate annealing: CosineAnnealing | MultiStep",
    )

    parser.add_argument(
        "-lr_min",
        "--lr_min",
        type=float,
        default=0.00001,
        help="eta_min parameter for CosineAnnealing"
    )


    parser.add_argument(
        "-wd", "--weight_decay", type=float, default=0.00005, help="Weight decay value"
    )
    parser.add_argument(
        "-mom", "--momentum", type=float, default=0.9, help="Momentum multiplier"
    )

    args = parser.parse_args()

    if args.filepath_to_arguments_json_config is not None:
        args_dict = load_arg_only_from_json(
            json_file_path=args.filepath_to_arguments_json_config, arg_dict=vars(args)
        )
        args = DictWithDotNotation(args_dict)

    if isinstance(args, argparse.Namespace):
        args = vars(args)

    args = DictWithDotNotation(args)

    return args
