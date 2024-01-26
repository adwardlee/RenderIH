from __future__ import absolute_import

import os
import sys
import shutil
import torch
import math
import numpy as np
import scipy.io
import datetime
import warnings
import traceback
import pickle
import matplotlib.pyplot as plt
from termcolor import colored, cprint
import hocontact.utils.func as func
from hocontact.utils.logger import logger


def load_opts(resume_path):
    # Identify if folder or checkpoint is provided
    if resume_path.endswith(".pth") or resume_path.endswith(".pth.tar"):
        resume_path = os.path.join(*resume_path.split("/")[:-1])
    opt_path = os.path.join(resume_path, "opt.pkl")
    with open(opt_path, "rb") as p_f:
        opts = pickle.load(p_f)
    return opts


def print_query(querylist, col=3, desp="Queries", align="c"):
    if not isinstance(querylist, list):
        querylist = list(querylist)
    if len(querylist) == 0:
        return

    querystr_list = [str(q) for q in querylist]
    querystr_list = sorted(querystr_list)

    def fn(templist, col):
        for i in range(0, len(templist), col):
            yield templist[i : i + col]

    querystr_list = fn(querystr_list, col)  # split

    import prettytable as pt

    logger.warn("{}  {}  {}".format("=" * 30, desp, ">" * 30), "magenta")
    tb = pt.PrettyTable(padding_width=5, header=False)
    for i, qstr in enumerate(querystr_list):  # formating print
        qstr = qstr + ["-"] * (col - len(qstr))
        tb.add_row(qstr)
    logger.info(str(tb))
    logger.warn("{}  {}  {}".format("<" * 30, desp, "<" * 30), "magenta")
    return


def print_args(args):
    opts = vars(args)
    logger.warn("{}  Options  {}".format("=" * 30, ">" * 30), "yellow")
    for k, v in sorted(opts.items()):
        logger.info("{:<30}  :  {}".format(k, v))
    logger.warn("{}  Options  {}".format("<" * 30, "<" * 30), "yellow")


def print_results(result_dict, title="Results"):
    logger.warn("{}  {}  {}".format("=" * 30, title, ">" * 30), "blue")
    for k, v in sorted(result_dict.items()):
        logger.info("{:<30}  :  {}".format(k, v))
    logger.warn("{}  {}  {}".format("<" * 30, title, "<" * 30), "blue")


def save_args(args, save_folder, opt_prefix="opt"):
    opts = vars(args)
    # Create checkpoint folder
    if not os.path.exists(save_folder):
        os.makedirs(save_folder, exist_ok=True)

    # Save options
    opt_filename = "{}.txt".format(opt_prefix)
    opt_path = os.path.join(save_folder, opt_filename)
    with open(opt_path, "a") as opt_file:
        opt_file.write("====== Options ======\n")
        for k, v in sorted(opts.items()):
            opt_file.write("{option}: {value}\n".format(option=str(k), value=str(v)))
        opt_file.write("=====================\n")
        opt_file.write("launched {} at {}\n".format(str(sys.argv[0]), str(datetime.datetime.now())))
    opt_picklename = "{}.pkl".format(opt_prefix)
    opt_picklepath = os.path.join(save_folder, opt_picklename)
    with open(opt_picklepath, "wb") as opt_file:
        pickle.dump(opts, opt_file)

    logger.info("Saved options to {}".format(opt_path))


def param_count(net):
    return sum(p.numel() for p in net.parameters()) / 1e6


def param_size(net):
    # ! treat all parameters to be float
    return sum(p.numel() for p in net.parameters()) * 4 / (1024 * 1024)


def reload_optimizer(optimizer, resume_path, map_location=None):
    if os.path.isfile(resume_path):
        print("=> loading optimizer checkpoint '{}'".format(resume_path))
        checkpoint = torch.load(resume_path, map_location=map_location)
    try:
        missing_states = set(optimizer.state_dict().keys()) - set(checkpoint["optimizer"].keys())
        if len(missing_states) > 0:
            warnings.warn("Missing keys in optimizer ! : {}".format(missing_states))
        optimizer.load_state_dict(checkpoint["optimizer"])
    except ValueError:
        traceback.print_exc()
        warnings.warn("Couldn' load optimizer from {}".format(resume_path))


def remapping_state_dict_from_hasson_to_honet(state_dict):
    # !   remappint STATE_DICT from pretrained model of HASSON[CVPR2020] to our HONet
    need_to_be_remove = []
    need_to_be_insert = {}

    for key in state_dict.keys():
        if "mano_layer_left" in key:
            need_to_be_remove.append(key)

        elif "mano_layer_right" in key:
            need_to_be_remove.append(key)
            new_key = key.replace("mano_layer_right", "mano_layer")
            need_to_be_insert[new_key] = state_dict[key]

        elif "scaletrans_branch_obj" in key:
            need_to_be_remove.append(key)
            new_key = key.replace("scaletrans_branch_obj", "obj_transhead")
            need_to_be_insert[new_key] = state_dict[key]

        elif "scaletrans_branch." in key:
            need_to_be_remove.append(key)
            new_key = key.replace("scaletrans_branch", "mano_transhead")
            need_to_be_insert[new_key] = state_dict[key]

    state_dict.update(need_to_be_insert)
    for key in need_to_be_remove:
        state_dict.pop(key)
    return state_dict


def reload_checkpoint(
    model,
    resume_path,
    optimizer=None,
    startswith=None,
    strict=True,
    as_parallel=False,
    map_location=None,
    reload_honet_checkpoints=None,
):
    if os.path.isfile(resume_path):
        print("=> loading checkpoint '{}'".format(resume_path))
        checkpoint = torch.load(resume_path, map_location=map_location)
        if as_parallel:
            if "module" in list(checkpoint["state_dict"].keys())[0]:
                state_dict = checkpoint["state_dict"]
            else:
                state_dict = {"module.{}".format(key): item for key, item in checkpoint["state_dict"].items()}
        else:
            if "module" in list(checkpoint["state_dict"].keys())[0]:
                state_dict = {".".join(key.split(".")[1:]): item for key, item in checkpoint["state_dict"].items()}
            else:
                state_dict = checkpoint["state_dict"]
        # filter out tensors not startswith given keyword and strip keyword out, if startswith is not None:
        if startswith is not None:
            state_dict = {
                ".".join(key.split(".")[1:]): item for key, item in state_dict.items() if key.startswith(startswith)
            }
            print(
                "=> loaded checkpoint '{}' (keyword {}, epoch {})".format(resume_path, startswith, checkpoint["epoch"])
            )
        else:
            print("=> loaded checkpoint '{}' (epoch {})".format(resume_path, checkpoint["epoch"]))

        if reload_honet_checkpoints is not None:
            assert as_parallel, "reload honet_checkpoints only used in parallel PiCR module"
            print(f"reload honet checkpoints from {reload_honet_checkpoints}")
            honet_state_dict = torch.load(reload_honet_checkpoints, map_location=map_location)["state_dict"]
            old_honet_list = []
            for k in state_dict.keys():
                if "module.ho_net" in old_honet_list:
                    old_honet_list.append(k)
            for k in old_honet_list:
                state_dict.pop(k)
            # * ==> remappint STATE_DICT from pretrained model of HASSON[CVPR2020] to our HONet
            honet_state_dict = remapping_state_dict_from_hasson_to_honet(honet_state_dict)
            for k, v in honet_state_dict.items():
                state_dict[f"module.ho_net.{k.replace('module.', '')}"] = v

        # * ==> remappint STATE_DICT from pretrained model of HASSON[CVPR2020] to our HONet
        state_dict = remapping_state_dict_from_hasson_to_honet(state_dict)

        missing_states = set(model.state_dict().keys()) - set(state_dict.keys())
        if len(missing_states) > 0:
            warnings.warn("Missing keys ! : {}".format(missing_states))
        model.load_state_dict(state_dict, strict=strict)
        if optimizer is not None:
            try:
                missing_states = set(optimizer.state_dict().keys()) - set(checkpoint["optimizer"].keys())
                if len(missing_states) > 0:
                    warnings.warn("Missing keys in optimizer ! : {}".format(missing_states))
                optimizer.load_state_dict(checkpoint["optimizer"])
            except ValueError:
                traceback.print_exc()
                warnings.warn("Couldn' load optimizer from {}".format(resume_path))
    else:
        raise ValueError("=> no checkpoint found at '{}'".format(resume_path))
    if "best_auc" in checkpoint:
        warnings.warn("Using deprecated best_acc instead of best_auc")
        best = checkpoint["best_auc"]
    elif "best_acc" in checkpoint:
        warnings.warn("Using deprecated best_acc instead of best_auc")
        best = checkpoint["best_acc"]
    elif "best_score" in checkpoint:
        best = checkpoint["best_score"]
    else:
        best = None
    return checkpoint["epoch"], best


def save_checkpoint(
    state, is_best, checkpoint="checkpoint", filename="checkpoint.pth", snapshot=None,
):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)

    if snapshot and state["epoch"] % snapshot == 0:
        shutil.copyfile(
            filepath, os.path.join(checkpoint, "checkpoint_{}.pth.tar".format(state["epoch"])),
        )

    if is_best:
        if "score" in state:
            shutil.copyfile(filepath, os.path.join(checkpoint, f"model_best_{round(state['score'], 3)}.pth.tar"))
        else:
            shutil.copyfile(filepath, os.path.join(checkpoint, "model_best.pth.tar"))


def clean_state_dict(state_dict):
    """save a cleaned version of model without dict and DataParallel

    Arguments:
        state_dict {collections.OrderedDict} -- [description]

    Returns:
        clean_model {collections.OrderedDict} -- [description]
    """

    clean_model = state_dict
    # create new OrderedDict that does not contain `module.`
    from collections import OrderedDict

    clean_model = OrderedDict()
    if any(key.startswith("module") for key in state_dict):
        for k, v in state_dict.items():
            name = k[7:]  # remove `module.`
            clean_model[name] = v
    else:
        return state_dict

    return clean_model


def save_optim_state(optimizer, filename):
    torch.save({"optim": optimizer.state_dict(),}, filename)


def load_optim_state(optimizer, filename):
    if not os.path.exists(filename):
        print("warning, no ckpt file for optimizer")
        return
    ckpt = torch.load(filename)
    optimizer.load_state_dict(ckpt["optim"])
