#!/usr/bin/env python3

import os
import argparse
import random
import datetime
import glob
from tqdm import tqdm
import numpy as np

import torch

from habitat import logger

from pointnav_vo.utils.config_utils import update_config_log
from pointnav_vo.utils.baseline_registry import baseline_registry
from pointnav_vo.config.rl_config.default import get_config as get_rl_config
from pointnav_vo.config.vo_config.default import get_config as get_vo_config


VIS_TYPE_DICT = {
    "rgb": "rgb",
    "depth": "d",
    "discretized_depth": "dd",
    "top_down_view": "proj",
}

GEO_SHORT_NAME = {
    "inverse_data_augment_only": "inv_aug",
    "inverse_joint_train": "inv_joint",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task-type",
        choices=["rl", "vo"],
        required=True,
        help="Specify the task category of experiment.",
    )
    parser.add_argument(
        "--noise",
        type=int,
        required=True,
        help="Specify whether enable noisy environment.",
    )
    parser.add_argument(
        "--run-type",
        choices=["train", "eval"],
        required=True,
        help="run type of the experiment (train or eval)",
    )
    parser.add_argument(
        "--exp-config",
        type=str,
        required=True,
        help="path to config yaml containing info about experiment",
    )
    parser.add_argument(
        "--n-gpu", type=str, required=False, help="DEPRECATED! please specify in .yaml | timestamp for current executing."
    )
    parser.add_argument(
        "--cur-time", type=str, required=True, help="timestamp for current executing."
    )
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="Modify config options from command line",
    )

    args = parser.parse_args()
    run_exp(**vars(args))


def run_exp(
    task_type: str,
    noise: int,
    exp_config: str,
    run_type: str,
    n_gpu: str,
    cur_time: str,
    opts=None,
) -> None:
    r"""Runs experiment given mode and config

    Args:
        exp_config: path to config file.
        run_type: "train" or "eval.
        opts: list of strings of additional config options.

    Returns:
        None.
    """

    if task_type == "rl":
        config = get_rl_config(exp_config, opts)
        model_infos = config.RL.Policy
    elif task_type == "vo":
        config = get_vo_config(exp_config, opts)
        model_infos = config.VO.MODEL
    else:
        pass

    config.defrost()
    config.exp_config = exp_config
    config.freeze()

    if task_type == "rl":
        rgb_noise = "NOISE_MODEL" in config.TASK_CONFIG.SIMULATOR.RGB_SENSOR
        depth_noise = "NOISE_MODEL" in config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR
        action_noise = "NOISE_MODEL" in config.TASK_CONFIG.SIMULATOR

    if noise == 1:
        if task_type == "vo":
            config.defrost()
            config.VO.DATASET.TRAIN = config.VO.DATASET.TRAIN_WITH_NOISE
            config.VO.DATASET.EVAL = config.VO.DATASET.EVAL_WITH_NOISE
            config.freeze()
        else:
            assert rgb_noise or depth_noise or action_noise
    elif noise == 0:
        if task_type == "vo":
            config.defrost()
            config.VO.DATASET.TRAIN = config.VO.DATASET.TRAIN_NO_NOISE
            config.VO.DATASET.EVAL = config.VO.DATASET.EVAL_NO_NOISE
            config.freeze()
        else:
            assert not rgb_noise and not depth_noise and not action_noise
    else:
        pass

    vo_pretrained_ckpt_type = "none"

    if run_type == "train":
        
        if config.RESUME_TRAIN:
            
            # new behavior
            if hasattr(config, 'RESUME_STATE_PATH'): 

                config.defrost()

                # auto resume from (non-unique) config path
                if config.RESUME_STATE_FILE == 'auto':
                    log_folder_name = config.exp_config.split('/')[-1].split('.')[0]
                    log_dir = os.path.join(config.LOG_DIR, log_folder_name)

                    # if log_dir doesn't exist or is empty
                    if not os.path.exists(os.path.join(log_dir,'checkpoints')) or not os.listdir(os.path.join(log_dir,'checkpoints')):
                        config.RESUME_TRAIN = False
                        config.RESUME_STATE_FILE = "start"
                    else:
                        config.RESUME_STATE_PATH = log_dir
                        config.RESUME_STATE_FILE = "latest"

                # resume latest from (unique) path
                elif config.RESUME_STATE_FILE == 'latest':
                    # strip /checkpoints because code assumes/creates it
                    log_dir = config.RESUME_STATE_PATH[:-11]

                # old behavior
                else:
                    log_dir = os.path.join(
                        # os.path.dirname(config.RESUME_STATE_FILE), f"resume_{cur_time}"
                        config.RESUME_STATE_PATH, f"resume_{cur_time}"
                    )
                 
                # get latest run
                if config.RESUME_STATE_FILE == 'latest':
                    assert task_type in ["vo", "rl"], "Invalid task_type, choose one of ['vo','rl'] !"
                    if task_type == "vo":
                        dirs = os.listdir(os.path.join(log_dir,'checkpoints'))
                        ckpt_ids = [int(dir.split('_')[-1].split('.')[0]) for dir in dirs if (dir[-4:] == '.pth' and dir != 'best_vo.pth')]
                        config.RESUME_STATE_FILE = f'ckpt_epoch_{np.max(ckpt_ids)}.pth'
                    elif task_type == 'rl':
                        config.RESUME_STATE_FILE = 'latest_rl_tune_vo.pth'
                
                # set resume state file
                config.RESUME_STATE_FILE = os.path.join(config.RESUME_STATE_PATH, 'checkpoints', config.RESUME_STATE_FILE)
                
                config.freeze()

                print(f"Resuming from {config.RESUME_STATE_FILE}")

            # old behavior
            else:
                log_dir = os.path.join(
                    os.path.dirname(config.RESUME_STATE_FILE), f"resume_{cur_time}"
                )
        else:
            # adding some tags to logging directory
            if task_type == "rl":
                log_folder_name = "{}_dt_{}".format(config.exp_config.split('/')[-1].split('.')[0], cur_time)
                if config.RL.TUNE_WITH_VO:
                    vo_pretrained_ckpt_type = config.VO.REGRESS_MODEL.pretrained_type

            elif task_type == "vo":
                if isinstance(config.VO.TRAIN.action_type, list):
                    act_str = "_".join([str(_) for _ in config.VO.TRAIN.action_type])
                else:
                    act_str = config.VO.TRAIN.action_type
                
                log_folder_name = "{}_dt_{}".format(config.exp_config.split('/')[-1].split('.')[0], cur_time)
            else:
                pass
            log_folder_name = "{}_s_{}".format(log_folder_name, config.TASK_CONFIG.SEED)
            log_dir = os.path.join(config.LOG_DIR, log_folder_name)

    elif "eval" in run_type:
        # save evaluation infos to the checkpoint's directory
        if os.path.isfile(config.EVAL.EVAL_CKPT_PATH):
            single_str = "single"
            log_dir = os.path.dirname(config.EVAL.EVAL_CKPT_PATH)
            tmp_eval_f = config.EVAL.EVAL_CKPT_PATH
        else:
            single_str = "mult"
            log_dir = config.EVAL.EVAL_CKPT_PATH
            tmp_eval_f = list(
                glob.glob(os.path.join(config.EVAL.EVAL_CKPT_PATH, "*.pth"))
            )[0]

        if task_type == "vo":
            log_dir = os.path.join(log_dir, f"eval_{cur_time}")
        elif task_type == "rl":
            tmp_config = torch.load(tmp_eval_f)["config"]
            ckpt_rgb_noise = (
                "NOISE_MODEL" in tmp_config.TASK_CONFIG.SIMULATOR.RGB_SENSOR
            )
            ckpt_depth_noise = (
                "NOISE_MODEL" in tmp_config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR
            )
            ckpt_action_noise = "NOISE_MODEL" in tmp_config.TASK_CONFIG.SIMULATOR
            if config.VO.USE_VO_MODEL:
                if config.VO.VO_TYPE == "REGRESS":
                    vo_pretrained_ckpt_type = config.VO.REGRESS_MODEL.pretrained_type
                else:
                    raise ValueError
            log_dir = os.path.join(
                log_dir,
                "seed_{}-{}-{}_ckpt-train_noise_rgb_{}_depth_{}_act_{}-"
                "eval_noise_rgb_{}_depth_{}_act_{}-vo_{}-mode_{}-rnd_n_{}-{}".format(
                    config.TASK_CONFIG.SEED,
                    config.EVAL.SPLIT,
                    single_str,
                    int(ckpt_rgb_noise),
                    int(ckpt_depth_noise),
                    int(ckpt_action_noise),
                    int(rgb_noise),
                    int(depth_noise),
                    int(action_noise),
                    vo_pretrained_ckpt_type,
                    config.VO.REGRESS_MODEL.mode,
                    config.VO.REGRESS_MODEL.rnd_mode_n,
                    cur_time,
                ),
            )
        else:
            pass
    else:
        raise ValueError

    if vo_pretrained_ckpt_type != "none" and config.VO.VO_TYPE == "REGRESS":
        config.defrost()
        config.VO.REGRESS_MODEL.pretrained_ckpt = config.VO.REGRESS_MODEL.all_pretrained_ckpt[
            vo_pretrained_ckpt_type
        ]
        config.freeze()

    config = update_config_log(config, run_type, log_dir)
    logger.add_filehandler(config.LOG_FILE)

    # reproducibility set up
    random.seed(config.TASK_CONFIG.SEED)
    np.random.seed(config.TASK_CONFIG.SEED)
    torch.manual_seed(config.TASK_CONFIG.SEED)
    torch.cuda.manual_seed_all(config.TASK_CONFIG.SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if run_type == "train":
        engine_name = config.ENGINE_NAME
    elif "eval" in run_type:
        if config.EVAL.EVAL_WITH_CKPT:
            if os.path.isfile(config.EVAL.EVAL_CKPT_PATH):
                eval_f_list = [config.EVAL.EVAL_CKPT_PATH]
            else:
                eval_f_list = list(
                    glob.glob(os.path.join(config.EVAL.EVAL_CKPT_PATH, "*.pth"))
                )
                eval_f_list = sorted(eval_f_list, key=lambda x: os.stat(x).st_mtime)
            engine_name = torch.load(eval_f_list[0])["config"].ENGINE_NAME
        else:
            raise NotImplementedError
    else:
        raise ValueError

    if task_type == "rl":
        trainer_init = baseline_registry.get_trainer(engine_name)
    elif task_type == "vo":
        trainer_init = baseline_registry.get_vo_engine(engine_name)
    else:
        trainer_init = None

    assert trainer_init is not None, f"{config.ENGINE_NAME} is not supported"

    if run_type == "train":
        trainer = trainer_init(config, run_type)
        trainer.train()
    elif "eval" in run_type:
        if task_type == "vo":
            for i, eval_f in tqdm(enumerate(eval_f_list), total=len(eval_f_list)):
                verbose = i == 0
                config.defrost()
                config.EVAL.EVAL_CKPT_PATH = eval_f
                config.freeze()
                trainer = trainer_init(config.clone(), run_type, verbose=verbose)

                if config.EVAL.EVAL_WITH_CKPT:
                    ckpt_epoch = int(
                        os.path.basename(eval_f).split("epoch_")[1].split(".")[0]
                    )
                else:
                    ckpt_epoch = 0

                for act in config.VO.EVAL.eval_acts:
                    trainer.eval(
                        eval_act=act,
                        split_name=f"eval_{act}" if act != "no_specify" else "eval",
                        epoch=ckpt_epoch,
                        save_pred=config.VO.EVAL.save_pred,
                        rank_pred=config.VO.EVAL.rank_pred,
                        rank_top_k=config.VO.EVAL.rank_top_k,
                    )
        else:
            trainer = trainer_init(config, run_type, verbose=False)
            trainer.eval()
    else:
        raise ValueError


if __name__ == "__main__":
    main()
