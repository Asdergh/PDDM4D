import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import plyfile
import logging
import wandb
from typing import (Literal, Optional, List, Any, Union, Dict)
from ..scene import (GsModule, GsModelOutput, GsModelConfig, AnchorGrowing)
from ..data import (MipNerf360v2DatasetConfig, get_dataset)
from ..metrics import CombinedVisualLoss, CombinedVisualLossConfig
from tqdm import tqdm
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
from inspect import signature
from dataclasses import (dataclass, field, fields, is_dataclass, asdict)
import abc
from torchvision.transforms import Resize, Normalize, Compose


def print_memo():
    (free_mem, total_mem) = th.cuda.mem_get_info()
    print(f"Free/Total memory: [{free_mem / (1024**3)}, {total_mem / (1020**3)}]-GB")

def _get_logger(file: str):
    logger = logging.getLogger()
    formatter = logging.Formatter(fmt="%(asctime)s - %(levelname)s - %(message)s")
    logger.setLevel(logging.DEBUG)
    if (file is not None):
        fhandler = logging.FileHandler(filename=file)
        fhandler.setLevel(logging.DEBUG)
        fhandler.setFormatter(formatter)
        logger.addHandler(fhandler)
    shandler = logging.StreamHandler()
    shandler.setLevel(logging.DEBUG)
    shandler.setFormatter(formatter)
    logger.addHandler(shandler)
    return logger

def get_transform(**kwargs):
    transforms = []
    size = kwargs.get("size", None)
    if size is not None:
        size = size if isinstance(size, tuple) else 2*(size, )
        transforms.append(Resize(size))
    mean = kwargs.get("mean", None)
    std = kwargs.get("std", None)
    if (mean is not None) \
        and (std is not None):
        transforms.append(Normalize(mean=mean, std=std))

    if not transforms:
        transforms.apend(nn.Identity())
    return Compose(transforms)
    
@dataclass
class DefaultConfig(abc.ABC):
    logdir: str
    epochs: int=1000
    batch_size: int=32
    shuffle: bool=False
    device: str="cuda"
    dataset_config: Union[
        MipNerf360v2DatasetConfig
    ] | Dict[str, Any]=field(default_factory=lambda: dict(name="360v2"))
    transform_config: Dict[str, Any]=field(default_factory=lambda: dict(size=224))

    def get_loader(self):
        dataset = get_dataset(self.dataset_config) \
                        if not isinstance(self.dataset_config, dict) \
                        else get_dataset(self.dataset_config["name"], **self.dataset_config)
        return DataLoader(dataset=dataset,
                            batch_size=self.batch_size,
                            shuffle=self.shuffle)

    def to_ommega_conf(self, path: str=None):
        config = dict()
        for attrib in fields(self):
            values = getattr(self, attrib.name)
            if is_dataclass(values):
                values = asdict(values)
            config[attrib.name] = values
        config = OmegaConf.create(config)
        if path is not None:
            OmegaConf.save(config, f=path, resolve=True)
        else:
            return config
        
    @classmethod
    @abc.abstractmethod
    def from_yaml(cfg, path: str):
        """config structure if different for every
        training case. from_yaml must be implemneted 
        for every training pipline."""
        raise NotImplemented("from_yaml() must be initialized \n"
                            "for every trianing config class")        


@dataclass
class Training3DConfig(DefaultConfig):
    gaussian_config: GsModelConfig | Dict[str, Any]=field(default_factory=lambda: dict())
    criterion_config: CombinedVisualLossConfig | Dict[str, Any]=field(default_factory=lambda: dict())
    trainer: str=field(default="3D", repr=False)

    def __post_init__(self):
        self.gaussian_config = self.gaussian_config\
                                if not isinstance(self.gaussian_config, dict)\
                                else GsModelConfig(**self.gaussian_config)
        self.criterion_config = self.criterion_config\
                                if not isinstance(self.criterion_config, dict)\
                                else CombinedVisualLossConfig(**self.criterion_config)

    @classmethod
    def from_yaml(cfg, path: str) -> 'Training3DConfig':
        if os.path.exists(path):
            print("reading!!!")
            cfg = OmegaConf.load(path)
            cfg = OmegaConf.to_container(cfg)
            gaussian_config = GsModelConfig(**cfg["gaussian_config"])
            criterion_config = CombinedVisualLossConfig(**cfg["criterion_config"])
            kwargs = {k: v for (k, v) in cfg.items() if k not in ["gaussian_config", "criterion_config"]}
            kwargs.update({"gaussian_config": gaussian_config,
                            "criterion_config": criterion_config})
            return Training3DConfig(**kwargs)
        

def train3D(cfg: Training3DConfig):

    def _parse_logdir(logdir: str):
        os.makedirs(logdir, exist_ok=True)
        paths = dict(
            logmsgs=os.path.join(logdir, "training.log"),
            sparse=os.path.join(logdir, "training.ply"),
            checkpoints=os.path.join(logdir, "training.ckpt")
        )
        return paths
    paths = _parse_logdir(cfg.logdir)
    logger = _get_logger(paths["logmsgs"])
    wandb.init(project="PCA-Anchor-Gs3D",
                config=cfg.gaussian_config,
                dir=cfg.logdir)
    
    def _save_weights(pc):
        pc.write_ply(paths["sparse"])
        gs_cfg = OmegaConf.structured(cfg.gaussian_config)
        gs_cfg = OmegaConf.to_container(gs_cfg)
        checkpoint = dict(gaussian_heads=pc.state_dict(),
                        optimizer_states=pc.optimizer.state_dict(),
                        sparse_states=paths["sparse"],
                        hyperparams=gs_cfg)
        th.save(checkpoint, f=paths["checkpoints"])

    transform = get_transform(**cfg.transform_config)
    criterion = CombinedVisualLoss(cfg.criterion_config).to(cfg.gaussian_config.device)
    pc = GsModule(cfg.gaussian_config, device=cfg.device)
    loader = cfg.get_loader()
    pc.setup_from_pts(**loader.dataset.sparse)
    densifier = AnchorGrowing(pc, cfg.gaussian_config, logger=logger)
    try:
        for epoch in tqdm(
            range(cfg.epochs), 
            desc="3D Training...", 
            colour="green"
        ):
            for batch_idx, batch in tqdm(
                enumerate(loader),
                desc="Batch processing ...",
                colour="blue"
            ):
                global_step = (epoch * len(loader)) + batch_idx
                (images, Twc, K) = (
                    batch["image"].to(cfg.gaussian_config.device),
                    batch["viewmat_w2c"].to(cfg.gaussian_config.device),
                    batch["intrinsics"].to(cfg.gaussian_config.device)
                )
                images = transform(images)
                neural_gs = pc.generate_splats()
                neural_gs.xyz.retain_grad()
                render_pkg = pc(Twc, K, gs=neural_gs, scale_modifier=1e-2)
                print(render_pkg["rgb"].min(), render_pkg["rgb"].mean(), render_pkg["rgb"].max())
                import matplotlib.pyplot as plt
                _, axis = plt.subplots()
                axis.imshow(render_pkg["rgb"][0, ...].permute(1, 2, 0).detach().cpu())
                plt.show()

                losses = criterion(images, render_pkg["rgb"])
                losses["total"].backward()
                densifier.step(global_step, 
                                xyz_full=neural_gs.xyz, 
                                opacities=neural_gs.opacities)
                pc.optimizer.step()
                pc.optimizer.zero_grad()
                print(global_step, losses["total"])
                if wandb.run is not None:
                    wandb.log({k: v.item() 
                            for (k, v) in losses.items()}, 
                            step=global_step)
    except KeyboardInterrupt:
        _save_weights(pc)
        wandb.finish()
    finally:
        _save_weights(pc)
        wandb.finish()


__TRAINERS__ = {"3D": (train3D, Training3DConfig)}

def create_config(trainer: Literal["3D", "4D"], path: str=None):
    path = path if path is not None else f"config_{trainer}.yaml"
    cfg = __TRAINERS__[trainer][-1](logdir=None)
    cfg.to_ommega_conf(path)

def train(cfg: str | Union[Training3DConfig]):
    if isinstance(cfg, str):
        if os.path.exists(cfg):
            cfg = OmegaConf.load(cfg)
            trainer_type = cfg["trainer"]
            (trainer, cfg_cls) = __TRAINERS__[trainer_type]
            cfg_cls = cfg_cls(**cfg)
            trainer(cfg_cls)
    elif isinstance(cfg, (Training3DConfig)):
        trainer, _ = __TRAINERS__[cfg.trainer]
        trainer(cfg)

    
if __name__ == "__main__":
    # create_config(trainer="3D")
    cfg = Training3DConfig.from_yaml("config_3D.yaml")
    print(cfg.trainer)
    train(cfg)

    # from ..scene.visualize import visualize
    # pc = GsModule.load_from_checkpoint("/home/ramzan/Desktop/projects/pddm4D/test_training/training.ckpt")
    # gs = pc.generate_splats()
    # gs.xyz *= 1e+2
    # for attrib in fields(gs):
    #     name = attrib.name
    #     values = getattr(gs, name)
    #     print(name, values.min(), values.mean(), values.max(), values.shape)

    # visualize(gs, cmap="turbo")

    