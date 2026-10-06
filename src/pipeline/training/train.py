import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import plyfile
import logging
import wandb
from typing import (Literal, Optional, List, Any, Union, Dict)
from ...scene import (GsModule, GsModelOutput, GsModelConfig, AnchorGrowing)
from ...data import (MipNerf360v2DatasetConfig, get_dataset)
from ...metrics import CombinedVisualLoss, CombinedVisualLossConfig
from tqdm import tqdm
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
from inspect import signature
from dataclasses import (dataclass, field, fields, is_dataclass, asdict)
import abc
from torchvision.transforms import Resize, Normalize, GaussianBlur, Compose
from omegaconf.dictconfig import DictConfig
from torchvision.utils import make_grid


def read_signature(cls, **kwargs):
    names = list(signature(cls.__init__).parameters.keys())
    kwargs = {key: value 
            for (key, value) in kwargs.items()
            if key in names}
    return kwargs

def load_cfg(cls, instance):
    if isinstance(instance, cls):
        return instance
    elif isinstance(instance, (dict, DictConfig)):
        return cls(**read_signature(cls, **instance))
    else:
        raise ValueError(f"unknow instance type: {type(instance)} \n"
                        f"for config cls: {cls}. ")

@dataclass
class SceneReconstractionConfig:
    dirpath: str
    steps: int=4000
    batch_size: int=32
    shuffle: bool=False
    wandb_verbose: bool=False
    wandb_project_name: str="anchor-gs-reconstraction"
    wandb_log_scalars: bool=False
    wandb_log_plots: bool=False
    wndb_plots_n: int=5
    wandb_log_scalars_every_step: int=5
    wandb_log_plots_evry_step: int=100
    render_scale_modifier: float=1.0
    gaussian_module_config: Dict[str, Any]=field(default_factory=lambda: dict())
    criterion_module_config: Dict[str, Any]=field(default_factory=lambda: dict())
    dataset_config: Dict[str, Any] = field(default_factory=lambda: dict(name="360v2"))
    img_transforms_kwargs: Dict[str, Any]=field(default_factory=lambda: dict())

    def __post_init__(self):
        self.gaussian_module_config = load_cfg(GsModelConfig, self.gaussian_module_config)
        self.criterion_module_config = load_cfg(CombinedVisualLossConfig, self.criterion_module_config)
        print(self.gaussian_module_config, self.criterion_module_config)

    def to_yaml(self, path: str):
        attributes = {key: value  
                    for (key, value) in asdict(self).items()
                    if key not in ["gaussian_module_config", 
                                    "criterion_module_config"]}
        attributes["gaussian_module_config"] = OmegaConf.structured(self.gaussian_module_config)
        attributes["criterion_module_config"] = OmegaConf.structured(self.criterion_module_config)
        config = OmegaConf.create(attributes)
        OmegaConf.save(config=config, f=path, resolve=True)

    @classmethod
    def load(cls, path: str):
        if os.path.exists(path):
            cfg = OmegaConf.load(path)
            cfg = OmegaConf.to_container(cfg, resolve=True)
            return cls(**cfg)
        else:
            raise FileNotFoundError(f"coudn't find config at location: {path}. ")

__VALID_IMAGE_TRANSFORMS__: List[nn.Module] = [
    Resize, 
    Normalize, 
    GaussianBlur
]
def get_gs_training_trnasforms(**kwargs):
    transforms = []
    for module in __VALID_IMAGE_TRANSFORMS__:
        args = read_signature(module, **kwargs)
        if args:
            transforms.append(module(**args))
    return Compose(transforms)

def training(cfg: SceneReconstractionConfig):
    paths = {"root": cfg.dirpath,
            "results": "training_results",
            "meta": "meta",
            "info": "meta/info_logger.log",
            "history": "meta/history"}
    for (key, path) in paths.items():
        if key == "root":
            continue
        paths[key] = os.path.join(path)
        os.makedirs(path, exist_ok=True)

    if cfg.wandb_verbose:
        wandb.init(project=cfg.wandb_project_name,
                    config=cfg,
                    dir=paths["history"])
        
    gaussian_cfg = cfg.gaussian_module_config
    target_size = (gaussian_cfg.raster_width, gaussian_cfg.raster_height)
    dataset_config = cfg.dataset_config
    if isinstance(dataset_config, dict):
        dataset_config["target_resolution"] = target_size
    elif is_dataclass(dataset_config):
        dataset_config.target_resolution = target_size

    img_transforms = get_gs_training_trnasforms(**cfg.img_transforms_kwargs)
    print(f"Custom Transofrmations for dataset: {img_transforms}")
    dataset = get_dataset(dataset_config)
    dataset.set_img_transforms(img_transforms)
    gaussian = GsModule(gaussian_cfg, device="cuda")
    gaussian.setup_from_pts(**dataset.sparse)
    densifier = AnchorGrowing(pc=gaussian, config=gaussian_cfg)

    criterion = CombinedVisualLoss(cfg.criterion_module_config).to("cuda")
    loader = DataLoader(dataset=dataset, batch_size=cfg.batch_size, shuffle=cfg.shuffle)
    stop_training = False
    step = 0
    while not stop_training:
        with tqdm(desc="Rconstraction ...",
                colour="green",
                ascii=":.:",
                total=cfg.steps) as pbar:
            for batch in tqdm(loader,
                            desc="Batches Processing ...",
                            colour="red",
                            ascii=":>"):

                (images, viewpoints, intrinsics) = list(map(lambda x: x.to("cuda"), (
                    batch["image"],
                    batch["viewmat_w2c"],
                    batch["intrinsics"]
                )))
                print(f"viewwpoints is None ?: {th.isnan(viewpoints).any()}")
                print(intrinsics[0])
                gs = gaussian.generate_splats()
                gs.xyz.retain_grad()
                render_pkg = gaussian(extrinsics=viewpoints,
                                    intrinsics=intrinsics,
                                    # far_near=(gaussian_cfg.far_plane,
                                    #             gaussian_cfg.near_plane),
                                    scale_modifier=cfg.render_scale_modifier,
                                    gs=gs)
                
                render_rgb = render_pkg["rgb"]
                losses = criterion(render_rgb, images)
                losses["total"].backward()
                densifier.step(xyz_full=gs.xyz, 
                            opacities=gs.opacities, 
                            step_idx=step)
                gaussian.optimizer.step()
                gaussian.optimizer.zero_grad()

                if cfg.wandb_verbose:
                    print(losses)
                    print(images.min(), images.mean(), images.max())
                    print(render_pkg["rgb"].min().detach(), render_pkg["rgb"].mean().detach(), render_pkg["rgb"].max().detach())
                    import matplotlib.pyplot as plt
                    _, axis = plt.subplots()
                    axis.imshow(make_grid(render_pkg["rgb"]).detach().cpu().permute(1, 2, 0))
                    plt.show()
                    if cfg.wandb_log_scalars\
                        and (step % cfg.wandb_log_scalars_every_step) == 0:
                        wandb.log({key: value.item() 
                                for (key, value) in losses.items()},
                                step=step)
                    if cfg.wandb_log_plots \
                        and (step % cfg.wandb_log_plots_evry_step) == 0:
                        plots = {key: wandb.Image(make_grid(value.detach().cpu()))
                                for (key, value) in zip(["gt", "rgb", "alpha"], 
                                                        (images, render_pkg["rgb"], render_pkg["alpha"]))}
                        wandb.log(plots, step=step)
                step += 1
                pbar.update(1)
                if step == cfg.steps:
                    stop_training = True
                    break


if __name__ == "__main__":

    config = SceneReconstractionConfig.load("config_3D.yaml")
    # transforms = get_gs_training_trnasforms(**config.img_transforms_kwargs)
    # gs = GsModule(config=config.gaussian_module_config, device="cuda")
    # dataset = get_dataset(config.dataset_config)
    # dataset.set_img_transforms(transforms)
    # sparse = dataset.sparse
    # gs.setup_from_pts(**sparse)

    # samples = 10
    # container = gs.generate_splats()
    # loader = DataLoader(dataset=dataset, batch_size=12, shuffle=True)
    # sample = next(iter(loader))

    # rendered = gs(extrinsics=sample["viewmat_w2c"].to("cuda"),
    #             intrinsics=sample["intrinsics"].to("cuda"),
    #             gs=container)
    # images = rendered["rgb"].detach().cpu()
    # image_grid = make_grid(images).permute(1, 2, 0)

    # import rerun as rr
    # rr.init("origin", spawn=True)
    # from ...data.vis import log_views, log_pts

    # log_views("origin/Frames", 
    #         viewmats=sample["viewmat_w2c"],
    #         Ks=sample["intrinsics"],
    #         images=images.permute(0, 2, 3, 1))
    # log_pts("origin/3D", xyz=sparse["xyz"], cmap=sparse["rgb"])

    # import matplotlib.pyplot as plt
    # _, axis = plt.subplots()
    # axis.imshow(image_grid)
    # plt.show()

    training(config)