import torch as th
import torch.nn as nn
import torch.nn.functional as Fn
import numpy as np
from inspect import signature

from functools import cached_property
from torchtyping import TensorType
from typing import (Optional, Any, Tuple, List)
from torch.optim import Adam, SGD
from dataclasses import (dataclass, field)
from incremental_pca_torch import IncrementalPCA
from warnings import warn
from gsplat import rasterization
import random as rd
from kornia.geometry.conversions import quaternion_to_rotation_matrix



@dataclass
class GsModelOutput:
    xyz:            Optional[th.Tensor | np.ndarray]=None
    opacities:      Optional[th.Tensor | np.ndarray]=None
    colors:         Optional[th.Tensor | np.ndarray]=None
    rotations:      Optional[th.Tensor | np.ndarray]=None
    scales:         Optional[th.Tensor | np.ndarray]=None
    covariences:    Optional[th.Tensor | np.ndarray]=None

    def __post_init__(self):
        if self.covariences is None:
            if (self.scales is not None) \
            and (self.rotations is not None):
                self.covariences = self._covar()

    def _covar(self):
        S = th.diag_embed(self.scales)
        R = quaternion_to_rotation_matrix(self.rotations)
        M = (R @ S)
        return M @ M.transpose(-1, -2)

    def to_numpy(self) -> 'GsModelOutput':
        attributes = {}
        for name, _ in signature(self.__init__)\
        .parameters.items():
            values = getattr(self, name)
            attributes[name] = values.cpu().detach().numpy()
            print(name, values.mean())
        return GsModelOutput(**attributes)
        
    


@dataclass
class GsSModelConfig:
    feat_dim:                   int=32
    offsets_n:                  int=6
    mlp_use_norms:              bool=False
    mlp_use_pca_projections:    bool=False
    colors_lr:                  float=0.03
    anchors_lr:                 float=0.01
    offsets_lr:                 float=0.01
    feats_lr:                   float=0.01
    opacities_lr:               float=0.01
    scales_lr:                  float=0.04
    rotations_lr:               float=0.04
    opacity_trashold:           float=0.01
    densify_from:               int=1000
    density_every:              int=100
    densify_until:              int=3000
    opacity_trashold:           float=0.01
    xyz_gradient_trashold:      float=0.01
    anchor_distance_trashold:   float=0.01
    optimizer_type:             str="adam"
    near_plane:                 float=1e-2
    far_plane:                  float=1e10
    
    
    
class GsModule(nn.Module):
    attributes: List[str] = ["feats", 
                            "anchors",
                            "offsets", "opacities",
                            "colors", "scales", 
                            "rotations"]
    def __init__(self, config: GsSModelConfig):
        super(GsModule, self).__init__()
        self.config = config

    def _configure_heads(self):
        """Anchor features projection head 
        initialization function."""
        for (head, channels, normalize) in (("scales",       3, False), 
                                            ("opacities",   1, True), 
                                            ("colors",      3, True), 
                                            ("rotations",   4, False)):
            dims = self.config.feat_dim + channels
            dims = (dims if not self.config.mlp_use_norms else dims + self.config.offsets_n)
            dims = (dims if not self.config.mlp_use_pca_projections else dims + self.config.offsets_n * 3)
            print(dims)
            module = nn.Sequential(nn.Linear(dims, self.config.offsets_n*channels),
                                    (nn.Identity() if not normalize else nn.LayerNorm(self.config.offsets_n*channels)),
                                    (nn.Identity() if head != "opacities" else nn.Sigmoid()))
            setattr(self, f"_mlp_{head}", module)

    def _configure_optimizer(self):
        """Optimizer Configuration."""
        self.optimizer = Adam(params=[{"params": getattr(self, f"_{attrib}"), 
                        "name": attrib,
                        "lr": getattr(self.config, f"{attrib}_lr")}
                        for attrib in self.attributes])

    
    def get_scene_diag(self, xyz):
        """Calculates the diagonal of 
        input PointCloud bounding box."""
        min = xyz.min(dim=0).values
        max = xyz.max(dim=0).values
        diag = th.linalg.norm(max - min)
        return diag 

    def get_scene_dist(self, xyz, chunk_portion: float=0.1, mode="min"):
        """Calculates min/max distances betwen points 
        in input PointCloud."""
        assert mode in ["min", "max"]
        optimal_dist = float("inf") if mode == "min" else float("-inf")
        chunk_size = max(1, int((xyz.shape[0] * chunk_portion)))
        chunks_n = int(xyz.shape[0] * chunk_portion)
        for cidx in range(chunks_n):
            chunk = xyz[cidx*chunk_size: (cidx + 1)*chunk_size, :]
            diffs = chunk.view(-1, 1, 3) - chunk.view(1, -1, 3)
            norms = th.linalg.norm(diffs, dim=-1)
            norms = norms[norms != 0.0]
            if norms.numel() != 0:
                if mode == "min":
                    min_norm = norms.min()
                    if min_norm < optimal_dist: optimal_dist = min_norm
                else:
                    max_norm = norms.max()
                    if max_norm > optimal_dist: optimal_dist = max_norm

        return optimal_dist
    
    def setup_model(self, xyz: TensorType["N", "xyz"],
                        rgb: TensorType["N", "rgb"]):
        """Model initialization function. """
        n = xyz.shape[0]
        min_dist = self.get_scene_dist(xyz, mode="min")
        print(min_dist)
        self.scene_scale = self.get_scene_diag(xyz)

        self._feats = nn.Parameter(th.zeros(n, self.config.feat_dim))
        self._anchors = nn.Parameter(xyz)
        self._offsets = nn.Parameter(th.zeros(n, self.config.offsets_n, 3))
        self._scales = nn.Parameter(th.ones(n, 3) * th.log(min_dist + 1e-5))
        self._rotations = nn.Parameter(th.zeros(n, 4))
        self._colors = nn.Parameter(rgb)
        self._opacities = nn.Parameter(th.zeros(n, 1))
        
        self._configure_optimizer()
        self._configure_heads()

    def generate_splats(self):
        """Neural Gaussian Generation function with PCA projection. 
        This implementation differese from initial version from: 
        https://github.com/city-super/Scaffold-GS. 
        
        Instead of letting model to analize the correlations 
        between viepwoints directions thereby restricting it 
        to a specific view generation, this implementation 
        looks for projections of anchor features 
        itself according to the offsets directions, letting 
        model to be more independet of viewpoints and become 
        more geometricaly sastainable"""

        features = self._feats
        anchors = self._anchors
        xyz = anchors.view(-1, 1, 3) + self._offsets
        dirs = xyz - anchors.view(-1, 1, 3)
        dirs = (dirs / th.linalg.norm(dirs, dim=-1, keepdims=True))

        dirs_feats = None
        if self.config.mlp_use_norms:
            dirs_feats = th.linalg.norm(dirs, dim=-1)
        if self.config.mlp_use_pca_projections:
            ipca = IncrementalPCA(n_components=3,
                                    batch_size=256,
                                    device=features.device)
            ipca.fit(features)
            features_pca = ipca.transform(features)
            features_pca = (features_pca.view(-1, 1, 3) * self._offsets).sum(dim=-1)
            dirs_feats = (dirs_feats if dirs_feats is not None else features)
            dirs_feats = th.cat([dirs_feats, features_pca], dim=-1)

        project_features = lambda name: \
            getattr(self, f"_mlp_{name}")(th.cat([features, getattr(self, f"_{name}")], dim=-1) 
                                        if dirs_feats is None 
                                        else th.cat([features, getattr(self, f"_{name}"), dirs_feats])).view(-1, 3)
        attributes = {"xyz": xyz.view(-1, 3)}
        attributes.update({name: project_features(name).view(-1, channels)
                            for (name, channels) in [("colors", 3), 
                                                    ("opacities", 1), 
                                                    ("scales", 3), 
                                                    ("rotations", 4)]})
        attributes["scales"] = th.exp(attributes["scales"])
        return GsModelOutput(**attributes)

    def __len__(self):
        if hasattr(self, "_anchors"): return self._anchors.shape[0]
        else: warn("Gs module is not initilized")

class AnchorGrowing:
    def __init__(self, pc: GsModule, config: GsSModelConfig):
        self.pc = pc
        self.config = config
        self._opacity_accumulator = None
        self._gradient_accumulator = None


    def update(self, xyz_gradients, anchor_avg_opacity):
        self._opacity_accumulator = (anchor_avg_opacity 
                                    if self._opacity_accumulator is None 
                                    else self._opacity_accumulator + anchor_avg_opacity)
        self._gradient_accumulator = (xyz_gradients
                                    if self._gradient_accumulator is None
                                    else self._gradient_accumulator + xyz_gradients)
        
    def _update_after_remove(self, mask):
        for name in self.pc.attributes:
            values = getattr(self.pc, f"_{name}")
            values = values[~mask]
            setattr(self.pc, f"_{name}", values)
        self._opacity_accumulator = self._opacity_accumulator[~mask]
        gaccum = self._gradient_accumulator.view(-1, self.config.offsets_n)
        gaccum = gaccum[~mask]
        self._gradient_accumulator = gaccum
    
    def remove_from_optimizer(self, mask: th.BoolTensor, 
                                    attributes: List[str]=["_feats", "_anchors",
                                                            "_offsets", "_colors",
                                                            "_opacities", "_scales",
                                                            "_rotations"]):
        def _handle_group(group: dict):
            params = group.get("params", None)
            if params is not None:
                params = params[~mask]
                group["params"] = nn.Parameter(params)
            state = group.get("state", None)
            if state is not None:
                state["exp_avg"] = state["exp_avg"][~mask]
                state["exp_avg_sq"] = state["exp_avg_sq"][~mask]
                

        for group in self.pc.optimizer.param_groups:
            name = group.get("name", None)
            if (name is not None) and (name in attributes):
                _handle_group(group)

        self._update_after_remove(mask)


    def _update_after_grow(self, anchors_mask):
        for name in self.pc.attributes:
            values = getattr(self.pc, f"_{name}")
            values = th.cat([values, values[anchors_mask]], dim=0)
            setattr(self.pc, f"_{name}", values)
        self._opacity_accumulator = th.cat([self._opacity_accumulator,
                                            self._opacity_accumulator[anchors_mask]],
                                            dim=0)
        self._gradient_accumulator = th.cat([self._gradient_accumulator,
                                            self._gradient_accumulator[anchors_mask]],
                                            dim=0)
        
    def grow_optimizer(self, xyz, anchors_mask):
        def _handle_group(group, inputs: Optional[th.Tensor]=None):
            params = group.get("params", None)
            if params is not None:
                inputs = inputs \
                            if inputs is not None \
                            else params[anchors_mask]
                params = th.cat([params, inputs], dim=0)
                group["params"] = nn.Parameter(params)
            state = group.get("state", None)
            if state is not None:
                state["exp_avg"] = th.cat([state["exp_avg"], 
                                            state["exp_avg"][anchors_mask]],
                                            dim=0)
                state["exp_avg_sq"] = th.cat([state["exp_avg_sq"], 
                                            state["exp_avg"][anchors_mask]],
                                            dim=0)

        for group in self.pc.optimizer.param_groups:
            name = group.get("name", None)
            if (name is not None) and (name != "anchors"):
                _handle_group(group)
            elif name == "anchors":
                _handle_group(group, xyz)

        self._update_after_grow(anchors_mask)

    def step(self, step_idx: int,
                    xyz_full: TensorType["N", "xyz"],
                    opacities: TensorType["N", "1"]):

        xyz_grads = th.linalg.norm(xyz_full.grad, dim=-1)
        distances = th.linalg.norm(xyz_full.view(-1, self.config.offsets_n, 3) \
                        - self.pc._anchors.view(-1, 1, 3), dim=-1).view(-1)
        distances_mask = (distances > self.config.anchor_distance_trashold)
        opacities = opacities.view(-1, self.config.offsets_n).mean(dim=-1)
        self.update(xyz_grads, opacities)
        if step_idx > self.config.densify_from:
            if (step_idx <= self.config.densify_until) \
                and (step_idx % self.config.densify_every) == 0:

                grow_mask = (self._gradient_accumulator > self.config.xyz_gradient_trashold)
                grow_indices = th.where(distances_mask & grow_mask)[0]
                anchor_indices = grow_indices % len(self.pc)
                grow_indices = th.cat([grow_indices[anchor_indices == idx][rd.randint(0, self.config.offsets_n - 1)]
                                    for idx in th.unique(anchor_indices)], 
                                    dim=0)
                
                xyz = xyz_full[grow_indices]
                # xyz = self.pc._anchors[grow_indices]
                anchor_mask = th.zeros((len(self.pc),), 
                                    dtype=th.bool, 
                                    device=self.pc._anchors.device)
                anchor_mask[th.unique(anchor_indices)] = True
                self.grow_optimizer(xyz, anchor_mask)

                omask = (self._opacity_accumulator 
                        < self.config.opacity_trashold).squeeze()
                self.remove_from_optimizer(omask)
        

if __name__ == "__main__":

    from viser import ViserServer
    # from viser4d import Viser4dServer
    from ..data.datasets.mip_nerf import MipNErf360Dataset

    path = "/run/media/ramzan/T7/datasets/neuiral_rendering/360_v2"
    dataset = MipNErf360Dataset(path, scene="kitchen")

    config = GsSModelConfig()
    pc = GsModule(config)
    pc.setup_model(**dataset.sparse)

    gs = pc.generate_splats().to_numpy()
    server = ViserServer()
    server.scene.add_gaussian_splats(name="gs", centers=gs.xyz,
                                    covariances=gs.covariences,
                                    rgbs=gs.colors,
                                    opacities=gs.opacities)
    import time 
    while True:
        time.sleep(10)