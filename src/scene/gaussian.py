import torch as th
import torch.nn as nn
import torch.nn.functional as Fn

from torchtyping import TensorType
from typing import (Optional, Any, Tuple, List)
from torch.optim import Adam, SGD
from dataclasses import (dataclass, field)
from incremental_pca_torch import IncrementalPCA
from warnings import warn




@dataclass
class GsModelOutput:
    xyz:            Optional[th.Tensor]=None
    opacities:      Optional[th.Tensor]=None
    colors:         Optional[th.Tensor]=None
    rotations:      Optional[th.Tensor]=None
    scales:         Optional[th.Tensor]=None
    covariences:    Optional[th.Tensor]=None


@dataclass
class GsSModelConfig:
    feat_dim:                   int=32
    offsets_n:                  int=6
    mlp_use_norms:              bool=False
    mlp_use_pca_projections:    bool=False
    colors_lr:                  float=0.03
    xyz_lr:                     float=0.01
    features_lr:                float=0.01
    opacities_lr:               float=0.01
    scales_lr:                  float=0.04
    rotations_lr:               float=0.04
    opacity_trashold:           float=0.01
    densify_from:               int=1000
    density_every:              int=100
    densify_until:              int=3000
    opacity_trashold:           float=0.01
    xyz_gradient_trashold:      float=0.01
    
    
class GsModule(nn.Module):
    attributes: List[str] = ["feats", "anchors",
                                "offsets", "opacities",
                                "colors", "scales", 
                                "rotations"]
    def __init__(self, config: GsSModelConfig):
        super(GsModule, self).__init__()
        self.config = config

    def _configure_heads(self):
        """Anchor features projection head 
        initialization function."""
        for (head, chanels, normalize) in (("scales",       3, False), 
                                            ("opacities",   1, True), 
                                            ("colors",      3, True), 
                                            ("rotations",   4, False)):
            dims = self.config.feat_dim + chanels
            dims = (dims if not self.config.mlp_use_norms else dims + self.config.offsets_n)
            dims = (dims if not self.config.mlp_use_pca_projections else dims + self.config.offsets_n * 3)
            module = nn.Sequential(nn.Linear(dims, self.config.offsets_n*chanels),
                                    (nn.Identity() if not normalize else nn.LayerNorm()))
            setattr(self, f"_mlp_{head}", module)

    def _configure_optimizer(self):
        """Optimizer Configuration."""
        self.optimizer = Adam(p=[{"params": getattr(self, f"_{attrib}"), 
                        "name": attrib,
                        "lr": getattr(self.config, f"{attrib}_lr")}
                        for attrib in self.attributes])
        
    def get_scene_diag(self, xyz):
        """Calculates the diagonal of 
        input PointCloud bounding box."""
        min = xyz.min(dim=0)
        max = xyz.max(dim=0)
        diag = th.linalg.norm(max - min)
        return diag 

    def get_scene_dist(self, xyz, chunk_portion: float=0.1, mode="min"):
        """Calculates min/max distances betwen points 
        in input PointCloud."""
        assert mode in ["min", "max"]
        optimal_dist = float("inf") if mode == "min" else float("-inf")
        chunk_size = int((xyz.shape[0] * chunk_portion) / 10)
        chunks_n = int(xyz.shape[0] * chunk_portion)
        for cidx in range(chunks_n):
            chunk = xyz[cidx*chunk_size: (cidx + 1)*chunk_size, :]
            diffs = chunk.view(-1, 1, 3) - chunk.vieww(1, -1, 3)
            norms = th.linalg.norm(diffs, dim=-1)
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
        min_dist = self.get_scene_dist(xyz, "min")
        self.scene_scale = self.get_scene_diag(xyz)

        self._feats = nn.Parameter(th.zeros(n, self.config.feat_dim))
        self._anchors = nn.Parameter(xyz)
        self._offsets = nn.Parameter(th.zeros(n, self.config.offsets_n, 3))
        self._scales = nn.Parameter(th.ones(n, 3) * min_dist)
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
        anchors = self._anchor
        xyz = anchors.view(-1, 1, 3) + self._offsets
        dirs = xyz - anchors.view(-1, 1, 3)
        dirs = (dirs / th.linalg.norm(dirs, dim=-1))

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
        attributes = {"xyz": xyz}.update({name: project_features(name) 
                                        for name in ["colors", 
                                                    "opacities", 
                                                    "scales", 
                                                    "rotations"]})
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
        opacities = opacities.view(-1, self.config.offsets_n).mean(dim=-1)
        self.update(xyz_grads, opacities)
        if step_idx > self.config.densify_from:
            if (step_idx <= self.config.densify_until) \
                and (step_idx % self.config.densify_every) == 0:
                grow_indices = th.where(self._gradient_accumulator 
                                        > self.config.xyz_gradient_trashold)[0]
                grow_indices = th.unique(grow_indices % len(self.pc))
                # xyz = xyz_full[grow_indices]
                xyz = self.pc._anchors[grow_indices]
                anchor_mask = th.zeros((len(self.pc),), 
                                    dtype=th.bool, 
                                    device=self.pc._anchors.device)
                anchor_mask[grow_indices] = True
                self.grow_optimizer(xyz, anchor_mask)

                omask = (self._opacity_accumulator 
                        < self.config.opacity_trashold).squeeze()
                self.remove_from_optimizer(omask)
        


