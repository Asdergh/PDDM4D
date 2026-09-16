import torch as th
import torch.nn as nn
import torch.nn.functional as Fn
import numpy as np
import math
import os

from inspect import signature
from functools import cached_property
from torchtyping import TensorType
from typing import (Optional, Any, Tuple, List, Dict)
from torch.optim import Adam, SGD
from dataclasses import (dataclass, field)
from incremental_pca_torch import IncrementalPCA
from warnings import warn
from gsplat import rasterization
import random as rd
from kornia.geometry.conversions import quaternion_to_rotation_matrix
from sklearn.neighbors import NearestNeighbors

from plyfile import (PlyData, PlyElement)
from tqdm import tqdm
import logging


@dataclass
class GsModelOutput:
    xyz:            Optional[th.Tensor | np.ndarray]=None
    opacities:      Optional[th.Tensor | np.ndarray]=None
    colors:         Optional[th.Tensor | np.ndarray]=None
    rotations:      Optional[th.Tensor | np.ndarray]=None
    scales:         Optional[th.Tensor | np.ndarray]=None
    covariances:    Optional[th.Tensor | np.ndarray]=None

    def __post_init__(self):
        if self.covariances is None:
            if (self.scales is not None) \
            and (self.rotations is not None):
                self.covariances = self._covar()

    def _covar(self):
        S = th.sqrt(th.diag_embed(self.scales))
        R = quaternion_to_rotation_matrix(self.rotations)
        M = (R @ S)
        return M @ M.transpose(-1, -2)

    def to_numpy(self) -> 'GsModelOutput':
        attributes = {}
        for name, _ in signature(self.__init__)\
        .parameters.items():
            values = getattr(self, name)
            # if name == "scales":
                # values *= 1e+1
            attributes[name] = values.cpu().detach().numpy()
            print(name, values.min(), values.mean(), values.max())
        return GsModelOutput(**attributes)

    def downsample(self, size: int=10):
        attributes = {}
        for name, _ in signature(self.__init__)\
        .parameters.items():
            values = getattr(self, name)
            # if name == "scales":
                # values *= 1e+1
            attributes[name] = values[::size, :]
        return GsModelOutput(**attributes)

    def __len__(self) -> int:
        return self.xyz.shape[0]


def _normalize(x: th.Tensor, a: float, b: float):
    return a + (((x - x.min()) * (b - a)) \
                / (x.max() - x.max()))

@dataclass
class GsModelConfig:
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
    mlp_colors_lr:              float=0.01
    mlp_opacities_lr:           float=0.01
    mlp_scales_lr:              float=0.01
    mlp_rotations_lr:           float=0.01
    opacity_trashold:           float=0.01
    densify_from:               int=100
    density_every:              int=100
    densify_until:              int=3000
    opacity_trashold:           float=0.01
    xyz_gradient_trashold:      float=0.01
    anchor_distance_trashold:   float=0.01
    optimizer_type:             str="adam"
    near_plane:                 float=1e-2
    far_plane:                  float=1e10
    device:                     str="cuda"
    raster_width:               int=224
    raster_height:              int=224
    heads_drop_rate:            float=0.23

    
class GsModule(nn.Module):
    attributes: Dict[str, int] = {"feats":      (None, None),
                                "anchors":      (3, None),
                                "offsets":      (None, None),
                                "opacities":    (1, "sigmoid"),
                                "colors":       (3, "sigmoid"),
                                "scales":       (3, "softplus"),
                                "rotations":    (4, "tanh")}
    def __init__(self, config: GsModelConfig=None, device: str="cuda"):
        super(GsModule, self).__init__()
        self.cfg = config
        if self.cfg is None:
            self.cfg = GsModelConfig()
        self.device = device

    def set_scale_bounds(self, a, b):
        a = max(a, 1e-5)
        self.scale_bounds = dict(min=a, max=b,
                                min_log=math.log(a),
                                max_log=math.log(b))

    def _make_learnable(self, x: th.Tensor):
        """convert simple th.Tensor into learnable param"""
        return nn.Parameter(x.to(self.device).requires_grad_(True))

    def _get_activation(sefl, name: str):
        "get activation function for mlp_head"
        _act_cls__ = dict(sigmoid=nn.Sigmoid,
                            tanh=nn.Tanh,
                            softmax=nn.Softmax,
                            softplus=nn.Softplus,
                            relu=nn.ReLU,
                            gelu=nn.GELU)[name]
        return _act_cls__(dim=-1) if name == "softmax" else _act_cls__() 
        
    def _configure_heads(self):
        """Anchor features projection head 
        initialization function."""
        for attrib, info in self.attributes.items():
            if attrib not in ["feats", "offsets", "anchors"]:
                dim = info[0]                                      \
                    + (self.cfg.mlp_use_norms)*self.cfg.offsets_n   \
                    + (self.cfg.mlp_use_pca_projections) * self.cfg.offsets_n
                setattr(self, f"_mlp_{attrib}", nn.Sequential(nn.Linear(
                    self.cfg.feat_dim, 
                    self.cfg.offsets_n*dim
                ),
                    nn.LayerNorm(self.cfg.offsets_n*dim),
                    nn.Dropout(p=self.cfg.heads_drop_rate),
                    self._get_activation(info[1])).to(self.device))

    def _configure_optimizer(self):
        """Optimizer Configuration."""
        self.optimizer = Adam(params=[
            {"params": getattr(self, f"_{attrib}"), 
            "name": attrib,
            "lr": getattr(self.cfg, f"{attrib}_lr")}
            for attrib in ["feats", "anchors", "offsets"]
        ] + [
            {"params": getattr(self, f"_mlp_{attrib}").parameters(),
            "name": f"mlp_{attrib}",
            "lr": getattr(self.cfg, f"mlp_{attrib}_lr")}
            for attrib in self.attributes
            if attrib not in ["feats", "anchors", "offsets"]
        ])

    def get_scene_diag(self, xyz):
        """Calculates the diagonal of 
        input PointCloud bounding box."""
        min = xyz.min(dim=0).values
        max = xyz.max(dim=0).values
        diag = th.linalg.norm(max - min)
        return diag 

    def get_scene_dists(self, xyz, 
                            chunk_portion: float=0.1, 
                            mode="min", 
                            k: int=5,
                            return_tensors: str="pt",
                            min_n: int=10000):

            #TODO add outlier removal to avoid numerical instabbilities
            _modes = dict(min=1, max=-1)
            assert mode in _modes
            n = xyz.shape[0]
            xyz = xyz                               \
                    if isinstance(xyz, np.ndarray)  \
                    else xyz.detach().cpu().numpy()
            chunk_size = max(1, int((n * chunk_portion)))
            chunks_n = int(n // chunk_size)
            (min_dist, max_dist) = (float("inf"), float("-inf"))
            def _handle_chunk(xyz_chunk, min_d, max_d):
                nn_search = NearestNeighbors(n_neighbors=k)
                nn_search.fit(xyz_chunk)
                distances, indices = nn_search.kneighbors(xyz_chunk)
                indices = indices[:, _modes[mode]]
                xyz_closest = xyz_chunk[indices]

                dist = xyz_chunk - xyz_closest
                ani_dist = np.abs(dist)
                euclid_dist = np.linalg.norm(dist, axis=-1)
                min_d = min(min_d, distances[:, _modes[mode]].min())
                max_d = max(max_d, distances[:, _modes[mode]].max())
                return dict(ani_dists=ani_dist,
                            euclid_dists=euclid_dist,
                            min_d=min_d, max_d=max_d)

            results = dict(ani_dists=[], euclid_dists=[])
            def _manage_chunk(sidx: int, e_idx: int, min_d: float, max_d: float):
                xyz_chunk = xyz[sidx: e_idx, :]
                chunk_dists = _handle_chunk(xyz_chunk, min_dist, max_dist)
                for name in results.keys():
                    results[name].append(chunk_dists[name])
                min_d = chunk_dists["min_d"]
                max_d = chunk_dists["max_d"]
                return (min_d, max_d)

            if n > min_n:
                for cidx in tqdm(range(chunks_n), desc="reading scene meter..."):
                    (min_dist, max_dist) =  _manage_chunk(cidx*chunk_size, (cidx + 1)*chunk_size, min_dist, max_dist)
                if (chunks_n*chunk_size) < n:
                    (min_dist, max_dist) = _manage_chunk(chunks_n*chunk_size, n, min_dist, max_dist)
            else:
                (min_dist, max_dist) = _manage_chunk(0, n, min_dist, max_dist)

            for (name, value) in results.items():
                value = np.concatenate(value)
                if return_tensors == "pt":
                    value = th.from_numpy(value).float()
                results[name] = value
            results.update({"min_dist": min_dist, "max_dist": max_dist})
            return results

    def setup_from_pts(self, xyz: TensorType["N", "xyz"],
                        rgb: TensorType["N", "rgb"]):
        """Model initialization function. """
        n = xyz.shape[0]
        dists = self.get_scene_dists(xyz, mode="min", return_tensors="pt")
        self.set_scale_bounds(dists["min_dist"], dists["max_dist"])
        self.scene_scale = self.get_scene_diag(xyz)
        
        self._feats = self._make_learnable(th.zeros(n, self.cfg.feat_dim))
        self._anchors = self._make_learnable(xyz)
        self._offsets = self._make_learnable(th.zeros(n, self.cfg.offsets_n, 3))
        
        self._configure_heads()
        self._configure_optimizer()

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

        features = self._feats.clone()
        anchors = self._anchors.clone()
        xyz = anchors.view(-1, 1, 3) + self._offsets
        dirs = xyz - anchors.view(-1, 1, 3)
        dirs = (dirs / th.linalg.norm(dirs, dim=-1, keepdims=True))

        def _add_features(features: th.Tensor):
            """add dir normals and pca projections to anchor features"""
            add_features = features.clone()
            if self.cfg.mlp_use_norms:
                norms = th.linalg.norm(dirs, dim=-1)
                add_features = th.cat([add_features, norms], dim=-1)
            if self.cfg.mlp_use_pca_projections:
                ipca = IncrementalPCA(n_components=3,
                                        batch_size=256,
                                        device=features.device)
                ipca.fit(features)
                features_pca = ipca.transform(features)
                pca_projections = (features_pca.view(-1, 1, 3) * self._offsets).sum(dim=-1)
                add_features = th.cat([add_features, pca_projections], dim=-1)
            return add_features

        features = _add_features(features)
        attributes = dict(xyz=xyz.view(-1, 3), **{
            attrib: getattr(self, f"_mlp_{attrib}")(features).view(-1, info[0])
            for attrib, info in self.attributes.items()
            if attrib not in ["feats", "anchors", "offsets"]
        })
        attributes["scales"] = th.exp(_normalize(attributes["scales"],
                                                a=self.scale_bounds["min_log"],
                                                b=self.scale_bounds["max_log"]))
        return GsModelOutput(**attributes)

    def read_ply(self, path: str):
        """Anchor Gs format ply file reader"""
        if os.path.exists(path):
            data = PlyData.read(path)["vertex"]
            self._anchors = self._make_learnable(
                th.from_numpy(np.stack(
                    [
                        data["x"], 
                        data["y"],
                        data["z"]
                    ], axis=-1
                )).float()
            )

            print([p.name for p in data.properties])
            feats = th.from_numpy(
                np.stack([data[p.name] 
                        for p in data.properties 
                        if "anchor_feats" in p.name], 
                        axis=-1)
            ).float()
            print(feats.shape)
            assert feats.shape[-1] == self.cfg.feat_dim, \
            ("wrong ply content format for current model configuration. \n"
            f"Current {self.cfg.feat_dim=} != loaded feat_dim:={feats.shape[1]}.")
            self._feats = self._make_learnable(feats)

            offsets = th.from_numpy(
                np.stack([data[p.name]
                        for p in data.properties
                        if "anchor_offsets" in p.name],
                        axis=-1)
            ).float()
            assert offsets.shape[-1] == self.cfg.offsets_n*3, \
            ("wrong ply content format for current model configuration. \n"
            f"Current {self.cfg.offsets_n=} != loaded offsets_dim:={offsets.shape[1]}.")
            self._offsets = self._make_learnable(offsets.view(-1, self.cfg.offsets_n, 3))

            xyz = self._anchors.clone()
            self.scene_scale = self.get_scene_diag(xyz)
            scale_params = self.get_scene_dists(xyz)
            self.set_scale_bounds(a=scale_params["min_dist"], 
                                    b=scale_params["max_dist"])
            self._configure_heads()
            self._configure_optimizer()

    def write_ply(self, path: str):
        """Anchor Gs format ply file writer"""
        if not hasattr(self, "_anchors"):
            raise RuntimeError("can not write empty GsModule into ply file")
        else:
            n = self._anchors.shape[0]
            feats_dtypes = list((f"anchor_feats_{i}", "f4") for i in range(self._feats.shape[0]))
            feats_values = self._feats.clone().unbind(dim=-1)
            feats_elements = dict(zip(feats_dtypes, feats_values))

            offsets_dtypes = list((f"anchor_offsets_{i}_{j}", "f4") 
                                    for i in range(self.cfg.offsets_n)
                                    for j in range(3))
            offsets_values = self._offsets.clone().view(n, -1).unbind(dim=-1)
            offsets_elements = dict(zip(offsets_dtypes, offsets_values))

            anchors_dtypes = [("x", "f4"), ("y", "f4"), ("z", "f4")]
            anchors_values = self._anchors.clone().unbind(dim=-1)
            anchors_elements = dict(zip(anchors_dtypes, anchors_values))

            elements = (anchors_elements | feats_elements | offsets_elements)
            e = np.empty((n, ), dtype=list(elements.keys()))
            for (key, values) in elements.items():
                e[key[0]] = values.detach().cpu().numpy()

            e = PlyElement.describe(e, "vertex")
            PlyData([e]).write(path)

    @classmethod
    def load_from_checkpoint(cfg, path: str) -> 'GsModule':
        """initialize model from single checkpoint"""
        if os.path.exists(path):
            ckpt = th.load(path)
            cfg = GsModelConfig(**ckpt["hyperparams"])
            pc = GsModule(cfg, device="cpu")

            pc.read_ply(ckpt["sparse_states"])
            pc.load_state_dict(ckpt["gaussian_heads"])
            pc.optimizer.load_state_dict(ckpt["optimizer_states"])
            return pc

    def __len__(self):
        if hasattr(self, "_anchors"): return self._anchors.shape[0]
        else: warn("Gs module is not initilized")

    def forward(self, 
            extrinsics: TensorType["C", "4", "4"],
            intrinsics: TensorType["C", "3", "3"],
            far_near: Tuple[float]=(1e-2, 1e10),
            features: Optional[TensorType["N", "d"]]=None,
            gs: Optional[GsModelOutput]=None,
            scale_modifier: float=1.0):

        gs = gs if gs is not None else self.generate_splats()
        features = gs.colors if features is None else features
        (render_rgb, render_alpha, meta) = rasterization(
            means=gs.xyz,
            quats=gs.rotations,
            scales=gs.scales * scale_modifier,
            opacities=gs.opacities.squeeze(),
            colors=gs.colors,
            viewmats=extrinsics,
            Ks=intrinsics,
            near_plane=far_near[0],
            far_plane=far_near[1],
            width=self.cfg.raster_width,
            height=self.cfg.raster_height,
            
        )
        return dict(rgb=render_rgb.permute(0, 3, 1, 2),
                    alpha=render_alpha.permute(0, 3, 1, 2),
                    meta=meta)
    
class AnchorGrowing:
    def __init__(self, 
                pc: GsModule, 
                config: GsModelConfig, 
                logger: Optional[logging.Logger]=None):
        self.pc = pc
        self.cfg = config
        self.logger = logger
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
        gaccum = self._gradient_accumulator.view(-1, self.cfg.offsets_n)
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
        distances = th.linalg.norm(xyz_full.view(-1, self.cfg.offsets_n, 3) \
                        - self.pc._anchors.view(-1, 1, 3), dim=-1).view(-1)
        distances_mask = (distances > self.cfg.anchor_distance_trashold)
        opacities = opacities.view(-1, self.cfg.offsets_n).mean(dim=-1)
        self.update(xyz_grads, opacities)
        if step_idx > self.cfg.densify_from:
            if (step_idx <= self.cfg.densify_until) \
                and (step_idx % self.cfg.densify_every) == 0:

                if self.logger is not None:
                    self.logger.info(f"Points before densification: {self.pc._anchors.shape[0]}")
                grow_mask = (self._gradient_accumulator > self.cfg.xyz_gradient_trashold)
                grow_indices = th.where(distances_mask & grow_mask)[0]
                anchor_indices = grow_indices % len(self.pc)
                grow_indices = th.cat([grow_indices[anchor_indices == idx][rd.randint(0, self.cfg.offsets_n - 1)]
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
                        < self.cfg.opacity_trashold).squeeze()
                self.remove_from_optimizer(omask)
                self._opacity_accumulator *= 0
                self._gradient_accumulator *= 0
                if self.logg is not None:
                    print(self.pc._anchors.shape[0])
                    self.logger.info(f"Points after densification: {self.pc._anchors.shape[0]}")
