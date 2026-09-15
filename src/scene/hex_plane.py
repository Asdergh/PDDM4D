import torch as th
import torch.nn as nn
import torch.nn.functional as Fn
from torch.nn.functional import grid_sample
from typing import (Tuple, Optional, Any, Callable, List, Dict)
from torchtyping import TensorType




class HexPlane(nn.Module):
    _volumes: List[str] = ["xy-zt", "xz-yt", "yz-xt"] 
    _plane_indices: Dict[str, tuple] = {
        "xy": (0, 1),
        "xz": (0, 2),
        "yz": (1, 2)
    }
    _tplane_indices: Dict[str, int] = {"xt": 0, "yt": 1, "zt": 2}
    def __init__(self, features_dim: int,
                        output_dim: Optional[int]=None,
                        grid_size: Tuple[int] | int=200,
                        time_grid_size: int=1000,
                        activation: Optional[Callable[..., nn.Module]]=None,
                        device: str="cuda",
                        initialize: bool=False):

        super(HexPlane, self).__init__()
        self.fdim = features_dim
        odim = output_dim if output_dim is not None else self.fdim
        grid_size = grid_size \
                    if isinstance(grid_size, tuple) \
                    else 3*(grid_size, )
        grid_size += (time_grid_size, )
        self.sizes = {name: grid_size[idx] 
                    for idx, name in 
                    enumerate(["x", "y", "z", "t"])}
        self.device = device
        activation: Callable[..., nn.Module] = (activation 
                                                    if activation is not None
                                                    else nn.ReLU())
        self.head = nn.Sequential(nn.Linear(3*self.fdim, odim),
                                    nn.LayerNorm(odim),
                                    activation)
        if initialize:
            self.set_planes_weights()

    def set_planes_weights(self, **kwargs):
        """(Description):
        Register or overwrite per-plane feature grids used by this HexPlane
        instance. Each keyword argument corresponds to exactly one plane
        name (either a purely spatial plane or a space-time plane), and the
        associated value is a tensor that becomes an nn.Parameter of the
        module. This method exists because HexPlane is typically created
        first (to fix feature dimensionality, grid sizes and device) and
        only later populated with actual weights, either from a checkpoint,
        from a pretrained grid, or from a manually designed initialization.
        Calling this method after __init__ allows changing the model's
        behaviour without rebuilding the whole module.

        For every plane whose name is NOT passed as a keyword argument,
        a zero-initialized grid of the correct shape is created
        automatically, so that the module is always left in a fully usable
        state after this call returns.

    (Arguments):
        kwargs: keyword arguments of the form {plane_name: weights_tensor},
                where
                    - plane_name is one of
                        spatial planes:   "xy", "xz", "yz"
                        temporal planes:  "xt", "yt", "zt"
                    The names map to axis pairs as follows:
                        "xy" -> (x, y),  "xz" -> (x, z),  "yz" -> (y, z),
                        "xt" -> (x, t),  "yt" -> (y, t),  "zt" -> (z, t)
                    - weights_tensor is a tensor of shape
                        (fdim, size_a, size_b)
                        where
                            fdim   = self.fdim (feature dimensionality),
                            size_a = self.sizes[first letter of plane_name],
                            size_b = self.sizes[second letter of plane_name].
                    The tensor is moved to self.device and wrapped into
                    an nn.Parameter of shape (1, fdim, size_a, size_b)
                    (an extra leading batch dimension is required by
                    torch.nn.functional.grid_sample).

    (Raises):
        ValueError: if a provided tensor's shape does not match the
                    expected (fdim, size_a, size_b) for its plane.
                    The error message reports both the observed and the
                    expected shape to simplify debugging.
        KeyError:   if an unknown plane name is passed (e.g. due to a
                    typo such as "zx" instead of "xz").

    (Notes):
        - The expected tensor size for a given plane depends on the
        current self.sizes dictionary, which stores the resolution of
        every axis (x, y, z, t). Before allocating a weights tensor,
        make sure it matches:
            self.sizes["x"], self.sizes["y"],
            self.sizes["z"], self.sizes["t"]
            For example, if self.sizes == {"x": 200, "y": 200, "z": 200,
            "t": 1000} and self.fdim == 16, then the tensor for plane "zt"
            must have shape (16, 200, 1000).
        - Passing a subset of planes is allowed; the remaining planes are
        filled with zeros. This is convenient for ablations or for
        partially freezing a model.
        - Repeated calls overwrite the previously registered parameters
        for the provided planes and leave the untouched planes intact.
        - To satisfy grid_sample's input requirements, weights are always
        stored with a leading singleton dimension: the actual Parameter
        visible to the optimizer has shape (1, fdim, size_a, size_b).
        - Typical usage:
            hexplane = HexPlane(features_dim=16, grid_size=200,
                                time_grid_size=1000, device="cuda")
            hexplane.set_planes_weights(
                xy=torch.randn(16, 200, 200),
                zt=torch.randn(16, 200, 1000),
            )  # the other four planes are set to zero automatically
        """
        def _get_size(name: str):
            target_size = tuple(map(lambda k: self.sizes[k], list(name)))
            target_size = (self.fdim, ) + target_size
            return target_size

        def _set_values(weights: Dict[str, th.Tensor]):
            planes = []
            for (key, values) in weights.items():
                if(key in self._plane_indices) \
                    or (key in self._tplane_indices):
                    print(key)
                    target_size = _get_size(key)
                    if (values.shape != target_size):
                        raise ValueError(f"Input tensor for {key} plane " \
                                        f"has wrong size: {values.shape[-1]}")
                    else:
                        setattr(self, f"_{key}_plane", 
                                nn.Parameter(values\
                                                .unsqueeze(dim=0)\
                                                .to(self.device)\
                                                .requires_grad_(True)))
                    planes.append(key)
            return planes
        
        planes = _set_values(kwargs)
        if len(planes) != (len(self._plane_indices) \
                            + len(self._tplane_indices)):
            planes = {
                name: th.zeros(_get_size(name))
                for name in list(self._plane_indices.keys()) 
                            + list(self._tplane_indices.keys())
                if (name not in planes) \
                    and not hasattr(self, f"_{name}_plane")
            }
            _ = _set_values(planes)


    def normalize(self, values, a, b):
        """Normalizes values from a := min
        b := max to [0, 1]"""
        return 2 * ((values - a) / (b - a)) - 1
    
    def forward(self, 
                    xyz:            TensorType["N", "xyz"],
                    timestampts:    TensorType["N", "1"],
                    min_axial:      Optional[th.tensor]=None,
                    max_axial:      Optional[th.tensor]=None):
        """(Description):
                Basic forward method implemented from: https://arxiv.org/pdf/2301.09632.
                    1) normalize all input types (xyz, timestampts) to valid clipping range [0, 1]
                    2) process each plane separatly with torch.nn.functional.grid_sample()
                    3) concatenate features from three planes along last axis: Tensor[N, 3*fdim]
                    4) forward call thought last head(...) nn.Module to get resulting features: Tensor[N, odim]
            (Arguments):
            param xyz: XYZ sparse points to sample with bilinear interpolater
            type xyz: th.Tensor[N, 3]

            param timestampts: per-point timestamps aligned with xyz; if None,
                                temporal planes are skipped and only spatial
                                planes are sampled
            type timestampts: th.Tensor[N, 1] or th.Tensor[N] or None

            param min_axial: lower bounds (x_min, y_min, z_min) used for
                            normalization; if None, computed as
                            torch.min(xyz, dim=0).values
            type min_axial: Optional[th.Tensor[3]]

            param max_axial: upper bounds (x_max, y_max, z_max) used for
                            normalization; if None, computed as
                            torch.max(xyz, dim=0).values
            type max_axial: Optional[th.Tensor[3]]

        (Returns):
            param features: fused per-point feature representation produced by
                            summing (or concatenating, depending on the design)
                            contributions from all three plane pairs
            type features: th.Tensor[N, fdim]  (or th.Tensor[N, 3*fdim] for concat)

        (Notes):
            - grid_sample expects grid values in [-1, 1]; that is why normalize()
                maps inputs via  2 * (v - v_min) / (v_max - v_min) - 1
            - timestampts are normalized independently from xyz using their own
                min/max, otherwise temporal resolution collapses for short sequences
            - each plane attribute is stored as nn.Parameter of shape
                (1, fdim, H, W) to satisfy grid_sample's 4D input requirement
            """
        
        def _get_bounds(value, mode="min") -> float | th.Tensor:
            if value is None:
                value = getattr(th, mode)(xyz, dim=0).values
            return value

        def _sample_weights(name: str) -> TensorType["N", "fdim"]:
            if name in self._plane_indices:
                indices = self._plane_indices[name]
                grid = xyz[:, indices].view(1, -1, 1, 2)
            elif name in self._tplane_indices \
                and timestampts is not None:
                indices = self._tplane_indices[name]
                times = timestampts \
                            if timestampts.ndim == 2 \
                            else timestampts.unsqueeze(dim=-1)
                grid = th.cat([xyz[:, indices, None], times], dim=-1)\
                        .view(1, -1, 1, 2)
            else:
                raise ValueError(f"uknown weigths group: {name}")

            
            values = grid_sample(getattr(self, f"_{name}_plane"), 
                                grid, mode="bilinear")
            return values.squeeze().transpose(0, 1)

        if timestampts is not None:
            tmin = timestampts.min()
            tmax = timestampts.max()
            timestampts = self.normalize(timestampts, tmin, tmax)

        xyz = self.normalize(xyz, 
                            _get_bounds(min_axial, "min").view(1, 3),
                            _get_bounds(max_axial, "max").view(1, 3))
        features = []
        for volumes in self._volumes:
            (p, tp) = volumes.split("-")
            values = _sample_weights(p)
            if timestampts is not None:
                tvalues = _sample_weights(tp)
                values *= tvalues
            features.append(values)
        features = self.head(th.cat(features, dim=-1))
        return features


