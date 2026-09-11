import torch as th
import torch.nn as nn
import torch.nn.functional as Fn
from torch.nn.functional import grid_sample
from typing import (Tuple, Optional, Any, Callable, List, Dict)
from torchtyping import TensorType




class HexPlane(nn.Module):
    _volumes: List[str] = ["xy-zt", "xz-yt", "yz-xt"] 
    _plane_indices: Dict[str, tuple] = {"xy": (0, 1),
                                "xz": (0, 2),
                                "yz": (1, 2)}
    _tplane_indices: Dict[str, int] = {"xt": 0, "yt": 1, "zt": 2}
    def __init__(self, features_dim: int,
                        grid_size: Tuple[int] | int=200,
                        time_grid_size: int=1000,
                        activation: Optional[Callable[..., nn.Module]]=None,
                        device: str="cuda",
                        initialize: bool=False):

        super(HexPlane, self).__init__()
        self.fdim = features_dim
        grid_size = grid_size \
                    if isinstance(grid_size, tuple) \
                    else 3*(grid_size, )
        grid_size += (time_grid_size, )
        self.sizes = {name: grid_size[idx] 
                    for idx, name in 
                    enumerate(["x", "y", "z", "t"])}
        self.device = device
        self.activation: Callable[..., nn.Module] = activation
        if initialize:
            self.set_planes_weights()

    def set_planes_weights(self, **kwargs):
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
        return 2 * ((values - a) / (b - a)) - 1
    
    def forward(self, 
                    xyz:            TensorType["N", "xyz"],
                    timestampts:    TensorType["N", "1"],
                    min_axial:      Optional[th.tensor]=None,
                    max_axial:      Optional[th.tensor]=None):

        def _get_bounds(value, mode="min"):
            if value is None:
                value = getattr(th, mode)(xyz, dim=0).values
            return value

        def _sample_weights(name: str):
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
            values = values if self.activation is None else self.activation(values)
            return values.squeeze()

        if timestampts is not None:
            tmin = timestampts.min()
            tmax = timestampts.max()
            timestampts = self.normalize(timestampts, tmin, tmax)

        xyz = self.normalize(xyz, 
                            _get_bounds(min_axial, "min").view(1, 3),
                            _get_bounds(max_axial, "max").view(1, 3))

        features = None
        for volumes in self._volumes:
            (p, tp) = volumes.split("-")
            values = _sample_weights(p)
            if timestampts is not None:
                tvalues = _sample_weights(tp)
                values *= tvalues
            features = values if features is None else (features + values)
        return features



if __name__ == "__main__":
    activation = None
    model = HexPlane(64, initialize=True, activation=activation, device="cpu")
    print(model)
    xyz = th.normal(0, 1, (100, 3))
    timestampts = th.normal(0, 1, (100, 1))
    values = model(xyz, timestampts)
    print(values.shape, values.min(), values.mean(), values.max())


    weights = {"xy": th.normal(0, 1, (64, 200, 200)), "zt": th.normal(0, 1, (64, 200, 1000))}
    model.set_planes_weights(**weights)
    values = model(xyz, timestampts)
    print(values.shape, values.min(), values.mean(), values.max())
