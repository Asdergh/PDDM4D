import torch as th
import os
import random as rd
import numpy as np
from dataclasses import (dataclass, field)
from tqdm import tqdm
from functools import cached_property
from PIL import Image
from torchvision.transforms.functional import to_tensor
from kornia.geometry.conversions import quaternion_to_rotation_matrix
from typing import (Optional, Literal, Dict, Any)
from torch.utils.data import Dataset
from nerfstudio.data.utils.colmap_parsing_utils import (read_cameras_binary,
                                                        read_points3D_binary,
                                                        read_images_binary)
from .registry import register_dataset



@dataclass
class MipNerf360v2DatasetConfig:
    path: str
    name: str=field(default="360v2", repr=False, compare=False)
    scene: Literal["bicycle", 
                    "bonsai",
                    "counter",
                    "garden",
                    "kitchen",
                    "room",
                    "stump"]="garden"
    resolution_type: int=0
    max_views: Optional[int]=None
    random_views: bool=False

@register_dataset("360v2", MipNerf360v2DatasetConfig)
class MipNerf360v2Dataset(Dataset):
    _res_factors = [1, 2, 4, 8]
    def __init__(self, config: MipNerf360v2DatasetConfig):
        super(MipNerf360v2Dataset, self).__init__()
        self.config = config
        content = list(os.listdir(config.path))
        self.path = os.path.join(config.path, config.scene)
        assert (config.scene in content), \
        (f"there is not such {config.scene=} in data folder.")
        content = list(os.listdir(self.path))
        assert ("sparse" in content), \
        (f"coudn't find sparse data at location: {self.path}. \n"
            "Mip_Nerf360v2 dataest have restricted data format \n"
            "that can figure at the link: https://jonbarron.info/mipnerf360/ \n")

        images_folders = list(f for f in os.listdir(self.path) if "images" in f)
        assert config.resolution_type < len(images_folders)
        self.images_folder = os.path.join(self.path, images_folders[self.config.resolution_type])
        self.sparse_data = os.path.join(self.path, "sparse")

        self.transform: callable = None
        self._read_data()

    @property
    def image_transoform(self):
        return self.trnasform

    @image_transoform.setter
    def image_transform(self, transform: callable):
        self.transform = transform

    def _get_factor(self, size: tuple):
        new_size = (size[0] / self._res_factors[self.config.resolution_type], 
                    size[1] / self._res_factors[self.config.resolution_type])
        return {"sx": new_size[0] / size[0], 
                "sy": new_size[1] / size[1]}

    @cached_property
    def sparse(self):
        pts3D = read_points3D_binary(os.path.join(self.sparse_data, "0/points3D.bin"))
        print(len(pts3D))
        (xyz, rgb) = [], []
        for idx in tqdm(self.pts3D_indices, desc="reading pts..."):
            pts = pts3D[idx]
            xyz.append(pts.xyz)
            rgb.append(pts.rgb)
        rgb = th.Tensor(rgb)
        rgb /= rgb.max()
        return {"xyz": th.Tensor(xyz), "rgb": rgb}

    def _read_data(self):
        self.intrinsics: Dict[int, Any] = read_cameras_binary(os.path.join(self.sparse_data, "0/cameras.bin"))
        images_info: Dict[int, Any] = read_images_binary(os.path.join(self.sparse_data, "0/images.bin"))

        max_views = min(self.config.max_views, len(images_info))   \
                        if self.config.max_views is not None       \
                        else len(images_info)
        view_indices = list(images_info.keys())
        view_indices = rd.sample(view_indices, max_views)          \
                        if self.config.random_views                \
                        else view_indices[:max_views]

        self.samples = []
        self.pts3D_indices = []
        for view_id in tqdm(view_indices, desc="Loading 360v2..."):
            sample = images_info[view_id]
            imagef = os.path.join(self.images_folder, sample.name)

            camera = self.intrinsics[sample.camera_id]
            scale = self._get_factor((camera.width, camera.height))
            params = th.from_numpy(camera.params).float()
            K = th.eye(3)
            K[0, 0] = scale["sx"]*params[0]
            K[0, 2] = scale["sx"]*params[2]
            K[1, 1] = scale["sy"]*params[1]
            K[1, 2] = scale["sy"]*params[3]

            Twc = Tcw = th.eye(4)
            t = th.from_numpy(sample.tvec).float()
            R = quaternion_to_rotation_matrix(th.from_numpy(sample.qvec).float())
            Tcw[:3, :3] = R
            Tcw[:3, 3] = t
            Twc = th.linalg.inv(Tcw)

            self.pts3D_indices.append(sample.point3D_ids[sample.point3D_ids != -1])
            sample = dict(image=imagef,
                            intrinsics=K,
                            viewmat_w2c=Twc,
                            viewmat_c2w=Tcw)
            self.samples.append(sample)
        self.pts3D_indices = np.unique(np.concatenate(self.pts3D_indices))\
        .astype(np.int32)\
        .tolist()

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx: int):
        if idx >= len(self):
            raise IndexError(f"{idx=} is out range for dataset lenght")
        image = to_tensor(Image.open(self.samples[idx]["image"]))
        image = image                       \
                if self.transform is None   \
                else self.transform(image)
        return {**self.samples[idx], "image": image}

