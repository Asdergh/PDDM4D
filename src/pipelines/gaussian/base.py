import numpy as np
import torch as th
import torch.nn as nn
import torch.functional as F
import lightning as l
from dataclasses import (dataclass, fields, field)
from torchtyping import TensorType
from typing import Tuple, Dict, Any, Optional
from gsplat import rasterization 
from incremental_pca_torch import IncrementalPCA
# from open3d.geometry import (KDTreeSearchParamKNN, PointCloud, estimate_normals)
from plyfile import (PlyData, PlyElement)
from ...utils.criterion import (CombinedVisualLoss, CombinedVisualLossConfig)
from ...scene.gaussian import (GsModule,
                                GsSModelConfig,
                                GsModelOutput,
                                AnchorGrowing)



# TODO :    1) complite training_step
#           2) implement render_normals
#           3) model parameters io
#           4) ply file gneration utilitie from GsModuleOutput
#           5) Camera params optimization [extrinsics/intrinsics]

@dataclass
class Base3DReconstructionPiepelineConfig:
    gs_model: GsSModelConfig=field(default=GsSModelConfig)
    criterion: CombinedVisualLossConfig=field(default=CombinedVisualLossConfig)

class Base3DReconstructionPiepeline(l.LightningModule):
    def __init__(self, config: GsSModelConfig, 
                criterion_config: CombinedVisualLossConfig):
        super(Base3DReconstructionPiepeline, self).__init__()
        self.config = config
        self.pc = GsModule(config)
        self.stratagy = AnchorGrowing(self.pc, config)
        self.criterion = CombinedVisualLoss(criterion_config)

    def setup(self, stage: str):
        if stage == "fit":
            dataset = self.trainer\
                                .train_dataloader()\
                                .dataset
            self.pc.setup_model(**dataset.sparse)

    def configure_optimizers(self):
        return {"optimizer": self.pc.optimizer}

    def render(self, extrinsics: TensorType["C", "4", "4"],
                    intrinsics: TensorType["C", "3", "3"],
                    far_near: Tuple[float]=(1e-2, 1e10),
                    features: Optional[TensorType["N", "d"]]=None):

        gs = self.pc.generate_splats()
        features = gs.colors if features is None else features
        (render_rgb, render_alpha, meta) = rasterization(
            means=gs.xyz,
            quats=gs.rotations,
            scales=gs.scales,
            opacities=gs.opacities,
            colors=gs.colors,
            viewmats=extrinsics,
            Ks=intrinsics,
            near_plane=far_near[0],
            far_plane=far_near[1]
        )
        return {"render_rgb": render_rgb, 
                "render_alpha": render_alpha,
                "meta": meta}

    def render_anchors(self, extrinsics: TensorType["C", "4", "4"],
                    intrinsics: TensorType["C", "3", "3"],
                    far_near: Tuple[float]=(1e-2, 1e10)):
        features = self.pc._feats
        return self.render(extrinsics, intrinsics, far_near, features=features)
    
    def _step(self, batch: Dict[str, Any], batch_idx: int, mode: str):
        (images, extrinsics, intrinsics) = (batch["images"],
                                            batch["extrinsics"],
                                            batch["intrinsics"])
        render_pkg = self.render(extrinsics, 
                                intrinsics, 
                                (self.pc.config.near_plane, 
                                self.pc.config.far_plane))
        losses = self.criterion(images, render_pkg)
        for (name, value) in losses.items():
            self.log(f"{mode}-{name}", value, on_step=True)

        gs = self.pc.generate_splats()
        self.stratagy.step(self.current_epoch, gs.xyz, gs.opacities)
        return losses

    def training_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, mode="train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, mode="val")

    def write_ply(self, path: str, 
                        gs: Optional[GsModelOutput]=None, 
                        mode: str="anchor"):
        """Writes gaussian splatting file from input gs: GsModuleOutput 
        or from egneration from current state pc module in pipeline. """
        correct_naming = dict(feats="feats",
                            rotations="rot", 
                            scales="scale", 
                            opacities="opacity",
                            colors="f_dc",
                            anchor_0="x", xyz_0="x",
                            anchor_1="y", xyz_1="y",
                            anchor_2="z", xyz_2="z")
        def _anchor_attributes():
            (values, dtypes) = {}, []
            for name in self.pc.attributes:
                value = getattr(self.pc, name)
                dtype = list(f"{correct_naming[name]}_{idx}" 
                                if name != "anchors" 
                                else correct_naming[f"{name}_{idx}"] 
                                for idx in range(value.shape[1]))
                value = dict(zip(dtype, np.split(value, value.shape[1], axis=-1)))
                values.update(value); dtypes += dtype
            dtypes = list(map(lambda attrib: (attrib, "f4"), dtypes))
            attribues = np.empty(dtype=dtypes)
            for name in values:
                attribues[name] = values[name]
            return attribues 
        
        def _default_gs_attribues(gs=None):
            (values, dtypes) = [], []
            gs = gs if gs is not None else self.pc.generate_splats().to_numpy()
            for name in fields(gs):
                value = getattr(gs, name)
                dtype = list(f"{correct_naming[name]}_{idx}" 
                                if name != "xyz" 
                                else correct_naming[f"{name}_{idx}"] 
                                for idx in range(value.shape[1]) if name != "covariences")
                value = dict(zip(dtype, np.split(value, value.shape[1], axis=-1)))
                values.update(value); dtypes += dtype
            dtypes = list(map(lambda attrib: (attrib, "f4"), dtypes))
            attribues = np.empty(dtype=dtypes)
            for name in values:
                attribues[name] = values[name]
            return attribues 

        if gs is not None: mode = "default"
        attribs_fns = dict(anchor=_anchor_attributes, 
                            default=_default_gs_attribues)
        attribues = attribs_fns[mode](gs)
        elements = PlyElement.describe(attribues)
        PlyData([elements]).write(path)

