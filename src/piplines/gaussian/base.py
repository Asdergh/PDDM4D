import torch as th
import torch.nn as nn
import torch.functional as F
import lightning as l
from torchtyping import TensorType
from typing import Tuple, Dict, Any
from gsplat import rasterization 

from ...scene.gaussian import (GsModule,
                                GsSModelConfig,
                                GsModelOutput)

class BaseGsPipeline(l.LightningModule):
    def __init__(self, config: GsSModelConfig):
        super(BaseGsPipeline, self).__init__()
        self.config = config
        self.pc = GsModule(config)

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
                    far_near: Tuple[float]=(1e-2, 1e10)):

        gs = self.pc.generate_splats()
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
    
    def train(self, batch: Dict[str, Any], batch_idx: int):

        (images, extrinsics, intrinsics) = (batch["images"],
                                            batch["extrinsics"],
                                            batch["intrinsics"])
        render_pkg = self.render(extrinsics, 
                                intrinsics, 
                                (self.pc.config.near_plane, 
                                self.pc.config.far_plane))
        
        

    