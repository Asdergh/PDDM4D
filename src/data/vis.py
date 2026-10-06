import rerun as rr
import rerun.blueprint as rrb
import torch as th
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as Rt

def log_single_view(path: str, viewmat, K, image=None, c=None):

    rr.log(path, rr.Transform3D(mat3x3=viewmat[:3, :3],
                                translation=viewmat[:3, 3]),
                                rr.Pinhole(image_from_camera=K, color=c))
    if image is not None:
        if image.shape[0] == 3:
            image = image.permute(1, 2, 0)
        rr.log(path, rr.Image(image=image))

def log_views(origin: str, viewmats, Ks, images=None, cs=None):
    assert len(viewmats) == len(Ks)
    if cs is not None:
        assert len(cs) == len(viewmats)
    if images is not None:
        assert len(images) == len(viewmats)
    for idx in range(len(viewmats)):
        image = None                \
                if images is None   \
                else images[idx, ...]
        log_single_view(f"{origin}/frame-{idx}", 
                        image=image,
                        K=Ks[idx, ...],
                        viewmat=viewmats[idx, ...])

def log_pts(path, xyz, cmap, meter: float=0.001):
    if cmap is not None:
        assert len(xyz) == len(cmap)
    rr.log(path, rr.Points3D(positions=xyz,
                            colors=cmap,
                            radii=[meter]))


def log_pcd_obb(path, xyz, c=None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    oob = pcd.get_oriented_bounding_box()

    quat = Rt.from_matrix(oob.R).as_quat()
    rr.log(path, rr.Boxes3D(centers=oob.center,
                            quaternions=quat,
                            colors=([0, 255, 0] if c is None else c),
                            half_sizes=(oob.extent / 2),
                            labels=["Viewpoints OBB"],
                            show_labels=True))
    
def log_dataset(dataset):
    pts_pkg = dataset.sparse
    origin = "origin"
    rr.init(origin, spawn=True)
    rr.send_blueprint(rrb.Blueprint(rrb.Spatial3DView(origin=origin)))
    view_Cs = []
    for idx, sample in enumerate(dataset):
        log_single_view(f"{origin}/Frame/frame-{idx}",
                        image=sample["image"],
                        K=sample["intrinsics"],
                        viewmat=sample["viewmat_w2c"])
        view_Cs.append(sample["viewmat_w2c"][:3, 3])
    view_Cs = th.stack(view_Cs).numpy()
    log_pcd_obb(f"{origin}/Viewpoints_OBB", view_Cs)
    log_pcd_obb(f"{origin}/PiontCloud_OBB", pts_pkg["xyz"], c=[255, 0, 0])
    log_pts(f"{origin}/pts3D", 
            pts_pkg["xyz"], 
            pts_pkg["rgb"] / pts_pkg["rgb"].max())

    
    