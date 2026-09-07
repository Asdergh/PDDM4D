import rerun as rr
import rerun.blueprint as rrb
import torch as th

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
        log_single_view(f"{origin}/frame-{idx}", image=image,
                        K=Ks[idx, ...],
                        viewmat=viewmats[idx, ...])

def log_pts(path, xyz, cmap, meter: float=0.001):
    if cmap is not None:
        assert len(xyz) == len(cmap)
    rr.log(path, rr.Points3D(positions=xyz,
                            colors=cmap,
                            radii=[meter]))

def log_dataset(dataset):
    pts_pkg = dataset.sparse
    origin = "origin"
    rr.init(origin, spawn=True)
    rr.send_blueprint(rrb.Blueprint(rrb.Spatial3DView(origin=origin)))
    for idx, sample in enumerate(dataset):
        log_single_view(f"{origin}/frame-{idx}",
                        image=sample["image"],
                        K=sample["intrinsics"],
                        viewmat=sample["viewmat_w2c"])
        
    log_pts(f"{origin}/pts3D", 
            pts_pkg["xyz"], 
            pts_pkg["rgb"] / pts_pkg["rgb"].max())