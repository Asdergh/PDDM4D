"""Interactive Gaussian Splatting scene visualizer based on viser.

The single class :class:`GaussianSplatVisualizer` takes a
:class:`SpatialContainer` and displays the scene in a web browser. Four render
modes are available in the browser GUI:

* ``rgb``             -- original Gaussian colors (``colors``);
* ``features_pca``    -- PCA projection of anchor features (``features``):
                        the first three principal components are normalized and
                        shown as RGB channels;
* ``depth``           -- depth as the distance from the Gaussian center to the
                        current client camera, colored with a viridis colormap
                        (recomputed as the camera moves);
* ``view_direction``  -- map of the angular distance between the Gaussian normal
                        and rays to fixed user viewpoints. If there are several
                        viewpoints, their individual maps are combined into one
                        (mean or distance-weighted).

To avoid overloading the simulator, switching modes never recreates the scene;
it only overwrites the ``rgbs`` field of the already created gaussian-splat
container on the ``viser`` server.

The entry point is either :meth:`GaussianSplatVisualizer.run` or the module-level
function :func:`visualize`, which accepts a :class:`SpatialContainer` directly.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import viser
from warnings import warn

try:
    import torch
    from torchtyping import TensorType
except Exception:
    torch = None
    TensorType = "TensorType"  # type: ignore[assignment]


def _quat_to_matrix_np(q: np.ndarray) -> np.ndarray:
    """Convert (N,4) quaternions in (x,y,z,w) order to rotation matrices (N,3,3)."""
    q = np.asarray(q, dtype=np.float64)
    q = q / np.maximum(np.linalg.norm(q, axis=-1, keepdims=True), 1e-12)
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.stack(
        [
            np.stack([1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)], axis=-1),
            np.stack([2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)], axis=-1),
            np.stack([2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)], axis=-1),
        ],
        axis=-2,
    )


def _quat_to_matrix_torch(q: "torch.Tensor") -> "torch.Tensor":
    """Torch version of :func:`_quat_to_matrix_np`."""
    q = q.float()
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return torch.stack(
        [
            torch.stack([1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)], dim=-1),
            torch.stack([2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)], dim=-1),
            torch.stack([2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)], dim=-1),
        ],
        dim=-2,
    )


def _matrix_to_quat_xyzw(R: np.ndarray) -> np.ndarray:
    """Convert rotation matrices (...,3,3) to quaternions (...,4) in (x,y,z,w) order.

    Uses the stable Shepperd method: for traces <= 0, the branch with the largest
    diagonal element is selected.
    """
    R = np.asarray(R, dtype=np.float64)
    out_shape = R.shape[:-2] + (4,)
    R = R.reshape((-1, 3, 3))
    n = R.shape[0]
    quat = np.zeros((n, 4), dtype=np.float64)

    trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
    diag = np.stack([R[:, 0, 0], R[:, 1, 1], R[:, 2, 2]], axis=-1)

    case1 = trace > 0.0
    if np.any(case1):
        s = np.sqrt(np.maximum(trace[case1] + 1.0, 0.0)) * 2.0
        quat[case1, 0] = (R[case1, 2, 1] - R[case1, 1, 2]) / s
        quat[case1, 1] = (R[case1, 0, 2] - R[case1, 2, 0]) / s
        quat[case1, 2] = (R[case1, 1, 0] - R[case1, 0, 1]) / s
        quat[case1, 3] = 0.25 * s

    rest = ~case1
    if np.any(rest):
        best_axis = np.argmax(diag[rest], axis=-1)
        for axis in range(3):
            sel = rest.copy()
            sel[rest] = best_axis == axis
            if not np.any(sel):
                continue
            Rsel = R[sel]
            if axis == 0:
                s = np.sqrt(
                    np.maximum(1.0 + Rsel[:, 0, 0] - Rsel[:, 1, 1] - Rsel[:, 2, 2], 0.0)
                ) * 2.0
                quat[sel, 0] = 0.25 * s
                quat[sel, 1] = (Rsel[:, 0, 1] + Rsel[:, 1, 0]) / s
                quat[sel, 2] = (Rsel[:, 0, 2] + Rsel[:, 2, 0]) / s
                quat[sel, 3] = (Rsel[:, 2, 1] - Rsel[:, 1, 2]) / s
            elif axis == 1:
                s = np.sqrt(
                    np.maximum(1.0 + Rsel[:, 1, 1] - Rsel[:, 0, 0] - Rsel[:, 2, 2], 0.0)
                ) * 2.0
                quat[sel, 0] = (Rsel[:, 0, 1] + Rsel[:, 1, 0]) / s
                quat[sel, 1] = 0.25 * s
                quat[sel, 2] = (Rsel[:, 1, 2] + Rsel[:, 2, 1]) / s
                quat[sel, 3] = (Rsel[:, 0, 2] - Rsel[:, 2, 0]) / s
            else:
                s = np.sqrt(
                    np.maximum(1.0 + Rsel[:, 2, 2] - Rsel[:, 0, 0] - Rsel[:, 1, 1], 0.0)
                ) * 2.0
                quat[sel, 0] = (Rsel[:, 0, 2] + Rsel[:, 2, 0]) / s
                quat[sel, 1] = (Rsel[:, 1, 2] + Rsel[:, 2, 1]) / s
                quat[sel, 2] = 0.25 * s
                quat[sel, 3] = (Rsel[:, 1, 0] - Rsel[:, 0, 1]) / s

    return quat.reshape(out_shape)


def _to_numpy(t: Union[np.ndarray, "torch.Tensor"], dtype: np.dtype) -> np.ndarray:
    """Convert a torch tensor or ndarray to a numpy array of the given dtype."""
    if torch is not None and isinstance(t, torch.Tensor):
        t = t.detach().float().cpu().numpy()
    return np.asarray(t, dtype=dtype)


_COLORMAP_KEYS = np.array([0.0, 0.21, 0.35, 0.5, 0.65, 0.8, 1.0], dtype=np.float64)
_COLORMAP_VALUES = np.array(
    [
        [0.267, 0.005, 0.329],
        [0.282, 0.139, 0.548],
        [0.254, 0.340, 0.625],
        [0.204, 0.534, 0.527],
        [0.266, 0.686, 0.359],
        [0.600, 0.804, 0.318],
        [0.993, 0.906, 0.144],
    ],
    dtype=np.float64,
)


def _fallback_colormap(values: np.ndarray) -> np.ndarray:
    """Fallback viridis-style colormap (piecewise linear interpolation)."""
    values = np.clip(np.asarray(values, dtype=np.float64), 0.0, 1.0)
    return np.stack(
        [
            np.interp(values, _COLORMAP_KEYS, _COLORMAP_VALUES[:, 0]),
            np.interp(values, _COLORMAP_KEYS, _COLORMAP_VALUES[:, 1]),
            np.interp(values, _COLORMAP_KEYS, _COLORMAP_VALUES[:, 2]),
        ],
        axis=-1,
    )


class GaussianSplatVisualizer:
    """Interactive Gaussian Splatting scene visualizer.

    The constructor accepts a :class:`SpatialContainer` and does nothing else --
    to start, call :meth:`run` (or the module-level :func:`visualize`).
    """

    MODES: Tuple[str, ...] = ("rgb", "features_pca", "depth", "view_direction")

    def __init__(
        self,
        container,
        host: str = "0.0.0.0",
        port: int = 8080,
        *,
        title: str = "Gaussian Splatting",
        start_position: Sequence[float] = (0.0, 0.0, 4.0),
        start_look_at: Sequence[float] = (0.0, 0.0, 0.0),
        viewpoint_blend: str = "mean",
        orient_normals_outward: bool = True,
        frustum_scale: float = 0.2,
        cmap: str="turbo"
    ) -> None:
        """
        Args:
            container: Scene with Gaussians; normals and covariances are taken
                from the ``covariances`` field (if absent, computed from
                ``scales``/``rotations``).
            host: Interface the viser server listens on.
            port: HTTP/websocket server port.
            title: Title of the GUI panel in the browser.
            start_position: Starting camera position for new clients.
            start_look_at: Point the starting camera looks at.
            viewpoint_blend: How to combine maps from multiple viewpoints:
                ``"mean"`` (average of cosines) or
                ``"distance_weighted"`` (weighted by proximity of the viewpoint
                to the Gaussian -- akin to interpolating between maps).
            orient_normals_outward: Whether to flip normals "away from the
                scene's center of mass". The sign of the eigenvector from the
                covariance decomposition is arbitrary, so without this option
                half the normals "look inward" and the view_direction map looks
                noisy. For convex and roughly convex scenes, keep this enabled.
            frustum_scale: Size of the displayed camera frustums for viewpoints.
        """
        self._cmap = cmap
        self._host = host
        self._port = port
        self._title = title
        self._start_position = tuple(float(v) for v in start_position)
        self._start_look_at = tuple(float(v) for v in start_look_at)
        assert viewpoint_blend in ("mean", "distance_weighted"), viewpoint_blend
        self._viewpoint_blend = viewpoint_blend
        self._orient_normals_outward = bool(orient_normals_outward)
        self._frustum_scale = float(frustum_scale)

        self._positions: np.ndarray = _to_numpy(container.xyz, np.float32).reshape(-1, 3)
        self._features = None
        if hasattr(container, "featuers"):
            features_np = np.asarray(_to_numpy(container.features, np.float32))
            feature_dim = int(features_np.shape[1]) if features_np.ndim == 2 else 1
            self._features: np.ndarray = features_np.reshape(-1, feature_dim)
        else:
            warn("seems like container doesnt have features fields" \
                "you coudn'y plot pca projections onto splats in such case")
        self._colors: np.ndarray = np.clip(
            _to_numpy(container.colors, np.float32).reshape(-1, 3), 0.0, 1.0
        )
        self._opacities: np.ndarray = _to_numpy(container.opacities, 
                                                np.float32).reshape(-1, 1)
        self._covariances: np.ndarray = _to_numpy(container.covariances, 
                                                np.float32).reshape(-1, 3, 3)
        n = self._positions.shape[0]
        assert self._covariances.shape == (n, 3, 3)
        assert self._colors.shape == (n, 3)
        assert self._opacities.shape == (n, 1)
        if self._features is not None:
            assert self._features.shape[0] == n

        self._lock = threading.Lock()
        self._mode: str = self.MODES[0]
        self._viewpoints: List[Tuple[np.ndarray, np.ndarray]] = []  # (position, wxyz)
        self._frustums: Dict[str, viser.CameraFrustumHandle] = {}
        self._status_handles: Dict[int, viser.GuiMarkdownHandle] = {}
        self._last_client_camera: Optional[Tuple[np.ndarray, np.ndarray, float]] = None
        self._last_depth_update: float = 0.0
        self._server: Optional[viser.ViserServer] = None
        self._splats: Optional[viser.GaussianSplatHandle] = None

        self._normal_cache: Optional[np.ndarray] = None
        self._pca_cache: Optional[np.ndarray] = None

    def run(self) -> None:
        """Start the viser server, build the GUI, and block the thread until stopped.

        The single "public" function that implements the whole pipeline: creates
        the gaussian-splat container, attaches all viser handlers/decorators
        (as nested subfunctions), and waits until the user stops the visualizer
        (Ctrl+C).
        """
        server = viser.ViserServer(host=self._host, port=self._port, verbose=False)
        self._server = server

        server.initial_camera.position = self._start_position  # type: ignore[attr-defined]
        server.initial_camera.look_at = self._start_look_at  # type: ignore[attr-defined]

        self._splats = server.scene.add_gaussian_splats(
            "gaussians",
            centers=self._positions,
            covariances=self._covariances,
            rgbs=self._colors.copy(),
            opacities=self._opacities,
        )

        @server.on_client_connect
        def _on_client_connect(client: viser.ClientHandle) -> None:
            """Attach the GUI for the new client; all handlers are nested subfunctions."""
            self._build_client_ui(client)

        host = server.get_host()
        port = server.get_port()
        url = f"http://localhost:{port}" if host in ("0.0.0.0", "::") else f"http://{host}:{port}"
        print(f"\n[GaussianSplatVisualizer] Open the viewer in a browser: {url}\n")

        try:
            server.sleep_forever()
        except KeyboardInterrupt:
            print("\n[GaussianSplatVisualizer] Stopping...")
        finally:
            server.flush()
            server.stop()

    def stop(self) -> None:
        """Stop the running viser server (if any)."""
        if self._server is not None:
            self._server.stop()

    def _build_client_ui(self, client: viser.ClientHandle) -> None:
        """Create the GUI for a specific client and register all viser handlers.

        The "Save current viewpoint" button captures the client's current camera
        view (via ``client.camera``) and draws a camera frustum in the scene.
        All ``on_click`` / ``on_update`` decorators are nested subfunctions.
        """

        with client.gui.add_folder(self._title):
            mode_group = client.gui.add_button_group("Render mode", list(self.MODES))

            @mode_group.on_click
            def _on_mode_clicked(_event: viser.GuiEvent) -> None:
                """Apply the selected mode to the shared splat container."""
                mode = mode_group.value
                if mode in self.MODES:
                    self._set_mode(mode)

            status_markdown = client.gui.add_markdown(self._status_markdown())

            with client.gui.add_folder("Viewpoints"):
                save_button = client.gui.add_button("Save current viewpoint")
                clear_button = client.gui.add_button("Clear viewpoints")

                @save_button.on_click
                def _on_save_clicked(_event: viser.GuiEvent) -> None:
                    """Capture the client's current camera as a new viewpoint."""
                    self._save_viewpoint(client)

                @clear_button.on_click
                def _on_clear_clicked(_event: viser.GuiEvent) -> None:
                    """Remove all saved viewpoints and their frustums."""
                    self._clear_viewpoints()

        self._status_handles[client.client_id] = status_markdown

        @client.camera.on_update
        def _on_camera_updated(cam: viser.CameraHandle) -> None:
            """Update the last known state of the client's camera.

            When depth mode is active, the map is recomputed relative to this
            camera (with throttling to avoid overloading the event loop).
            """
            with self._lock:
                self._last_client_camera = (
                    np.asarray(cam.position, dtype=np.float64),
                    np.asarray(cam.wxyz, dtype=np.float64),
                    float(cam.fov),
                )
                now = time.time()
                refresh_depth = self._mode == "depth" and (
                    now - self._last_depth_update > 0.25
                )
                if refresh_depth:
                    self._last_depth_update = now
            if refresh_depth:
                self._set_mode("depth")

    def _save_viewpoint(self, client: viser.ClientHandle) -> None:
        """Save the client's current camera view as a viewpoint + camera frustum."""
        camera = client.camera
        try:
            position = np.asarray(camera.position, dtype=np.float64)
            wxyz = np.asarray(camera.wxyz, dtype=np.float64)
            aspect = float(camera.aspect)
            fov = float(camera.fov)
        except Exception:
            print("[GaussianSplatVisualizer] Camera state not synced yet; ignored.")
            return
        if not np.isfinite(aspect) or aspect <= 0.01:
            aspect = 16.0 / 9.0
        if not np.isfinite(fov) or fov <= 0.0:
            fov = 60.0 * np.pi / 180.0

        with self._lock:
            self._viewpoints.append((position.copy(), wxyz.copy()))
            index = len(self._viewpoints) - 1
            name = f"viewpoint_{index}"
            frustum = self._server.scene.add_camera_frustum(
                name,
                fov=fov,
                aspect=aspect,
                scale=self._frustum_scale,
                wxyz=wxyz,
                position=position,
                color=(0, 230, 110),
                thickness=0.015,
            )
            self._frustums[name] = frustum
            mode = self._mode

        print(
            f"[GaussianSplatVisualizer] Viewpoint #{index} saved: "
            f"position={np.round(position, 3).tolist()} fov={np.round(fov, 3)}"
        )
        if mode == "view_direction":
            self._set_mode(mode)
        self._update_status_markdown()

    def _clear_viewpoints(self) -> None:
        """Remove all saved viewpoints (and their frustums) from the scene."""
        with self._lock:
            frustums = list(self._frustums.values())
            self._frustums.clear()
            self._viewpoints.clear()
            mode = self._mode
        for frustum in frustums:
            frustum.remove()
        if mode == "view_direction":
            self._set_mode("rgb")
        self._update_status_markdown()
        print("[GaussianSplatVisualizer] Viewpoints cleared.")

    def _get_depth_reference(self) -> Optional[np.ndarray]:
        """Reference position for the depth map: the last client camera, otherwise
        the last saved viewpoint, otherwise None."""
        with self._lock:
            if self._last_client_camera is not None:
                return self._last_client_camera[0]
            if self._viewpoints:
                return self._viewpoints[-1][0]
        return None

    def _set_mode(
        self,
        mode: str,
        reference_position: Optional[np.ndarray] = None,
    ) -> None:
        """Apply mode ``mode``, overwriting only the ``rgbs`` of the splat container.

        If the mode cannot be computed (no viewpoints for ``view_direction`` or
        no reference for ``depth``), the current colors are left unchanged.
        """
        if mode == "rgb":
            colors = self._colors.copy()
        elif mode == "features_pca":
            if self._features is None: 
                warn("featuers can't be used as a colormap\n" \
                    "pass gs container that will satisfy the \n" \
                    "format described in __int__ sig. description")
                return
            colors = self._pca_colors
        elif mode == "depth":
            reference = (
                reference_position
                if reference_position is not None
                else self._get_depth_reference()
            )
            if reference is None:
                print("[GaussianSplatVisualizer] No camera reference for depth mode; ignored.")
                return
            colors = self._depth_colors(reference)
        elif mode == "view_direction":
            colors = self._view_direction_colors()
            if colors is None:
                print(
                    "[GaussianSplatVisualizer] view_direction requires at least one "
                    "saved viewpoint."
                )
                return
        else:
            raise ValueError(f"Unknown render mode: {mode}")

        with self._lock:
            self._mode = mode
        if self._splats is not None:
            self._splats.rgbs = colors
        self._update_status_markdown()

    def _pca_colors(self) -> np.ndarray:
        """Color from the PCA projection of anchor features.

        Features are centered, SVD is applied, and the array is projected onto
        the first three right singular vectors; each component is normalized by
        the 1-99 percentiles to [0,1] and assigned to an RGB channel. The result
        is cached because the features in the container do not change.
        """
        if self._pca_cache is None:
            features = self._features.astype(np.float64)
            features = features - features.mean(axis=0, keepdims=True)
            _, _, vh = np.linalg.svd(features, full_matrices=False)
            k = min(3, vh.shape[0])
            projection = features @ vh[:k].T
            if k < 3:
                projection = np.pad(projection, ((0, 0), (0, 3 - k)))
            colors = np.empty_like(projection, dtype=np.float32)
            for i in range(3):
                column = projection[:, i]
                lo, hi = np.percentile(column, [1.0, 99.0])
                if hi - lo < 1e-9:
                    hi = lo + 1.0
                colors[:, i] = np.clip((column - lo) / (hi - lo), 0.0, 1.0)
            self._pca_cache = colors
        return self._pca_cache.copy()

    def _depth_colors(self, reference_position: np.ndarray) -> np.ndarray:
        """Depth map relative to position ``reference_position``.

        The depth of a Gaussian is the distance from its center to the reference.
        Values are normalized to [0,1] via 1-99 percentiles (robust to outliers)
        and colored with a viridis colormap: close -- dark blue, far -- yellow.
        """
        distances = np.linalg.norm(
            self._positions.astype(np.float64)
            - np.asarray(reference_position, dtype=np.float64),
            axis=1,
        )
        lo, hi = np.percentile(distances, [1.0, 99.0])
        if hi - lo < 1e-9:
            hi = lo + 1.0
        depth01 = np.clip((distances - lo) / (hi - lo), 0.0, 1.0)
        return self._apply_colormap(depth01, self.cmap)

    def _normal_vectors(self) -> np.ndarray:
        """Unit normals of the Gaussians -- the direction of the ellipsoid's smallest radius.

        For covariance ``Sigma``, the normal is the eigenvector corresponding to
        the smallest eigenvalue (the axis of the "flattest" measurement of the
        ellipsoid); it is also the axis of smallest scale ``R @ e_{argmin s}``.
        The eigendecomposition gives an arbitrary sign, so with
        ``orient_normals_outward=True`` the normals are flipped away from the
        scene's center of mass (a heuristic for convex scenes). The result is
        cached.
        """
        if self._normal_cache is None:
            _, eigenvectors = np.linalg.eigh(self._covariances.astype(np.float64))
            normals = eigenvectors[:, :, 0]  # smallest eigenvalue
            if self._orient_normals_outward:
                outward = (
                    self._positions.astype(np.float64)
                    - self._positions.astype(np.float64).mean(axis=0, keepdims=True)
                )
                flip = np.einsum("ni,ni->n", normals, outward) < 0.0
                normals[flip] *= -1.0
            self._normal_cache = normals.astype(np.float32)
        return self._normal_cache

    def _combine_cosine_maps(
        self,
        cosine_map: np.ndarray,
        viewpoints_distances: np.ndarray,
    ) -> np.ndarray:
        """Combine the cosine maps of individual viewpoints into one map (N,).

        * ``mean``: plain average (each viewpoint contributes equally).
        * ``distance_weighted``: weights ``softmax(-dist / median(dist))`` --
        viewpoints closer to the Gaussian influence its color more strongly;
        this is "akin to interpolating" between individual maps.

        Args:
            cosine_map: (N, M) maps of cosines n·d for M viewpoints.
            viewpoints_distances: (N, M) distances to viewpoints.
        """
        if cosine_map.shape[1] == 1:
            return cosine_map[:, 0]
        if self._viewpoint_blend == "distance_weighted":
            scale = max(float(np.median(viewpoints_distances)), 1e-6)
            weights = np.exp(-viewpoints_distances / scale)
            weights = weights / weights.sum(axis=1, keepdims=True)
            return (weights * cosine_map).sum(axis=1)
        return cosine_map.mean(axis=1)

    def _view_direction_colors(self) -> Optional[np.ndarray]:
        """Map of the angular distance between Gaussian normals and viewpoints.

        For each viewpoint j, the cosine of the angle between the normal n_i and
        the direction to the viewpoint center ``d_ij`` is computed. Individual
        maps are combined into one (see :meth:`_combine_cosine_maps`), then
        ``arccos`` converts the cosine to an angle [0, pi], normalized to [0,1]:
        0 -- the Gaussian "looks" straight at the viewpoint, 1 -- it faces away.
        The value is colored with a viridis colormap. Returns ``None`` if there
        are no viewpoints.
        """
        with self._lock:
            if not self._viewpoints:
                return None
            viewpoint_positions = np.array(
                [p for p, _ in self._viewpoints], dtype=np.float64
            )
            blend = self._viewpoint_blend
        positions = self._positions.astype(np.float64)
        normals = self._normal_vectors().astype(np.float64)

        delta = viewpoint_positions[None, :, :] - positions[:, None, :]  # (N,M,3)
        dist = np.linalg.norm(delta, axis=-1, keepdims=True)
        direction = delta / np.maximum(dist, 1e-9)
        cosine_map = np.einsum("njk,nk->nj", direction, normals)  # (N,M) in [-1,1]
        combined = self._combine_cosine_maps(cosine_map, dist[..., 0])
        angular = np.arccos(np.clip(combined, -1.0, 1.0)) / np.pi  # [0,1]
        return self._apply_colormap(angular, self.cmap)

    @staticmethod
    def _apply_colormap(values01: np.ndarray, cmap: str="turbo") -> np.ndarray:
        """Color values in [0,1] into float32 RGB; viridis from matplotlib, or the
        built-in gradient if matplotlib is unavailable."""
        values01 = np.clip(values01, 0.0, 1.0)
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.cm as cm
            if not hasattr(cm, cmap):
                warn(f"unknow colormap: {cmap}")
                cmap = "turbo"
            return getattr(cm, cmap)(values01)[:, :3].astype(np.float32)
        except Exception:
            return _fallback_colormap(values01).astype(np.float32)

    def _status_markdown(self) -> str:
        """Current visualizer state as a markdown string."""
        with self._lock:
            mode = self._mode
            n_viewpoints = len(self._viewpoints)
        return (
            f"**Mode:** `{mode}`  |  **Viewpoints:** `{n_viewpoints}`\n\n"
            "- `rgb` -- original colors\n"
            "- `features_pca` -- PCA of anchor features as RGB\n"
            "- `depth` -- depth from the current camera\n"
            "- `view_direction` -- angular distance to viewpoints\n\n"
            "_Press **Save current viewpoint** to capture the camera._"
        )

    def _update_status_markdown(self) -> None:
        """Update the markdown status for all connected clients."""
        text = self._status_markdown()
        for handle in list(self._status_handles.values()):
            try:
                handle.content = text
            except Exception:
                pass


def visualize(container, **kwargs) -> GaussianSplatVisualizer:
    """Create a :class:`GaussianSplatVisualizer` for ``container`` and run it immediately.

    A one-line entry point that accepts a container directly; additional
    ``**kwargs`` are forwarded to the visualizer constructor.
    """
    visualizer = GaussianSplatVisualizer(container, **kwargs)
    visualizer.run()
    return visualizer

