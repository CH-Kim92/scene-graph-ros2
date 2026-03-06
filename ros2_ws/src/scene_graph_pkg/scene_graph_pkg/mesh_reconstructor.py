"""
mesh_reconstructor.py
─────────────────────
Reconstructs a surface mesh of the scene from RGBD frames using
Open3D's TSDF volume integration.

Since the camera is FIXED, no pose estimation is needed –
all frames are integrated at the identity pose.

Provides:
  - integrate_frame()    : add an RGBD frame to the TSDF volume
  - extract_mesh()       : Marching-Cubes mesh from the volume
  - extract_pointcloud() : coloured point cloud from the volume
  - reset()              : clear the volume
"""

from __future__ import annotations
import numpy as np
import open3d as o3d
import threading
import time
from typing import Optional, Tuple


class MeshReconstructor:
    """
    TSDF-based mesh reconstruction for a fixed-position RGBD camera.

    Parameters
    ----------
    voxel_length    : float – voxel size (m). 0.02 = 2 cm, good for table-top.
    sdf_trunc       : float – truncation distance (m). Usually 4× voxel_length.
    depth_scale     : float – conversion factor from raw depth to metres.
                              pyrealsense2 already outputs metres → 1.0.
    depth_max       : float – max depth to integrate (m).
    integrate_every : int   – integrate every N frames (1 = every frame).
    """

    def __init__(
        self,
        voxel_length: float    = 0.02,
        sdf_trunc: float       = 0.08,
        depth_scale: float     = 1.0,
        depth_max: float       = 4.0,
        integrate_every: int   = 2,
    ):
        self.depth_scale     = depth_scale
        self.depth_max       = depth_max
        self.integrate_every = integrate_every
        self._frame_count    = 0
        self._lock           = threading.Lock()

        self.volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=voxel_length,
            sdf_trunc=sdf_trunc,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
        )

        # Identity pose – camera does not move
        self._pose = np.eye(4)

        print(f"[MeshReconstructor] TSDF volume ready. "
              f"voxel={voxel_length*100:.1f}cm, trunc={sdf_trunc*100:.1f}cm")

    # ── Public API ────────────────────────────────────────────────────────────

    def integrate_frame(
        self,
        rgb: np.ndarray,          # H×W×3  uint8
        depth: np.ndarray,        # H×W    float32 metres
        intrinsics: dict,         # {fx, fy, cx, cy, width, height}
    ) -> None:
        """Integrate one RGBD frame into the TSDF volume (thread-safe)."""
        self._frame_count += 1
        if self._frame_count % self.integrate_every != 0:
            return

        # Clamp depth to valid range
        depth_clamped = np.where(
            (depth > 0.1) & (depth < self.depth_max),
            depth, 0.0
        ).astype(np.float32)

        o3d_rgb   = o3d.geometry.Image(rgb.astype(np.uint8))
        o3d_depth = o3d.geometry.Image(depth_clamped)
        rgbd      = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d_rgb,
            o3d_depth,
            depth_scale=1.0 / self.depth_scale,
            depth_trunc=self.depth_max,
            convert_rgb_to_intensity=False,
        )

        intr = o3d.camera.PinholeCameraIntrinsic(
            width=intrinsics.get('width', rgb.shape[1]),
            height=intrinsics.get('height', rgb.shape[0]),
            fx=intrinsics['fx'],
            fy=intrinsics['fy'],
            cx=intrinsics['cx'],
            cy=intrinsics['cy'],
        )

        with self._lock:
            self.volume.integrate(rgbd, intr, np.linalg.inv(self._pose))

    def extract_mesh(
        self,
        simplify: bool = True,
        target_triangles: int = 80_000,
    ) -> Optional[o3d.geometry.TriangleMesh]:
        """Run Marching Cubes on the TSDF volume and return a triangle mesh."""
        with self._lock:
            mesh = self.volume.extract_triangle_mesh()

        if mesh is None or len(mesh.vertices) == 0:
            return None

        mesh.compute_vertex_normals()

        if simplify and len(mesh.triangles) > target_triangles:
            mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=target_triangles)
            mesh.compute_vertex_normals()

        # Remove statistical outliers from the mesh via point cloud
        mesh = self._clean_mesh(mesh)
        return mesh

    def extract_pointcloud(
        self,
        voxel_downsample: float = 0.01,
    ) -> o3d.geometry.PointCloud:
        """Extract a dense coloured point cloud from the TSDF volume."""
        with self._lock:
            pcd = self.volume.extract_point_cloud()

        if voxel_downsample > 0:
            pcd = pcd.voxel_down_sample(voxel_size=voxel_downsample)

        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        return pcd

    def reset(self) -> None:
        """Clear the TSDF volume (e.g. after a scene change)."""
        with self._lock:
            self.volume = o3d.pipelines.integration.ScalableTSDFVolume(
                voxel_length=self.volume.voxel_length,
                sdf_trunc=self.volume.sdf_trunc,
                color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
            )
        self._frame_count = 0
        print("[MeshReconstructor] Volume reset.")

    def mesh_to_dict(
        self,
        mesh: o3d.geometry.TriangleMesh,
        max_vertices: int = 10_000,
    ) -> dict:
        """
        Convert mesh to a compact JSON-serialisable dict for WebSocket streaming.
        Vertices are subsampled for bandwidth.
        """
        verts  = np.asarray(mesh.vertices)
        tris   = np.asarray(mesh.triangles)
        colors = np.asarray(mesh.vertex_colors) if mesh.has_vertex_colors() else None
        norms  = np.asarray(mesh.vertex_normals) if mesh.has_vertex_normals() else None

        # Subsample if too large
        if len(verts) > max_vertices:
            idx    = np.random.choice(len(verts), max_vertices, replace=False)
            verts  = verts[idx]
            colors = colors[idx] if colors is not None else None
            norms  = norms[idx]  if norms  is not None else None
            tris   = None        # Can't remap triangles after vertex subsample

        return {
            "vertices": verts.flatten().tolist(),
            "triangles": tris.flatten().tolist() if tris is not None else [],
            "colors":   (colors * 255).astype(np.uint8).flatten().tolist()
                        if colors is not None else [],
            "normals":  norms.flatten().tolist() if norms is not None else [],
            "vertex_count":   len(verts),
            "triangle_count": len(tris) if tris is not None else 0,
        }

    def pointcloud_to_dict(
        self,
        pcd: o3d.geometry.PointCloud,
        max_points: int = 50_000,
    ) -> dict:
        """Convert point cloud to compact dict for streaming."""
        pts = np.asarray(pcd.points)
        clr = np.asarray(pcd.colors) if pcd.has_colors() else None

        if len(pts) > max_points:
            idx = np.random.choice(len(pts), max_points, replace=False)
            pts = pts[idx]
            clr = clr[idx] if clr is not None else None

        return {
            "points": pts.flatten().tolist(),
            "colors": (clr * 255).astype(np.uint8).flatten().tolist()
                      if clr is not None else [],
            "count":  len(pts),
        }

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _clean_mesh(mesh: o3d.geometry.TriangleMesh) -> o3d.geometry.TriangleMesh:
        mesh.remove_degenerate_triangles()
        mesh.remove_duplicated_triangles()
        mesh.remove_duplicated_vertices()
        mesh.remove_non_manifold_edges()
        return mesh
