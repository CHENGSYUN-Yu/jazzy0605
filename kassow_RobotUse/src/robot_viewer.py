"""
3D robot viewer using yourdfpy + pyrender (EGL offscreen).
Rendering runs in a dedicated background thread to avoid
conflicting with dearpygui's own OpenGL context.
"""
import os
import threading
import queue
import numpy as np

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import trimesh
import pyrender
import yourdfpy

_URDF_PATH = os.path.expanduser(
    "~/ros2_ws/install/kr810_for_urdf_2robot/share/"
    "kr810_for_urdf_2robot/urdf/KR810_for_urdf_2robot.SLDASM.urdf"
)
_MESH_BASE = os.path.expanduser(
    "~/ros2_ws/install/kr810_for_urdf_2robot/share/kr810_for_urdf_2robot"
)

_JOINT_NAMES = [
    "Left_joint_1", "Left_joint_2", "Left_joint_3", "Left_jont_4",
    "Left_joint_5", "Left_joint_6", "Left_joint_7",
    "Right_joint_1", "Right_joint_2", "Right_joint_3", "Right_joint_4",
    "Right_joint_5", "Right_joint_6", "Right_link_7",
]

VIEW_W, VIEW_H = 640, 480


def _filename_handler(fname: str = "", **_) -> str:
    if fname.startswith("package://kr810_for_urdf_2robot/"):
        return os.path.join(_MESH_BASE, fname.replace("package://kr810_for_urdf_2robot/", ""))
    return fname


def _spherical_cam_pose(centroid: np.ndarray,
                         azimuth: float, elevation: float,
                         distance: float) -> np.ndarray:
    """Build camera pose from spherical coordinates around centroid."""
    az, el = azimuth, elevation
    cam_pos = centroid + distance * np.array([
        np.cos(el) * np.sin(az),
        np.cos(el) * (-np.cos(az)),
        np.sin(el),
    ])
    fwd = centroid - cam_pos
    norm = np.linalg.norm(fwd)
    if norm < 1e-6:
        return np.eye(4)
    fwd /= norm
    world_up = np.array([0.0, 0.0, 1.0])
    right = np.cross(fwd, world_up)
    r_norm = np.linalg.norm(right)
    if r_norm < 1e-6:
        right = np.array([1.0, 0.0, 0.0])
    else:
        right /= r_norm
    up = np.cross(right, fwd)
    M = np.eye(4)
    M[:3, 0] = right
    M[:3, 1] = up
    M[:3, 2] = -fwd
    M[:3, 3] = cam_pos
    return M


class RobotViewer:
    VIEW_W = VIEW_W
    VIEW_H = VIEW_H

    # default camera spherical coordinates
    CAM_AZ_DEFAULT  =  0.8    # radians (azimuth)
    CAM_EL_DEFAULT  =  0.4    # radians (elevation)
    CAM_DIST_DEFAULT = 2.5    # metres

    def __init__(self):
        self._angles: dict[str, float] = {j: 0.0 for j in _JOINT_NAMES}
        self._cam = {
            "az":   self.CAM_AZ_DEFAULT,
            "el":   self.CAM_EL_DEFAULT,
            "dist": self.CAM_DIST_DEFAULT,
        }
        self._state_lock = threading.Lock()

        # render thread puts RGBA float32 lists here; main thread picks up
        self._result_queue: queue.SimpleQueue = queue.SimpleQueue()

        self._ready = False
        self._error: str | None = None

        self._render_event = threading.Event()
        self._stop_event   = threading.Event()

        self._thread = threading.Thread(target=self._render_loop, daemon=True)
        self._thread.start()

    # ── public API ────────────────────────────────────────────────────────────

    def stop(self) -> None:
        self._stop_event.set()
        self._render_event.set()  # 喚醒 render loop 讓它偵測到 stop

    def set_angles(self, angles: dict[str, float]) -> None:
        with self._state_lock:
            self._angles.update(angles)
        self._render_event.set()

    def set_camera(self, az: float | None = None,
                   el: float | None = None,
                   dist: float | None = None) -> None:
        with self._state_lock:
            if az   is not None: self._cam["az"]   = az
            if el   is not None: self._cam["el"]   = el
            if dist is not None: self._cam["dist"] = dist
        self._render_event.set()

    @property
    def ready(self) -> bool:
        return self._ready

    @property
    def error(self) -> str | None:
        return self._error

    def get_texture(self) -> list | None:
        """
        Return latest rendered RGBA float32 list for dpg.set_value(),
        or None if nothing new since last call.
        """
        # drain queue, keep only the newest frame
        new = None
        while True:
            try:
                new = self._result_queue.get_nowait()
            except queue.Empty:
                break
        if new is not None:
            self._latest_tex = new
            return new
        return None

    # ── render loop (runs entirely in background thread) ──────────────────────

    def _render_loop(self) -> None:
        try:
            robot, pr_meshes, centroid = self._load()
        except Exception as e:
            self._error = str(e)
            return

        renderer = pyrender.OffscreenRenderer(VIEW_W, VIEW_H)
        self._ready = True
        self._render_event.set()    # trigger first render immediately

        while not self._stop_event.is_set():
            self._render_event.wait()
            self._render_event.clear()
            if self._stop_event.is_set():
                break

            with self._state_lock:
                cfg = dict(self._angles)
                cam = dict(self._cam)

            cam_pose = _spherical_cam_pose(
                centroid, cam["az"], cam["el"], cam["dist"]
            )

            try:
                tex = self._do_render(robot, pr_meshes, cam_pose, renderer, cfg)
                self._result_queue.put(tex)
            except Exception as e:
                self._error = str(e)

        renderer.delete()

    def _load(self):
        robot = yourdfpy.URDF.load(_URDF_PATH, filename_handler=_filename_handler)
        tm_scene = robot.scene

        # Build link-name → rgba colour map from URDF material definitions
        color_map: dict[str, np.ndarray] = {}
        for link_name, link in robot.link_map.items():
            for vis in (link.visuals or []):
                if vis.material and vis.material.color is not None:
                    color_map[link_name] = np.array(vis.material.color.rgba,
                                                     dtype=np.float32)
                    break

        # Manual colour overrides
        color_map["base_link"] = np.array([0.15, 0.45, 0.90, 1.0], dtype=np.float32)

        _RED = np.array([0.85, 0.10, 0.10, 1.0], dtype=np.float32)
        for prefix in ("Left_link_", "Right_link_"):
            for i in (2, 4, 6):
                color_map[f"{prefix}{i}"] = _RED

        pr_meshes = []
        for name, geom in tm_scene.geometry.items():
            if not isinstance(geom, trimesh.Trimesh):
                continue
            # geometry name is "Left_link_1.STL" → link name "Left_link_1"
            link_name = name.replace(".STL", "")
            rgba = color_map.get(link_name, np.array([0.8, 0.82, 0.93, 1.0]))

            material = pyrender.MetallicRoughnessMaterial(
                baseColorFactor=rgba.tolist(),
                metallicFactor=0.15,
                roughnessFactor=0.6,
                alphaMode="OPAQUE",
            )
            pr_mesh = pyrender.Mesh.from_trimesh(geom, smooth=False,
                                                  material=material)
            pr_meshes.append((name, pr_mesh))

        centroid = np.array(tm_scene.centroid)
        return robot, pr_meshes, centroid

    @staticmethod
    def _do_render(robot, pr_meshes, cam_pose, renderer, cfg) -> list:
        robot.update_cfg(cfg)
        tm_scene = robot.scene

        pr_scene = pyrender.Scene(
            ambient_light=[0.4, 0.4, 0.4],
            bg_color=[0.0, 0.0, 0.0, 1.0],
        )
        for name, pr_mesh in pr_meshes:
            node_names = tm_scene.graph.geometry_nodes.get(name, [])
            T = tm_scene.graph[node_names[0]][0] if node_names else np.eye(4)
            pr_scene.add(pr_mesh, pose=T)

        pr_scene.add(pyrender.PerspectiveCamera(yfov=np.pi / 3), pose=cam_pose)
        pr_scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=6.0),
                     pose=cam_pose)

        color, _ = renderer.render(pr_scene)
        rgba = np.ones((VIEW_H, VIEW_W, 4), dtype=np.float32)
        rgba[:, :, :3] = color.astype(np.float32) / 255.0
        return rgba.flatten().tolist()
