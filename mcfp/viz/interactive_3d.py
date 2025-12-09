import numpy as np
import pyvista as pv
from pathlib import Path
from typing import Dict, List, Optional
import difflib
import scipy.spatial.transform as st
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from mcfp.sim.robot_model import RobotModel
from mcfp.utils.logging import setup_logger

logger = setup_logger(name="mcfp.viz.interactive_3d")

class InteractiveCapabilityViz:
    """
    Interactive 3D Visualization tool for MCFP Feasibility Fields.
    
    Features:
    - Hard Filter: Only points with g_ws == 1 are visualized.
    - Layout: Left (Charts 40%), Center (Charts 40%), Right (Sidebar 20%).
    - Style: Academic style with desaturated colors (RdYlGn, inferno).
    """

    def __init__(self, 
                 data_path: str, 
                 urdf_path: str,
                 mesh_dir: str,
                 downsample_rate: int = 1):
        self.data_path = Path(data_path)
        self.urdf_path = Path(urdf_path)
        self.mesh_dir = Path(mesh_dir)
        
        # Hyperparameters
        self.safety_cutoff = 0.2
        self.gamma_safety = 0.6
        self.gamma_dexterity = 1.8

        # 1. Load & Filter Data (Hard Filter Step)
        logger.info(f"Loading data from {self.data_path}...")
        raw_npz = np.load(self.data_path)
        
        # --- HARD FILTER LOGIC START ---
        if 'g_ws' in raw_npz:
            g_ws_raw = raw_npz['g_ws']
            # Only keep points where workspace reachability is 1
            valid_mask = (g_ws_raw == 1)
            count_before = len(g_ws_raw)
            count_after = np.sum(valid_mask)
            logger.info(f"Applying Hard Filter (g_ws==1): {count_before} -> {count_after} points retained.")
            
            # Create a new filtered dictionary
            self.filtered_data = {}
            for key in raw_npz.files:
                arr = raw_npz[key]
                # Filter only arrays that match the number of points (metric arrays)
                if hasattr(arr, 'shape') and arr.shape[0] == count_before:
                    self.filtered_data[key] = arr[valid_mask]
                else:
                    # Keep metadata/scalars as is
                    self.filtered_data[key] = arr
        else:
            logger.warning("g_ws not found in data! Skipping hard filter.")
            self.filtered_data = dict(raw_npz)
        # --- HARD FILTER LOGIC END ---

        self.q_star_full = self.filtered_data.get('q_star', None)
        
        # 2. Compute Scores (Based on filtered data)
        self.scores = self._compute_scores(self.filtered_data)
        
        # 3. Create Geometry
        self.point_cloud = self._create_point_cloud_source()
        self.volume = self._interpolate_continuous_volume()
        self.bounds = self.volume.bounds

        # 4. Init Robot
        logger.info(f"Loading RobotModel: {self.urdf_path}")
        self.robot = RobotModel(self.urdf_path, logger=logger)
        self.mesh_map = self._bind_meshes_to_links()
        
        # State & Storage
        self.robot_actors = {
            'top_left': {}, 'top_right': {}, 
            'bottom_left': {}, 'bottom_right': {}
        }
        self.slice_actors = {
            'left': {'x': None, 'y': None, 'z': None},
            'right': {'x': None, 'y': None, 'z': None}
        }
        self.threshold_actors = {'top_left': None, 'top_right': None}
        self.cursor_actors = {} 
        self.hud_actor = None   
        self.slider_widgets = [] # IMPORTANT: Keep references to prevent GC

        self.current_slices = {
            'x': (self.bounds[0] + self.bounds[1]) / 2.0,
            'y': (self.bounds[2] + self.bounds[3]) / 2.0,
            'z': (self.bounds[4] + self.bounds[5]) / 2.0
        }
        self.current_thresh = {'Safety': 0.0, 'Dexterity': 0.0}

    def _get_desaturated_cmap(self, cmap_name, saturation=0.6):
        """Helper to create soft, academic-style colormaps."""
        cmap = plt.get_cmap(cmap_name)
        colors = cmap(np.linspace(0, 1, 256))
        hsv = mcolors.rgb_to_hsv(colors[:, :3])
        hsv[:, 1] *= saturation # Desaturate
        new_colors = mcolors.hsv_to_rgb(hsv)
        if colors.shape[1] == 4:
            new_colors = np.hstack([new_colors, colors[:, 3:]])
        return mcolors.ListedColormap(new_colors, name=f"{cmap_name}_soft")

    def _normalize_sub_metric(self, data: np.ndarray, metric_name: str) -> np.ndarray:
        clean_data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
        clean_data = np.maximum(clean_data, 0.0)
        if len(clean_data) == 0: return clean_data
        q99 = np.percentile(clean_data, 99)
        if q99 < 1e-9: return np.zeros_like(clean_data)
        return np.clip(clean_data / q99, 0.0, 1.0)

    def _compute_scores(self, d) -> Dict[str, np.ndarray]:
        n = len(d['cell_centers'])
        
        # Safety components
        # Note: g_ws is NOT used for calculation anymore, only for hard filtering in __init__
        n_self = self._normalize_sub_metric(d.get('g_self', np.zeros(n)), 'g_self')
        n_lim = self._normalize_sub_metric(d.get('g_lim', np.zeros(n)), 'g_lim')
        
        # Revised Logic: Safety is purely determined by collision and limits
        raw_safety = np.minimum(n_self, n_lim)
        
        s_range = 1.0 - self.safety_cutoff
        if s_range > 0:
            safety_rescaled = np.clip((raw_safety - self.safety_cutoff) / s_range, 0.0, 1.0)
        else:
            safety_rescaled = raw_safety
        safety_final = np.power(safety_rescaled, self.gamma_safety)

        # Dexterity components
        n_man = self._normalize_sub_metric(d.get('g_man', np.zeros(n)), 'g_man')
        n_iso = self._normalize_sub_metric(d.get('g_iso', np.zeros(n)), 'g_iso')
        n_sigma = self._normalize_sub_metric(d.get('g_sigma', np.zeros(n)), 'g_sigma')
        n_red = self._normalize_sub_metric(d.get('g_red', np.zeros(n)), 'g_red')
        n_rot = self._normalize_sub_metric(d.get('g_rot', np.zeros(n)), 'g_rot')
        
        w = {'man': 0.4, 'iso': 0.15, 'sigma': 0.15, 'red': 0.15, 'rot': 0.15} 
        raw_dex = (np.power(n_man, w['man']) * np.power(n_iso, w['iso']) * np.power(n_sigma, w['sigma'])* np.power(n_red, w['red'])* np.power(n_rot, w['rot']))
        
        # Gate dexterity by safety (still useful visually)
        dex_gated = raw_dex * (raw_safety > 0.01).astype(np.float32)
        dex_final = np.power(dex_gated, self.gamma_dexterity)

        # Return scores (g_ws is returned for debug if needed, but it should be all 1s now)
        return {"Safety": safety_final, "Dexterity": dex_final}

    def _create_point_cloud_source(self) -> pv.PolyData:
        # Uses filtered data
        centers = self.filtered_data["cell_centers"]
        pdata = pv.PolyData(centers.astype(np.float32))
        pdata.point_data["Safety"] = self.scores["Safety"]
        pdata.point_data["Dexterity"] = self.scores["Dexterity"]
        # Indices now refer to the filtered array, keeping sync with q_star
        pdata.point_data["original_index"] = np.arange(len(centers))
        return pdata

    def _interpolate_continuous_volume(self, resolution=80) -> pv.ImageData:
        logger.info(f"Interpolating continuous field (Grid: {resolution}^3)...")
        bounds = self.point_cloud.bounds
        pad = 0.05
        dims = (resolution, resolution, resolution)
        grid = pv.ImageData()
        grid.dimensions = dims
        grid.origin = (bounds[0]-pad, bounds[2]-pad, bounds[4]-pad)
        grid.spacing = (
            (bounds[1]-bounds[0]+2*pad)/(resolution-1),
            (bounds[3]-bounds[2]+2*pad)/(resolution-1),
            (bounds[5]-bounds[4]+2*pad)/(resolution-1),
        )
        
        # Interpolation will naturally be tighter since source points are pre-filtered
        raw_vol = grid.interpolate(self.point_cloud, radius=max(grid.spacing) * 3.0, sharpness=2.0, strategy='null_value', null_value=0.0)

        def apply_gaussian(name, sigma):
            arr = raw_vol[name].copy()
            arr_3d = arr.reshape(dims, order='F')
            smooth_3d = ndimage.gaussian_filter(arr_3d, sigma=sigma)
            return smooth_3d.ravel(order='F')

        final_vol = raw_vol.copy(deep=True) 
        final_vol["Safety"] = apply_gaussian("Safety", 1.5)
        final_vol["Dexterity"] = apply_gaussian("Dexterity", 1.5)
        return final_vol

    def _bind_meshes_to_links(self) -> Dict[str, pv.PolyData]:
        """
        Load meshes and intelligently map them to URDF link names using fuzzy matching.
        """
        temp_q = np.zeros(self.robot.num_joints)
        active_poses = self.robot.link_poses(temp_q)
        urdf_link_names = list(active_poses.keys())

        stls = list(self.mesh_dir.glob("*.stl"))
        mapping = {}
        
        logger.info(f"Mapping {len(stls)} mesh files to {len(urdf_link_names)} URDF links...")

        for stl_path in stls:
            filename = stl_path.stem.lower() # e.g., "link_1" or "base_link"
            mesh = None
            try:
                mesh = pv.read(stl_path)
            except Exception as e:
                logger.warning(f"Failed to load {stl_path}: {e}")
                continue

            matched_name = None

            if "base" in filename:
                matched_name = "base_link"

            elif filename in urdf_link_names:
                matched_name = filename

            else:
                clean_fname = filename.replace("_", "").replace("-", "")

                for urdf_name in urdf_link_names:
                    clean_urdf = urdf_name.lower().replace("_", "").replace("-", "")
                    if clean_fname == clean_urdf:
                        matched_name = urdf_name
                        break

                if matched_name is None:
                    matches = difflib.get_close_matches(filename, urdf_link_names, n=1, cutoff=0.6)
                    if matches:
                        matched_name = matches[0]

            if matched_name:
                mapping[matched_name] = mesh
                # logger.info(f"  [Match] File '{stl_path.name}' -> Link '{matched_name}'")
            else:
                mapping[filename] = mesh
                logger.warning(f"  [No Match] File '{stl_path.name}' loaded as '{filename}' (No kinematic link found)")
                
        return mapping

    def _update_robot_pose(self, q: np.ndarray, view_key: str):
        """
        Update robot pose. Handles base_link separately.
        """

        poses = self.robot.link_poses(q)

        for link_name, actor in self.robot_actors.get(view_key, {}).items():
            mat = np.eye(4) 
            if "base" in link_name.lower():
                pass
            elif link_name in poses:
                pos, quat = poses[link_name]
                mat[:3, :3] = st.Rotation.from_quat(quat).as_matrix()
                mat[:3, 3] = pos
            else:
                for p_key in poses.keys():
                    if link_name.replace("_","") == p_key.replace("_",""):
                        pos, quat = poses[p_key]
                        mat[:3, :3] = st.Rotation.from_quat(quat).as_matrix()
                        mat[:3, 3] = pos
                        break

            actor.user_matrix = mat

    def _add_robot_to_view(self, plotter, view_key: str):
        """Initial rendering of robot meshes."""
        q_zeros = np.zeros(self.robot.num_joints)
        
        for link, mesh in self.mesh_map.items():

            actor = plotter.add_mesh(
                mesh.copy(), color="#E0E0E0", opacity=1.0, 
                smooth_shading=True, name=f"robot_{link}_{view_key}",
                specular=0.5, specular_power=15
            )
            self.robot_actors[view_key][link] = actor

        self._update_robot_pose(q_zeros, view_key)

    # --- UPDATES ---
    def _update_threshold_cloud(self, metric: str, value: float):
        self.current_thresh[metric] = value
        if metric == 'Safety':
            subplot_idx = (0, 0); view_key = 'top_left'
            # Soft RdYlGn for Safety
            cmap = self._get_desaturated_cmap("RdYlGn", saturation=0.65)
        else:
            subplot_idx = (0, 1); view_key = 'top_right'
            # Soft Inferno for Dexterity
            cmap = self._get_desaturated_cmap("inferno", saturation=0.8)
        
        self.plotter.subplot(*subplot_idx)
        if self.threshold_actors[view_key]: self.plotter.remove_actor(self.threshold_actors[view_key])
        
        try: thresh_mesh = self.point_cloud.threshold(value, scalars=metric)
        except: thresh_mesh = pv.PolyData() 
        
        if thresh_mesh.n_points > 0:
            self.threshold_actors[view_key] = self.plotter.add_mesh(
                thresh_mesh, scalars=metric, cmap=cmap, clim=[0.0, 1.0],
                render_points_as_spheres=True, point_size=3, 
                opacity=0.6, # Soft opacity
                show_scalar_bar=False, lighting=False
            )
        else: self.threshold_actors[view_key] = None

    def _update_single_plane(self, axis: str, value: float):
        self.current_slices[axis] = value
        center = list(self.volume.center)
        axis_idx = {'x':0, 'y':1, 'z':2}[axis]
        center[axis_idx] = value
        
        # Note: g_ws check removed from clip_scalar since data is pre-filtered
        raw_slice = self.volume.slice(normal=axis, origin=center)
        valid_slice = raw_slice # No additional clipping needed

        # Left (Safety)
        self.plotter.subplot(1, 0)
        if self.slice_actors['left'][axis]: self.plotter.remove_actor(self.slice_actors['left'][axis])
        if valid_slice.n_points > 0:
            try: safe_mesh = valid_slice.clip_scalar(scalars="Safety", value=0.001, invert=False)
            except: safe_mesh = pv.PolyData()
            if safe_mesh.n_points > 0:
                self.slice_actors['left'][axis] = self.plotter.add_mesh(
                    safe_mesh, scalars="Safety", 
                    cmap=self._get_desaturated_cmap("RdYlGn", saturation=0.65), 
                    clim=[0.0, 1.0],
                    opacity=0.85, # Semi-transparent slice
                    show_scalar_bar=False, lighting=False
                )

        # Right (Dexterity)
        self.plotter.subplot(1, 1)
        if self.slice_actors['right'][axis]: self.plotter.remove_actor(self.slice_actors['right'][axis])
        if valid_slice.n_points > 0:
            try: dex_mesh = valid_slice.clip_scalar(scalars="Dexterity", value=0.001, invert=False)
            except: dex_mesh = pv.PolyData()
            if dex_mesh.n_points > 0:
                self.slice_actors['right'][axis] = self.plotter.add_mesh(
                    dex_mesh, scalars="Dexterity", 
                    cmap=self._get_desaturated_cmap("inferno", saturation=0.8),
                    clim=[0.0, 1.0],
                    opacity=0.85, # Semi-transparent slice
                    show_scalar_bar=False, lighting=False
                )
        self._update_probe()

    def _update_probe(self):
        """Update HUD info (Top-Right, Index 2)."""
        probe_pos = (self.current_slices['x'], self.current_slices['y'], self.current_slices['z'])
        idx = self.point_cloud.find_closest_point(probe_pos)
        closest_p = self.point_cloud.points[idx]
        dist = np.linalg.norm(np.array(probe_pos) - closest_p)
        
        safety = self.point_cloud.point_data["Safety"][idx]
        is_valid = (dist < 0.05) and (safety > 0.001)

        q_str = "N/A"
        if is_valid and self.q_star_full is not None:
            orig_idx = self.point_cloud.point_data["original_index"][idx]
            q = self.q_star_full[orig_idx]
            q_formatted = [f"{val:.2f}" for val in q]
            if len(q_formatted) > 3:
                mid = len(q_formatted)//2
                q_str = f"[{', '.join(q_formatted[:mid])},\n   {', '.join(q_formatted[mid:])}]"
            else:
                q_str = f"[{', '.join(q_formatted)}]"
            
            for key in ['top_left', 'top_right', 'bottom_left', 'bottom_right']:
                self._update_robot_pose(q, key)

        dex = self.point_cloud.point_data["Dexterity"][idx]
        
        if is_valid:
            status_line = "STATUS: VALID"
            s_val = f"{safety:.3f}"
            d_val = f"{dex:.3f}"
        else:
            status_line = "STATUS: VOID / UNSAFE"
            s_val = "---"
            d_val = "---"

        text_block = (
            f"MCFP INSPECTION\n"
            f"===============\n"
            f"{status_line}\n\n"
            f"COORDINATES:\n"
            f"X: {probe_pos[0]:.3f}\n"
            f"Y: {probe_pos[1]:.3f}\n"
            f"Z: {probe_pos[2]:.3f}\n\n"
            f"SCORES:\n"
            f"Safety: {s_val}\n"
            f"Dexterity: {d_val}\n\n"
            f"JOINTS (q*):\n"
            f"{q_str}"
        )
        
        # Explicitly focus on Index 2 (Top Right)
        self.plotter.subplot(0, 2)
        if self.hud_actor: self.plotter.remove_actor(self.hud_actor)
        
        self.hud_actor = self.plotter.add_text(
            text_block, 
            position='upper_left', 
            font_size=10, 
            color='black', 
            font='courier',
            name='hud_info'
        )

        self._update_cursor(probe_pos, is_valid)

    def _update_cursor(self, pos, valid):
        color = '#228B22' if valid else '#B22222'
        sphere = pv.Sphere(radius=0.015, center=pos)
        for r, c, key in [(0, 0, 'top_left'), (0, 1, 'top_right'), (1, 0, 'bottom_left'), (1, 1, 'bottom_right')]:
            self.plotter.subplot(r, c)
            if key in self.cursor_actors and self.cursor_actors[key]: self.plotter.remove_actor(self.cursor_actors[key])
            self.cursor_actors[key] = self.plotter.add_mesh(sphere, color=color, lighting=False, name=f'cursor_{key}')

    # --- CALLBACKS ---
    def _cb_x(self, v): self._update_single_plane('x', v)
    def _cb_y(self, v): self._update_single_plane('y', v)
    def _cb_z(self, v): self._update_single_plane('z', v)
    def _cb_thresh_safety(self, v): self._update_threshold_cloud('Safety', v)
    def _cb_thresh_dexterity(self, v): self._update_threshold_cloud('Dexterity', v)

    def _setup_subplot_style(self, row, col, title):
        self.plotter.subplot(row, col)
        self.plotter.add_text(title, font_size=12, font='times', position='upper_edge', color='black')
        self.plotter.show_grid(color='gray', font_size=8, font_family='times')
        self.plotter.show_axes()
        self.plotter.set_background('white')

    def run(self):
        pv.set_plot_theme("document")
        self.plotter = pv.Plotter(shape=(2, 3), window_size=(2000, 1000), title="MCFP Analysis Studio")
        
        # --- VIEWPORT DEFINITION (40% - 40% - 20%) ---
        self.plotter.renderers[0].viewport = (0.0, 0.5, 0.4, 1.0) # Top Left
        self.plotter.renderers[1].viewport = (0.4, 0.5, 0.8, 1.0) # Top Center
        self.plotter.renderers[2].viewport = (0.8, 0.5, 1.0, 1.0) # Top Right (Info)
        self.plotter.renderers[3].viewport = (0.0, 0.0, 0.4, 0.5) # Bot Left
        self.plotter.renderers[4].viewport = (0.4, 0.0, 0.8, 0.5) # Bot Center
        self.plotter.renderers[5].viewport = (0.8, 0.0, 1.0, 0.5) # Bot Right (Controls)

        # --- 1. CHARTS SETUP ---
        self._setup_subplot_style(0, 0, "Safety Cloud")
        self._add_robot_to_view(self.plotter, 'top_left')
        
        self._setup_subplot_style(0, 1, "Dexterity Cloud")
        self._add_robot_to_view(self.plotter, 'top_right')
        
        self._setup_subplot_style(1, 0, "Safety Field")
        self._add_robot_to_view(self.plotter, 'bottom_left')
        
        # Legend Safety (Soft colors)
        ghost_s = pv.PolyData([0,0,0]); ghost_s["Safety"] = [0.0]
        self.plotter.add_mesh(ghost_s, scalars="Safety", 
             cmap=self._get_desaturated_cmap("RdYlGn", saturation=0.65), 
             clim=[0.0, 1.0], opacity=0.0,
             show_scalar_bar=True, scalar_bar_args={"title": "Safety Score", "vertical": False, "position_x": 0.2, "position_y": 0.05, "width": 0.6, "height": 0.06, "color": "black", "font_family": "times"})

        self._setup_subplot_style(1, 1, "Dexterity Field")
        self._add_robot_to_view(self.plotter, 'bottom_right')
        
        # Legend Dexterity (Soft colors)
        ghost_d = pv.PolyData([0,0,0]); ghost_d["Dexterity"] = [0.0]
        self.plotter.add_mesh(ghost_d, scalars="Dexterity", 
             cmap=self._get_desaturated_cmap("inferno", saturation=0.8), 
             clim=[0.0, 1.0], opacity=0.0,
             show_scalar_bar=True, scalar_bar_args={"title": "Dexterity Index", "vertical": False, "position_x": 0.2, "position_y": 0.05, "width": 0.6, "height": 0.06, "color": "black", "font_family": "times"})

        # --- 2. SIDEBAR BACKGROUNDS ---
        self.plotter.subplot(0, 2); self.plotter.set_background('white')
        self.plotter.subplot(1, 2); self.plotter.set_background('white')
        
        # --- 3. SLIDERS SETUP (STRICTLY ENFORCED) ---
        ss = {'style': 'modern'}
        b = self.bounds

        # Slider 1: Safety
        self.plotter.subplot(1, 2) 
        w1 = self.plotter.add_slider_widget(self._cb_thresh_safety, [0.0, 1.0], value=0.0, title="Min Safety", 
            pointa=(0.1, 0.90), pointb=(0.9, 0.90), **ss)
        self.slider_widgets.append(w1)

        # Slider 2: Dexterity
        self.plotter.subplot(1, 2) 
        w2 = self.plotter.add_slider_widget(self._cb_thresh_dexterity, [0.0, 1.0], value=0.0, title="Min Dexterity", 
            pointa=(0.1, 0.73), pointb=(0.9, 0.73), **ss)
        self.slider_widgets.append(w2)

        # Slider 3: X Slice
        self.plotter.subplot(1, 2) 
        w3 = self.plotter.add_slider_widget(self._cb_x, [b[0], b[1]], value=self.current_slices['x'], title="X Slice", 
            pointa=(0.1, 0.56), pointb=(0.9, 0.56), **ss)
        self.slider_widgets.append(w3)

        # Slider 4: Y Slice
        self.plotter.subplot(1, 2) 
        w4 = self.plotter.add_slider_widget(self._cb_y, [b[2], b[3]], value=self.current_slices['y'], title="Y Slice", 
            pointa=(0.1, 0.39), pointb=(0.9, 0.39), **ss)
        self.slider_widgets.append(w4)

        # Slider 5: Z Slice
        self.plotter.subplot(1, 2) 
        w5 = self.plotter.add_slider_widget(self._cb_z, [b[4], b[5]], value=self.current_slices['z'], title="Z Slice", 
            pointa=(0.1, 0.22), pointb=(0.9, 0.22), **ss)
        self.slider_widgets.append(w5)

        # --- 4. DATA INIT & LINKING ---
        self._update_threshold_cloud('Safety', 0.0)
        self._update_threshold_cloud('Dexterity', 0.0)
        self._update_single_plane('x', self.current_slices['x'])
        self._update_single_plane('y', self.current_slices['y'])
        self._update_single_plane('z', self.current_slices['z'])
        
        self.plotter.link_views(views=[0, 1, 3, 4])
        
        logger.info("MCFP Analysis Studio Ready.")
        self.plotter.show()