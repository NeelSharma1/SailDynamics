"""Matplotlib explorer; importing the physics model never opens a GUI."""
from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import numpy as np

from .constants import EARTH_RADIUS, LIGHT_SPEED, SOLAR_CONSTANT, AU
from .simulation import Simulation
from .environment import environment
from .frames.orbital import SatelliteOrbitalFrame
from .sail.boundary import SailBoundary


class Explorer:
    def __init__(self, simulation: Simulation):
        import matplotlib.pyplot as plt
        from matplotlib import colors
        from matplotlib.cm import ScalarMappable
        from matplotlib.widgets import Button, Slider
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection

        self.plt, self.sim = plt, simulation
        self.index, self.offset_s, self.playing = 0, 0.0, False
        self.arrows = []
        config, mesh = self.sim.config, self.sim.sail.mesh
        self.fig = plt.figure(figsize=(17, 10), facecolor="#f5f7fb")
        self.fig.canvas.manager.set_window_title("Sail Dynamics · Solar radiation pressure")
        self.fig.suptitle("SUNLIGHT → FACETS → SPACECRAFT FORCE", x=.055, y=.98,
                          ha="left", fontsize=19, fontweight="bold", color="#162d4b")
        self.fig.text(.055, .945,
            f"{config.altitude_m/1000:g} km circular orbit · {config.inclination_deg:g}° inclination · "
            f"{config.period_s/60:.2f} min period · Sun-tracking plane · prescribed wrinkles · UTC",
            fontsize=10, color="#49617a")
        self.orbit_ax = self.fig.add_axes([.025, .56, .28, .33], projection="3d")
        self.sail_ax = self.fig.add_axes([.33, .53, .31, .36], projection="3d")
        self.map_ax = self.fig.add_axes([.715, .66, .17, .23])
        self.stats = self.fig.text(.69, .585, "", va="top", fontsize=9, family="monospace",
                                   color="#162d4b", linespacing=1.55)
        self.annual_ax = self.fig.add_axes([.065, .265, .55, .185])
        self.detail_ax = self.fig.add_axes([.70, .265, .245, .135])
        self.detail_torque_ax = self.detail_ax.twinx()
        self._draw_orbit_background()

        self.sail_collection = Poly3DCollection(mesh.vertices[mesh.triangles],
                                               linewidths=.16, edgecolors="#26354866")
        self.sail_ax.add_collection3d(self.sail_collection)
        hub_mesh = self.sim.hub.mesh
        self.hub_collection = Poly3DCollection(hub_mesh.vertices[hub_mesh.triangles],
                                               facecolors="#46576f", edgecolors="#192b42",
                                               linewidths=.2)
        self.sail_ax.add_collection3d(self.hub_collection)
        self.geometry_label = self.fig.text(.345, .495, "", fontsize=8, color="#49617a")
        self.pressure_norm = colors.Normalize(0, 2*SOLAR_CONSTANT*1.04/LIGHT_SPEED*1e6)
        self.pressure_cmap = plt.get_cmap("plasma")
        self.pressure_bar = self.fig.colorbar(
            ScalarMappable(norm=self.pressure_norm, cmap=self.pressure_cmap),
            cax=self.fig.add_axes([.652, .63, .008, .23]))
        self.pressure_bar.set_label("SRP force / facet area [µPa]", fontsize=8)
        width = config.diameter_m
        self.sail_ax.set(xlim=(-width*.6, width*.6), ylim=(-width*.6, width*.6),
                         zlim=(-max(width*.28, abs(config.attachment_m[2])*1.6), width*.12),
                         xlabel="X [m]", ylabel="Y [m]", zlabel="Z [m]")
        self.sail_ax.set_box_aspect((1, 1, .5))
        self.sail_ax.view_init(elev=25, azim=-62)
        self.sail_ax.set_title("Radial sail · membrane SRP load", fontsize=12, pad=8)
        self.sail_ax.text2D(.02, .00, "Height exaggerated · blue: facets · green: net force",
                            transform=self.sail_ax.transAxes, fontsize=7, color="#49617a")
        attach = np.asarray(config.attachment_m)
        self.sail_ax.plot([attach[0], 0], [attach[1], 0], [attach[2], 0], color="#263548", lw=2)
        self.sail_ax.scatter(*attach, marker="s", s=75, color="#192b42", depthshade=False)
        self.map_collection = self.map_ax.tripcolor(
            mesh.vertices[:, 0], mesh.vertices[:, 1], mesh.triangles,
            facecolors=np.zeros(len(mesh.triangles)), cmap="inferno", vmin=0, vmax=1420,
            edgecolors="#ffffff18", linewidth=.1)
        self.hub_map_collection = self.map_ax.tripcolor(
            hub_mesh.vertices[:, 0], hub_mesh.vertices[:, 1], hub_mesh.triangles,
            facecolors=np.zeros(len(hub_mesh.triangles)), cmap="inferno", vmin=0, vmax=1420,
            edgecolors="#192b4288", linewidth=.2)
        self.map_ax.text(0, 0, "hub", ha="center", va="center", fontsize=7,
                         color="#192b42", bbox={"facecolor": "white", "alpha": .7,
                                                "edgecolor": "none", "pad": .5})
        self.map_ax.set(aspect="equal", xlabel="Sail X [m]", ylabel="Sail Y [m]")
        self.map_ax.set_title("Sunlight on each facet", fontsize=12)
        self.fig.colorbar(self.map_collection, ax=self.map_ax, fraction=.05, pad=.07).set_label(
            "Incident irradiance [W/m²]", fontsize=8)

        self.hour_slider = Slider(self.fig.add_axes([.15, .165, .66, .022]),
                                  "Year hour", 0, len(self.sim.hours)-1,
                                  valinit=0, valstep=1, valfmt="%0.0f")
        self.orbit_slider = Slider(self.fig.add_axes([.15, .115, .68, .022]),
                                   "Orbit offset [min]", -config.period_s/120,
                                   config.period_s/120, valinit=0, valfmt="%0.3f")
        self.wrinkle_slider = Slider(self.fig.add_axes([.15, .065, .24, .022]),
                                     "Wrinkle peak [m]", 0, max(width*.06, config.wrinkle_m),
                                     valinit=config.wrinkle_m, valfmt="%0.3f")
        self.hub_slider = Slider(self.fig.add_axes([.51, .065, .12, .022]),
                                 "Hub Ø [m]", min(width*.01, config.hub_diameter_m),
                                 max(width*.95, config.hub_diameter_m),
                                 valinit=config.hub_diameter_m, valfmt="%0.2f")
        self.play_button = Button(self.fig.add_axes([.86, .155, .075, .038]), "Play year")
        self.export_button = Button(self.fig.add_axes([.69, .055, .10, .038]), "Export CSV")
        self.reset_button = Button(self.fig.add_axes([.82, .055, .115, .038]), "Reset offset")
        self.shape_button = Button(self.fig.add_axes([.89, .105, .085, .038]),
                                    "Shape: circle" if config.sail_shape == 'circular' else "Shape: octagon")
        self.shape_button.label.set_fontsize(9)
        self.footer = self.fig.text(.055, .018,
            "Scroll or ←/→: 1 hour · Shift + ←/→: 1 day · Space: play/pause · "
            "click either time plot to seek · drag 3D views to rotate", fontsize=9, color="#49617a")
        self.hour_slider.on_changed(self._hour_changed)
        self.orbit_slider.on_changed(self._offset_changed)
        self.wrinkle_slider.on_changed(self._wrinkle_changed)
        self.hub_slider.on_changed(self._hub_changed)
        self.play_button.on_clicked(self._toggle_play)
        self.export_button.on_clicked(self._export)
        self.reset_button.on_clicked(lambda _: self.orbit_slider.set_val(0))
        self.shape_button.on_clicked(self._shape_changed)
        self._set_hub_slider_limit()
        self.fig.canvas.mpl_connect("key_press_event", self._key)
        self.fig.canvas.mpl_connect("scroll_event", self._scroll)
        self.fig.canvas.mpl_connect("button_press_event", self._seek)
        self.fig.canvas.mpl_connect("close_event", lambda _: self.timer.stop())
        self.timer = self.fig.canvas.new_timer(interval=160)
        self.timer.add_callback(self._tick)
        self._draw_annual()
        self._draw_detail()
        self.update()

    def _draw_orbit_background(self):
        ax = self.orbit_ax
        u, v = np.meshgrid(np.linspace(0, 2*np.pi, 65), np.linspace(0, np.pi, 33))
        self.earth_points = np.stack((np.cos(u)*np.sin(v), np.sin(u)*np.sin(v), np.cos(v)), -1)
        self.earth_surface = None
        orbit = SatelliteOrbitalFrame(self.sim.config).positions_geocentric(
            np.linspace(0, self.sim.config.period_s, 400))/EARTH_RADIUS
        ax.plot(*orbit.T, color="#2874b2", lw=1.3, alpha=.65)
        self.satellite_dot, = ax.plot([], [], [], "o", color="#e75535", markersize=7)
        ax.set(xlim=(-1.5, 1.5), ylim=(-1.5, 1.5), zlim=(-1.5, 1.5),
               xlabel="ECI X [Rᴇ]", ylabel="ECI Y [Rᴇ]", zlabel="ECI Z [Rᴇ]")
        ax.set_box_aspect((1, 1, 1))
        ax.view_init(elev=24, azim=35)
        ax.set_title("Earth orbit · sunlight & eclipse", fontsize=12)
        self.sun_arrow = None
        self.orbit_label = self.fig.text(.055, .49, "", fontsize=8)

    def _draw_annual(self):
        ax = self.annual_ax
        ax.clear()
        days = self.sim.hours/86400
        magnitude = np.linalg.norm(self.sim.annual_loads.forces_body, axis=1)*1e6
        ax.plot(days, magnitude, ".", color="#2874b2", ms=1.6, alpha=.5, rasterized=True,
                label="Hourly instantaneous samples")
        upper = self.sim.year.unshadowed_flux*np.linalg.norm(self.sim.unit_force)*1e6
        ax.plot(days, upper, color="#e88425", lw=1.3, label="Full-sun force for this mesh")
        self.annual_cursor = ax.axvline(0, color="#e75535", lw=1)
        self.annual_dot, = ax.plot([], [], "o", color="#e75535", ms=5)
        ax.set(xlim=(0, self.sim.config.duration_s/86400), ylim=(0, max(upper.max()*1.12, 1)),
               xlabel=f"Days since {self.sim.config.year}-01-01 UTC", ylabel="Assembly SRP force [µN]")
        ax.set_title("One year · 1-hour samples (eclipses can fall between samples)",
                     fontsize=10, loc="left")
        ax.legend(loc="lower right", fontsize=7)
        ax.grid(alpha=.2)

    def _draw_detail(self):
        ax = self.detail_ax
        ax.clear()
        self.detail_torque_ax.clear()
        period = self.sim.config.period_s
        offsets = np.linspace(-period/2, period/2, int(np.ceil(period/60))+1)
        times = self.sim.hours[self.index] + offsets
        valid = (times >= 0) & (times <= self.sim.config.duration_s)
        offsets, times = offsets[valid], times[valid]
        # Include contact refinement in this local orbit rather than relying on
        # minute samples to happen to hit a short penumbra.
        # The backend's full-year schedule is not recomputed for every UI tick.
        # A local fine grid near visibility changes gives immediate interaction.
        coarse_env = environment(times, self.sim.config)
        transitions = np.flatnonzero(np.abs(np.diff(coarse_env.sunlight)) > .001)
        additions = [np.linspace(times[i], times[i+1], 241) for i in transitions]
        if additions:
            times = np.unique(np.concatenate([times, *additions]))
            offsets = times-self.sim.hours[self.index]
        env = environment(times, self.sim.config)
        local_loads = self.sim.loads(env)
        forces = np.linalg.norm(local_loads.forces_body, axis=-1)*1e6
        ax.fill_between(offsets/60, forces, color="#f4b85c", alpha=.55)
        ax.plot(offsets/60, forces, color="#ba6315", lw=1)
        torques = np.linalg.norm(local_loads.torques_body, axis=-1)*1e6
        self.detail_torque_ax.plot(offsets/60, torques, color="#0a8f82", lw=1)
        self.detail_torque_ax.set(ylim=(0, max(torques.max()*1.15, 1e-3)), ylabel="Torque [µN·m]")
        self.detail_torque_ax.yaxis.set_label_position('right')
        self.detail_torque_ax.yaxis.tick_right()
        self.detail_torque_ax.tick_params(axis='y', labelsize=7, colors="#0a8f82")
        self.detail_torque_ax.yaxis.label.set_color("#0a8f82")
        self.detail_torque_ax.yaxis.label.set_size(8)
        self.detail_torque_dot, = self.detail_torque_ax.plot([], [], 'o', color="#0a8f82", ms=4)
        self.detail_cursor = ax.axvline(self.offset_s/60, color="#e75535", lw=1)
        self.detail_dot, = ax.plot([], [], "o", color="#e75535", ms=4)
        ax.set(xlim=(-period/120, period/120), ylim=(0, max(forces.max()*1.15, 1)),
               xlabel="Minutes from selected hour", ylabel="Force [µN]")
        ax.set_title("Selected orbit · minutes + eclipse refinement", fontsize=9, loc="left")
        ax.tick_params(labelsize=8)
        ax.grid(alpha=.2)

    def update(self):
        config, mesh = self.sim.config, self.sim.sail.mesh
        self.sail_ax.set_title(f"{config.sail_shape.capitalize()} sail · membrane SRP", fontsize=12, pad=8)
        seconds = float(self.sim.hours[self.index] + self.offset_s)
        point = self.sim.spatial.snapshot(seconds)
        env, loads = point.environment, point.loads
        facets, irradiance = self.sim.sail.facet_loads(point.beam_flux[:point.sail_count])
        force = loads.forces_body
        magnitude = float(np.linalg.norm(force))
        pressure = np.linalg.norm(facets, axis=1)/mesh.areas*1e6
        self.sail_collection.set_verts(mesh.vertices[mesh.triangles])
        self.sail_collection.set_facecolor(self.pressure_cmap(self.pressure_norm(pressure)))
        hub_mesh = self.sim.hub.mesh
        self.hub_collection.set_verts(hub_mesh.vertices[hub_mesh.triangles])
        self.map_collection.set_verts(mesh.vertices[mesh.triangles, :2])
        self.map_collection.set_array(irradiance)
        self.hub_map_collection.set_verts(hub_mesh.vertices[hub_mesh.triangles, :2])
        _, hub_irradiance = self.sim.hub.facet_loads(point.beam_flux[point.sail_count:])
        self.hub_map_collection.set_array(hub_irradiance)
        self.geometry_label.set_text(
            f"{config.sail_shape.capitalize()} · material {self.sim.material_area:.2f} m² · "
            f"projected {self.sim.projected_area:.2f} m² · "
            f"hub Ø {config.hub_diameter_m:g} m\n"
            f"Membrane |F| {np.linalg.norm(loads.sail_forces_body)*1e6:.2f} µN · "
            f"hub |F| {np.linalg.norm(loads.hub_forces_body)*1e6:.2f} µN")
        for arrow in self.arrows:
            arrow.remove()
        self.arrows = []
        # Representative facet vectors share a physical force/area scale.
        stride = max(1, len(facets)//45)
        centers = mesh.centers[::stride]
        force_per_area = facets[::stride]/mesh.areas[::stride, None]
        vectors = force_per_area/(2*SOLAR_CONSTANT/LIGHT_SPEED)*config.diameter_m*.10
        self.arrows.append(self.sail_ax.quiver(*centers.T, *vectors.T,
                                               color="#175d7d", linewidth=.7,
                                               arrow_length_ratio=.25))
        attach = np.asarray(config.attachment_m)
        max_force = 2*SOLAR_CONSTANT/LIGHT_SPEED*self.sim.projected_area
        vector = force/max_force*config.diameter_m*.16
        self.arrows.append(self.sail_ax.quiver(*attach, *vector, color="#0aab9b",
                                               linewidth=2.8, arrow_length_ratio=.25))
        position = env.satellite_m/EARTH_RADIUS
        self.satellite_dot.set_data_3d([position[0]], [position[1]], [position[2]])
        if self.earth_surface is not None:
            self.earth_surface.remove()
        shade = np.maximum(np.einsum("...i,i->...", self.earth_points, env.sun_unit), 0)
        colors = self.plt.get_cmap("Blues")(.62-.35*shade)
        colors[..., :3] *= (.30+.70*shade[..., None])
        self.earth_surface = self.orbit_ax.plot_surface(*np.moveaxis(self.earth_points, -1, 0),
                                                        facecolors=colors, linewidth=0,
                                                        shade=False, antialiased=False,
                                                        rcount=33, ccount=65)
        if self.sun_arrow is not None:
            self.sun_arrow.remove()
        self.sun_arrow = self.orbit_ax.quiver(*(env.sun_unit*1.48), *(-env.sun_unit*.55),
                                             color="#f0aa25", linewidth=2, arrow_length_ratio=.3)
        fraction = float(env.sunlight)
        state = "UMBRA" if fraction < 1e-8 else "SUNLIT" if fraction > 1-1e-8 else "PENUMBRA"
        self.orbit_label.set_text(f"{state} · visible Sun {fraction:.1%}\nGold arrow: incoming light")
        self.orbit_label.set_color("#ba6315" if fraction > 0 else "#49617a")
        stamp = config.start + timedelta(seconds=seconds)
        torque = np.linalg.norm(loads.torques_body)
        flat = float(env.flux)/LIGHT_SPEED*(
            self.sim.sail.material_area*(1+config.specular+2*config.diffuse/3)
            + self.sim.hub.projected_area*(1+config.hub_specular+2*config.hub_diffuse/3))
        ratio_text = f"{magnitude/flat:.2%}" if flat > 0 else "— (eclipse)"
        self.stats.set_text(
            f"{stamp:%Y-%m-%d %H:%M:%S} UTC   {state}\n"
            f"Sun {env.distance_m/AU:.4f} AU | visible {fraction:6.1%}\n"
            f"Intercepted power {float(loads.power_w)/1000:7.2f} kW\n"
            f"Force {magnitude*1e6:7.2f} µN | a {magnitude/config.mass_kg*1e6:6.3f} µm/s²\n"
            f"F XYZ [{force[0]*1e6:+.2f}, {force[1]*1e6:+.2f}, {force[2]*1e6:+.2f}] µN\n"
            f"Mount torque {torque*1e6:.2f} µN·m | flat ratio {ratio_text}")
        day = seconds/86400
        self.annual_cursor.set_xdata([day, day])
        self.annual_dot.set_data([day], [magnitude*1e6])
        self.detail_cursor.set_xdata([self.offset_s/60, self.offset_s/60])
        self.detail_dot.set_data([self.offset_s/60], [magnitude*1e6])
        self.detail_torque_dot.set_data([self.offset_s/60], [torque*1e6])
        self.fig.canvas.draw_idle()

    def _hour_changed(self, value):
        self.index = int(value)
        self.offset_s = 0
        self.orbit_slider.eventson = False
        self.orbit_slider.set_val(0)
        self.orbit_slider.eventson = True
        self._draw_detail()
        self.update()

    def _offset_changed(self, value):
        center = self.sim.hours[self.index]
        seconds = np.clip(center+value*60, 0, self.sim.config.duration_s)
        self.offset_s = float(seconds-center)
        if abs(self.offset_s/60-value) > 1e-8:
            self.orbit_slider.eventson = False
            self.orbit_slider.set_val(self.offset_s/60)
            self.orbit_slider.eventson = True
        self.update()

    def _wrinkle_changed(self, value):
        try:
            self.sim.change_wrinkles(value)
        except ValueError as error:
            self._restore_slider(self.wrinkle_slider, self.sim.config.wrinkle_m, error)
            return
        self._draw_annual()
        self._draw_detail()
        self.update()

    def _hub_changed(self, value):
        try:
            self.sim.change_hub_diameter(value)
        except ValueError as error:
            self._restore_slider(self.hub_slider, self.sim.config.hub_diameter_m, error)
            return
        self._draw_annual()
        self._draw_detail()
        self.update()

    def _restore_slider(self, slider, value, error):
        slider.eventson = False
        slider.set_val(value)
        slider.eventson = True
        self.footer.set_text(str(error))
        self.fig.canvas.draw_idle()

    def _set_hub_slider_limit(self):
        self.hub_slider.valmax = max(self.sim.config.hub_diameter_m,
                                     1.9*SailBoundary(self.sim.config).min_radius_m)
        self.hub_slider.ax.set_xlim(self.hub_slider.valmin, self.hub_slider.valmax)

    def _shape_changed(self, _=None):
        shape = 'octagonal' if self.sim.config.sail_shape == 'circular' else 'circular'
        try:
            self.sim.change_shape(shape)
        except ValueError as error:
            self.footer.set_text(str(error))
            self.fig.canvas.draw_idle()
            return
        self.shape_button.label.set_text('Shape: octagon' if shape == 'octagonal' else 'Shape: circle')
        self._set_hub_slider_limit()
        self.footer.set_text(f'{shape.capitalize()} sail · preserved material area {self.sim.material_area:.6f} m² '
                             f'and hub diameter {self.sim.config.hub_diameter_m:g} m')
        self._draw_annual()
        self._draw_detail()
        self.update()

    def _toggle_play(self, _=None):
        self.playing = not self.playing
        self.play_button.label.set_text("Pause" if self.playing else "Play year")
        self.timer.start() if self.playing else self.timer.stop()
        self.fig.canvas.draw_idle()

    def _tick(self):
        if self.index >= len(self.sim.hours)-1:
            self._toggle_play()
        else:
            self.hour_slider.set_val(self.index+1)

    def _key(self, event):
        if event.key in (" ", "space"):
            self._toggle_play()
        elif event.key in ("left", "right", "shift+left", "shift+right"):
            step = (24 if "shift" in event.key else 1)*(-1 if "left" in event.key else 1)
            self.hour_slider.set_val(np.clip(self.index+step, 0, len(self.sim.hours)-1))

    def _scroll(self, event):
        if event.inaxes in (self.orbit_ax, self.sail_ax):
            return  # Preserve Matplotlib's 3-D zoom interaction.
        self.hour_slider.set_val(np.clip(self.index+int(np.sign(event.step)),
                                         0, len(self.sim.hours)-1))

    def _seek(self, event):
        if event.button != 1 or event.xdata is None:
            return
        if event.inaxes is self.annual_ax:
            self.hour_slider.set_val(np.clip(round(event.xdata*24), 0, len(self.sim.hours)-1))
        elif event.inaxes in (self.detail_ax, self.detail_torque_ax):
            self.orbit_slider.set_val(np.clip(event.xdata, self.orbit_slider.valmin,
                                              self.orbit_slider.valmax))

    def _export(self, _=None):
        path = self.sim.export_csv(Path("output")/f"sail_{self.sim.config.year}_hourly.csv")
        self.footer.set_text(f"Saved {len(self.sim.hours):,} hourly samples to {path.resolve()}")
        self.fig.canvas.draw_idle()

    def show(self):
        self.plt.show()
