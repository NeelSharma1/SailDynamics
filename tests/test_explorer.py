"""Exercise real Matplotlib widgets/callbacks with a noninteractive renderer."""
from types import SimpleNamespace
import unittest

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from sail_dynamics import Config, Simulation
from sail_dynamics.view import Explorer


class ExplorerTests(unittest.TestCase):
    def test_time_wrinkle_and_play_controls(self):
        view = Explorer(Simulation(Config(radial_cells=6, angular_segments=32)))
        try:
            view.hour_slider.set_val(100)
            self.assertEqual(view.index, 100)
            view.orbit_slider.set_val(10)
            self.assertEqual(view.offset_s, 600)
            view._key(SimpleNamespace(key='shift+right'))
            self.assertEqual(view.index, 124)
            self.assertEqual(view.offset_s, 0)
            original = np.linalg.norm(view.sim.sail.unit_force)
            material_area = view.sim.material_area
            wrinkled_area = view.sim.projected_area
            view.wrinkle_slider.set_val(0)
            self.assertGreater(np.linalg.norm(view.sim.sail.unit_force), original)
            self.assertAlmostEqual(view.sim.material_area, material_area)
            self.assertGreater(view.sim.projected_area, wrinkled_area)
            self.assertIn('material', view.geometry_label.get_text())
            self.assertIn('projected', view.geometry_label.get_text())
            outer_area = view.sim.projected_area
            flat_force = np.linalg.norm(view.sim.unit_force)
            view.hub_slider.set_val(3)
            self.assertEqual(view.sim.config.hub_diameter_m, 3)
            self.assertAlmostEqual(view.sim.projected_area, outer_area)
            self.assertLess(np.linalg.norm(view.sim.unit_force), flat_force)
            # The sunlight-map hole and filled hub must move with the mesh.
            inner_ring = view.sim.sail.mesh.vertices[:view.sim.config.angular_segments, :2]
            np.testing.assert_allclose(np.linalg.norm(inner_ring, axis=-1), 1.5)
            self.assertIn('hub Ø 3 m', view.geometry_label.get_text())
            view._seek(SimpleNamespace(button=1, xdata=25, inaxes=view.annual_ax))
            self.assertEqual(view.index, 600)
            view._scroll(SimpleNamespace(inaxes=view.annual_ax, step=-1))
            self.assertEqual(view.index, 599)
            view._seek(SimpleNamespace(button=1, xdata=2, inaxes=view.detail_torque_ax))
            self.assertEqual(view.offset_s, 120)
            self.assertEqual(len(view.detail_torque_dot.get_ydata()), 1)
            self.assertEqual(view.detail_torque_ax.yaxis.get_label_position(), 'right')
            area = view.sim.projected_area
            hub_vertices = view.sim.hub.mesh.vertices.copy()
            view._shape_changed()
            self.assertEqual(view.sim.config.sail_shape, 'octagonal')
            self.assertAlmostEqual(view.sim.projected_area, area)
            np.testing.assert_array_equal(view.sim.hub.mesh.vertices, hub_vertices)
            self.assertIn('Octagonal', view.geometry_label.get_text())
            view._shape_changed()
            self.assertEqual(view.sim.config.sail_shape, 'circular')
            # An infeasible wrinkle slider change rolls back instead of adding
            # material or leaving the controls inconsistent with the mesh.
            view.hub_slider.set_val(9.5)
            view.wrinkle_slider.set_val(.6)
            self.assertEqual(view.sim.config.wrinkle_m, 0)
            self.assertEqual(view.wrinkle_slider.val, 0)
            self.assertIn('Cannot conserve membrane area', view.footer.get_text())
            view._toggle_play()
            self.assertTrue(view.playing)
            view._tick()
            self.assertEqual(view.index, 600)
            view._toggle_play()
            view.hour_slider.set_val(0)
            view.orbit_slider.set_val(-10)
            self.assertEqual(view.offset_s, 0)
            view.hour_slider.set_val(len(view.sim.hours)-1)
            view.orbit_slider.set_val(10)
            self.assertEqual(view.offset_s, 0)
            view.fig.canvas.draw()
        finally:
            view.timer.stop()
            plt.close(view.fig)
