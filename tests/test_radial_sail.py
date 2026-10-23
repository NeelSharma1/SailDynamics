"""Annular topology, hub partition, and assembly loads across hub sizes."""
from collections import Counter
from dataclasses import replace
import unittest

import numpy as np
from numpy.testing import assert_allclose

from sail_dynamics import Config, Simulation
from sail_dynamics.constants import LIGHT_SPEED, SOLAR_CONSTANT
from sail_dynamics.environment import environment
from sail_dynamics.sail import Sail, Hub, SailBoundary
from sail_dynamics.sail.wrinkles import WrinkleField


class RadialSailTests(unittest.TestCase):
    def test_annular_topology_and_clamped_rings(self):
        config = Config(radial_cells=8, angular_segments=48)
        sail = Sail(config)
        mesh = sail.mesh
        sectors = config.angular_segments
        self.assertEqual(len(mesh.vertices), (config.radial_cells+1)*sectors)
        radius = np.linalg.norm(mesh.vertices[:, :2], axis=-1)
        assert_allclose(radius[:sectors], config.hub_diameter_m/2)
        assert_allclose(radius[-sectors:], sail.boundary.max_radius_m)
        self.assertLess(sail.boundary.max_radius_m, config.diameter_m/2)
        assert_allclose(mesh.vertices[:sectors, 2], 0)
        assert_allclose(mesh.vertices[-sectors:, 2], 0)
        self.assertTrue(np.all(mesh.normals[:, 2] > 0))
        edges = Counter(tuple(sorted((int(a), int(b)))) for triangle in mesh.triangles
                        for a, b in zip(triangle, np.roll(triangle, -1)))
        self.assertEqual(sum(count == 1 for count in edges.values()), 2*sectors)
        self.assertEqual(max(edges.values()), 2)
        # No duplicated seam vertices and no seam boundary running radially.
        for (a, b), count in edges.items():
            if count == 1:
                self.assertAlmostEqual(radius[a], radius[b])

    def test_hub_matches_membrane_and_circle_area_converges(self):
        for diameter in [.1, 1, 8, 9.9]:
            config = Config(hub_diameter_m=diameter, radial_cells=4, wrinkle_m=0)
            sail, hub = Sail(config), Hub(config)
            assert_allclose(sail.mesh.vertices[:config.angular_segments], hub.mesh.vertices[1:])
            expected_hub_area = config.angular_segments/8*np.sin(
                2*np.pi/config.angular_segments)*diameter**2
            self.assertAlmostEqual(hub.projected_area, expected_hub_area)
            outer_area = config.angular_segments/8*np.sin(
                2*np.pi/config.angular_segments)*config.diameter_m**2
            self.assertAlmostEqual(sail.projected_area+hub.projected_area, outer_area)
            self.assertLess(abs(outer_area/(np.pi*(config.diameter_m/2)**2)-1), .001)

    def test_hub_optics_and_total_assembly_force(self):
        for specular, diffuse, factor in [(0, 0, 1), (1, 0, 2), (0, 1, 5/3)]:
            config = Config(radial_cells=4, wrinkle_m=0, hub_specular=specular, hub_diffuse=diffuse)
            sim = Simulation(config)
            assert_allclose(sim.hub.unit_force, [0, 0, -factor*sim.hub.projected_area/LIGHT_SPEED], atol=1e-18)
            assert_allclose(sim.unit_force, sim.sail.unit_force+sim.hub.unit_force)
            env = environment(100_000, config)
            loads = sim.loads(env)
            assert_allclose(loads.forces_body, env.flux*sim.unit_force)
            self.assertAlmostEqual(float(loads.power_w), env.flux*sim.projected_area)

    def test_hub_resize_preserves_outer_power_but_changes_load(self):
        sim = Simulation(Config(radial_cells=4, wrinkle_m=0))
        original_area = sim.projected_area
        original_force = np.linalg.norm(sim.unit_force)
        sim.change_hub_diameter(5)
        self.assertAlmostEqual(sim.projected_area, original_area)
        self.assertLess(np.linalg.norm(sim.unit_force), original_force)
        with self.assertRaises(ValueError):
            sim.change_hub_diameter(11)
        self.assertEqual(sim.config.hub_diameter_m, 5)
        # Uniform absorbing optics make the force independent of hub partition.
        all_absorbing = Simulation(replace(sim.config, specular=0, diffuse=0))
        expected = all_absorbing.projected_area/LIGHT_SPEED
        assert_allclose(all_absorbing.unit_force, [0, 0, -expected], atol=1e-18)

    def test_placeholder_wrinkles_are_periodic(self):
        r, theta = np.meshgrid(np.linspace(0, 1, 15), np.linspace(0, 2*np.pi, 65), indexing='ij')
        heights = WrinkleField(.2, seed=7).heights(r, theta)
        assert_allclose(heights[:, 0], heights[:, -1], atol=1e-14)
        self.assertAlmostEqual(np.max(np.abs(heights)), .2)

    def test_wrinkles_conserve_material_draw_in_rim_and_keep_hub(self):
        for shape in ('circular', 'octagonal'):
            for model in ('modal', 'stochastic'):
                for seed in (2, 7, 2026):
                    config = Config(sail_shape=shape, wrinkle_model=model, seed=seed,
                                    radial_cells=6, angular_segments=32, hub_diameter_m=1.5)
                    flat = Sail(replace(config, wrinkle_m=0))
                    previous_area = flat.projected_area
                    for height in (.01, .18, .6):
                        with self.subTest(shape=shape, model=model, seed=seed, height=height):
                            case = replace(config, wrinkle_m=height)
                            sail, hub = Sail(case), Hub(case)
                            # Area is measured independently from the actual 3-D mesh.
                            points = sail.mesh.vertices[sail.mesh.triangles]
                            area = .5*np.linalg.norm(np.cross(points[:, 1]-points[:, 0],
                                                             points[:, 2]-points[:, 0]), axis=-1).sum()
                            self.assertAlmostEqual(area, flat.projected_area, delta=flat.projected_area*1e-11)
                            self.assertLess(sail.projected_area, previous_area)
                            self.assertLess(sail.boundary.max_radius_m, SailBoundary(case).max_radius_m)
                            self.assertGreater(sail.boundary.min_radius_m, case.hub_diameter_m/2)
                            self.assertAlmostEqual(np.max(np.abs(sail.mesh.vertices[:, 2])), height)
                            assert_allclose(sail.mesh.vertices[:case.angular_segments], hub.mesh.vertices[1:])
                            self.assertTrue(np.all(sail.mesh.normals[:, 2] > 0))
                            previous_area = sail.projected_area

    def test_unfittable_wrinkles_rejected_without_changing_simulation(self):
        sim = Simulation(Config(hub_diameter_m=9.9, radial_cells=4, wrinkle_m=0))
        original_mesh = sim.sail.mesh
        with self.assertRaisesRegex(ValueError, 'Cannot conserve membrane area'):
            sim.change_wrinkles(.18)
        self.assertEqual(sim.config.wrinkle_m, 0)
        self.assertIs(sim.sail.mesh, original_mesh)
