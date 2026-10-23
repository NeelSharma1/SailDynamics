"""Equal material areas, an unchanged hub, and an eight-sided outer boundary."""
import csv
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from numpy.testing import assert_allclose

from sail_dynamics import Config, Simulation
from sail_dynamics.constants import SOLAR_CONSTANT
from sail_dynamics.environment import environment
from sail_dynamics.illumination import SpatialLoadEvaluator
from sail_dynamics.monte_carlo import StudySettings, run_study, magnitudes, METRICS
from sail_dynamics.sail import Sail, Hub, SailBoundary
from sail_dynamics.sampling import eclipse_refinement_times


class ShapeTests(unittest.TestCase):
    def test_equal_material_area_and_identical_hub(self):
        for segments in (8, 16, 48, 96, 192):
            for height in (0, .18, .6):
                config = Config(angular_segments=segments, wrinkle_m=height)
                circle = Sail(config)
                oct_config = replace(config, sail_shape='octagonal')
                octagon = Sail(oct_config)
                self.assertAlmostEqual(circle.material_area, octagon.material_area, places=10)
                if height == 0:
                    self.assertAlmostEqual(circle.projected_area, octagon.projected_area, places=11)
                self.assertEqual(circle.mesh.vertices.shape, octagon.mesh.vertices.shape)
                # Inner ring and hub never change while choosing the outer shape.
                assert_allclose(octagon.mesh.vertices[:segments], circle.mesh.vertices[:segments], atol=1e-15)
                assert_allclose(Hub(config).mesh.vertices, Hub(oct_config).mesh.vertices, atol=0)
                self.assertTrue(np.all(octagon.mesh.normals[:, 2] > 0))

    def test_boundary_is_exact_regular_octagon(self):
        config = Config(sail_shape='octagonal')
        sail = Sail(config)
        boundary = sail.boundary
        n = config.angular_segments
        points = sail.mesh.vertices[-n:, :2]
        corners = points[::n//8]
        sides = np.linalg.norm(np.roll(corners, -1, axis=0)-corners, axis=-1)
        assert_allclose(sides, sides[0], atol=1e-14)
        # Every angular sample lies on one of the eight actual straight sides.
        for side in range(8):
            normal = np.array([np.cos((side+.5)*np.pi/4), np.sin((side+.5)*np.pi/4)])
            indices = np.arange(side*n//8, (side+1)*n//8+1) % n
            assert_allclose(points[indices]@normal, boundary.min_radius_m, atol=1e-14)
        # Independent shoelace integral verifies area, including the hub opening.
        area = .5*np.abs(np.sum(points[:, 0]*np.roll(points[:, 1], -1)
                               -points[:, 1]*np.roll(points[:, 0], -1)))
        self.assertAlmostEqual(area, sail.projected_area+Hub(config).projected_area, places=11)
        self.assertAlmostEqual(area, boundary.projected_area_m2, places=11)
        self.assertLess(boundary.max_radius_m, SailBoundary(config).max_radius_m)

    def test_flat_full_sun_loads_match_at_equal_area(self):
        config = Config(wrinkle_m=0)
        circular, octagonal = Simulation(config), Simulation(replace(config, sail_shape='octagonal'))
        assert_allclose(circular.unit_force, octagonal.unit_force, atol=1e-18)
        assert_allclose(octagonal.unit_torque, 0, atol=1e-18)
        self.assertAlmostEqual(circular.projected_area, octagonal.projected_area)
        self.assertAlmostEqual(np.linalg.norm(circular.unit_force)*SOLAR_CONSTANT,
                               np.linalg.norm(octagonal.unit_force)*SOLAR_CONSTANT)
        octagonal.change_shape('circular')
        self.assertEqual(octagonal.config.sail_shape, 'circular')
        self.assertEqual(octagonal.config.hub_diameter_m, config.hub_diameter_m)

    def test_invalid_shape_or_hub_fit_is_rejected(self):
        for kwargs in ({'sail_shape': 'triangle'}, {'sail_shape': 'octagonal', 'angular_segments': 15},
                       {'sail_shape': 'octagonal', 'hub_diameter_m': 9.9}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                Config(**kwargs)
        sim = Simulation(Config(hub_diameter_m=9.9, radial_cells=4, wrinkle_m=0))
        with self.assertRaises(ValueError):
            sim.change_shape('octagonal')
        self.assertEqual(sim.config.sail_shape, 'circular')

    def test_octagonal_shadow_search_matches_unpruned_and_exports_shape(self):
        config = Config(sail_shape='octagonal', radial_cells=4, angular_segments=16)
        settings = StudySettings(trials=2, duration_days=.07, refine_step_s=.5)
        result = run_study(config, settings)
        extent = np.hypot(SailBoundary(config).max_radius_m, settings.max_wrinkle_m)
        refined, _ = eclipse_refinement_times(config, settings.duration_days*86400, .5, extent)
        times = np.unique(np.r_[result.minutes, refined])
        for row in result.trials:
            case = replace(config, wrinkle_model='stochastic', seed=row['pattern_seed'], wrinkle_m=row['wrinkle_m'])
            evaluator = SpatialLoadEvaluator(case, Sail(case), Hub(case))
            brute = magnitudes(evaluator.loads(environment(times, case))).max(axis=0)
            assert_allclose([row['max_'+metric] for metric in METRICS], brute, atol=1e-12, rtol=0)
        with tempfile.TemporaryDirectory() as directory:
            result.export(directory)
            summary = json.loads((Path(directory)/'summary.json').read_text())
            self.assertEqual(summary['config']['sail_shape'], 'octagonal')
            self.assertEqual(summary['geometry']['shape'], 'octagonal')
            self.assertGreater(summary['geometry']['outer_corner_to_corner_m'], config.diameter_m)
            geometry = summary['peaks']['assembly_torque_Nm']['geometry']
            self.assertLessEqual(geometry['outer_scale'], 1+1e-12)
            self.assertIn('membrane_material_area_m2', geometry)
            npz = np.load(Path(directory)/'assembly_torque_Nm_peak.npz')
            self.assertEqual(str(npz['sail_shape']), 'octagonal')
            self.assertAlmostEqual(float(npz['total_material_area_m2']), SailBoundary(config).reference_area_m2)
            path = Simulation(config).export_csv(Path(directory)/'hourly.csv')
            with path.open() as stream:
                row = next(csv.DictReader(stream))
            self.assertEqual(row['sail_shape'], 'octagonal')
            self.assertAlmostEqual(float(row['reference_circle_diameter_m']), config.diameter_m)
