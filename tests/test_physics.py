"""Physical limits, frame round trips, eclipses, and complete-year exports."""
import csv
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import numpy as np
from numpy.testing import assert_allclose

from sail_dynamics import Config, Simulation
from sail_dynamics.constants import AU, EARTH_RADIUS, LIGHT_SPEED, SOLAR_CONSTANT
from sail_dynamics.environment import environment
from sail_dynamics.frames import (HeliocentricEclipticFrame, GeocentricEquatorialFrame,
                                  SatelliteOrbitalFrame, SailBodyFrame)
from sail_dynamics.sail import Sail, Mesh, Hub
from sail_dynamics.sail.optics import facet_loads
from sail_dynamics.shadows import visible_sun_fraction


class OpticalTests(unittest.TestCase):
    def test_flat_analytic_limits(self):
        # Independent momentum limits: absorption P, mirror 2P, Lambertian 5P/3.
        for specular, diffuse, coefficient in [(0, 0, 1), (1, 0, 2), (0, 1, 5/3),
                                                (.85, .10, 1+.85+2*.10/3)]:
            with self.subTest(specular=specular, diffuse=diffuse):
                config = Config(radial_cells=4, wrinkle_m=0, specular=specular, diffuse=diffuse)
                sail = Sail(config)
                forces, irradiance = sail.facet_loads(SOLAR_CONSTANT)
                area = config.angular_segments/8*np.sin(2*np.pi/config.angular_segments)*(
                    config.diameter_m**2-config.hub_diameter_m**2)
                expected = coefficient*SOLAR_CONSTANT*area/LIGHT_SPEED
                assert_allclose(forces.sum(axis=0), [0, 0, -expected], atol=1e-16)
                assert_allclose(irradiance, SOLAR_CONSTANT)
                assert_allclose(sail.unit_torque, 0, atol=1e-18)

    def test_tilted_mirror_momentum(self):
        # Tilt 45° about Y. Specular outgoing photons point along +X.
        vertices = [[0, 0, 0], [1, 0, -1], [0, 1, 0]]
        mesh = Mesh.from_vertices(vertices, [[0, 1, 2]])
        config = Config(specular=1, diffuse=0)
        force, _ = facet_loads(mesh, SOLAR_CONSTANT, config)
        intercepted = SOLAR_CONSTANT*.5
        expected = intercepted/LIGHT_SPEED*np.array([-1, 0, -1])
        assert_allclose(force[0], expected, atol=1e-18)

    def test_wrinkle_geometry_and_power_conservation(self):
        config = Config(radial_cells=24)
        sail = Sail(config)
        mesh = sail.mesh
        self.assertEqual(len(mesh.triangles), 2*config.radial_cells*config.angular_segments)
        self.assertTrue(np.all(mesh.normals[:, 2] > 0))
        assert_allclose(np.linalg.norm(mesh.normals, axis=1), 1)
        self.assertAlmostEqual(np.max(np.abs(mesh.vertices[:, 2])), config.wrinkle_m)
        area = config.angular_segments/8*np.sin(2*np.pi/config.angular_segments)*(
            config.diameter_m**2-config.hub_diameter_m**2)
        self.assertAlmostEqual(mesh.areas.sum(), area)
        self.assertLess(sail.projected_area, area)
        forces, irradiance = sail.facet_loads(SOLAR_CONSTANT)
        self.assertAlmostEqual(np.sum(irradiance*mesh.areas), SOLAR_CONSTANT*sail.projected_area)
        self.assertLess(np.linalg.norm(forces.sum(axis=0)),
                        np.linalg.norm(Sail(replace(config, wrinkle_m=0)).unit_force)*SOLAR_CONSTANT)
        # Absorption transfers axial momentum for the reduced projected area.
        absorber = Sail(replace(config, specular=0, diffuse=0))
        assert_allclose(absorber.unit_force, [0, 0, -absorber.projected_area/LIGHT_SPEED], atol=1e-18)

    def test_mount_torque_translation(self):
        centered = Sail(Config(radial_cells=8, wrinkle_m=0, attachment_m=(0, 0, 0)))
        offset = np.array([1, -.5, -.8])
        moved = Sail(replace(centered.config, attachment_m=tuple(offset)))
        assert_allclose(moved.unit_torque, centered.unit_torque-np.cross(offset, centered.unit_force),
                        atol=1e-18)

    def test_umbra_has_no_optical_load(self):
        sail = Sail(Config(radial_cells=4))
        force, irradiance = sail.facet_loads(0)
        assert_allclose(force, 0)
        assert_allclose(irradiance, 0)


class FrameTests(unittest.TestCase):
    def test_heliocentric_geocentric_origins_and_inverse(self):
        config = Config()
        earth = HeliocentricEclipticFrame().earth_position(0, config)
        frame = GeocentricEquatorialFrame()
        assert_allclose(frame.positions_from_heliocentric(earth, earth), 0)
        points = np.array([[0, 0, 0], [AU, 2*AU, -AU]], float)
        transformed = frame.positions_from_heliocentric(points, earth)
        assert_allclose(frame.positions_to_heliocentric(transformed, earth), points, atol=1e-4)
        sun = frame.sun_position(0, config)
        self.assertLess(sun[2], 0)  # Northern winter: Sun south of celestial equator.

    def test_orbital_frame_and_circular_period(self):
        config = Config(raan_deg=23, inclination_deg=67)
        frame = SatelliteOrbitalFrame(config)
        positions = frame.positions_geocentric([0, config.period_s/4, config.period_s])
        assert_allclose(np.linalg.norm(positions, axis=-1), EARTH_RADIUS+config.altitude_m)
        assert_allclose(positions[0], positions[-1], atol=1e-8)
        self.assertAlmostEqual(np.dot(positions[0], positions[1]) / np.dot(positions[0], positions[0]), 0)
        assert_allclose(frame.vectors_from_geocentric(positions),
                        frame.positions_orbital([0, config.period_s/4, config.period_s]), atol=1e-8)
        self.assertAlmostEqual(np.linalg.det(frame.rotation_to_geocentric), 1)

    def test_sail_frame_rotations_and_translations(self):
        directions = np.array([[1, 0, 0], [0, 0, 1], [0, 0, -1], [1, 2, 3]], float)
        directions /= np.linalg.norm(directions, axis=1)[:, None]
        frame = SailBodyFrame(directions)
        rotations = frame.rotation_to_geocentric
        assert_allclose(rotations[..., 2], directions)
        assert_allclose(np.linalg.det(rotations), 1)
        assert_allclose(np.einsum('...ji,...jk->...ik', rotations, rotations),
                        np.broadcast_to(np.eye(3), rotations.shape), atol=1e-15)
        vectors = np.array([[1, 2, 3]]*4, float)
        assert_allclose(frame.vectors_from_geocentric(frame.vectors_to_geocentric(vectors)), vectors)
        centers = np.array([[1e6, 2e6, -3e6]]*4)
        assert_allclose(frame.positions_from_geocentric(
            frame.positions_to_geocentric(vectors, centers), centers), vectors, atol=1e-9)

    def test_seasonal_distance(self):
        config = Config()
        positions = HeliocentricEclipticFrame().earth_position(np.arange(365)*86400, config)
        radii = np.linalg.norm(positions, axis=-1)/AU
        self.assertTrue(.982 < radii.min() < .985)
        self.assertTrue(1.015 < radii.max() < 1.018)
        self.assertLess(int(radii.argmin()), 8)
        self.assertTrue(179 < int(radii.argmax()) < 190)


class ShadowAndYearTests(unittest.TestCase):
    def test_day_night_and_penumbra(self):
        radius = EARTH_RADIUS+400_000
        sun = np.array([AU, 0, 0])
        self.assertEqual(float(visible_sun_fraction([radius, 0, 0], sun)), 1)
        self.assertEqual(float(visible_sun_fraction([-radius, 0, 0], sun)), 0)
        beta = np.arcsin(EARTH_RADIUS/radius)
        angles = np.linspace(beta-.012, beta+.012, 1001)
        # Sweep from night toward daylight near Earth's apparent limb.
        positions = radius*np.stack((-np.cos(angles), np.sin(angles), np.zeros_like(angles)), -1)
        fractions = visible_sun_fraction(positions, sun)
        self.assertTrue(np.all(np.isfinite(fractions)))
        self.assertEqual(fractions[0], 0)
        self.assertEqual(fractions[-1], 1)
        self.assertTrue(np.all(np.diff(fractions) >= -1e-8))
        self.assertTrue(np.any((fractions > .05) & (fractions < .95)))
        self.assertLess(np.max(np.diff(fractions)), .01)

    def test_eclipse_duration_matches_cylindrical_limit(self):
        config = Config(inclination_deg=0, radial_cells=4)
        seconds = np.linspace(0, config.period_s, 30001, endpoint=False)
        satellite = SatelliteOrbitalFrame(config).positions_geocentric(seconds)
        sunlight = visible_sun_fraction(satellite, [AU, 0, 0])
        expected_dark_fraction = np.arcsin(EARTH_RADIUS/(EARTH_RADIUS+config.altitude_m))/np.pi
        self.assertAlmostEqual(np.mean(1-sunlight), expected_dark_fraction, delta=.0002)

    def test_year_grid_export_and_inertial_force(self):
        sim = Simulation(Config(radial_cells=4))
        self.assertEqual(len(sim.hours), 8761)
        self.assertEqual(sim.hours[-1], 365*86400)
        self.assertTrue(np.all(np.isfinite(sim.annual_loads.forces_inertial)))
        assert_allclose(np.linalg.norm(sim.annual_loads.forces_body, axis=-1),
                        np.linalg.norm(sim.annual_loads.forces_inertial, axis=-1))
        self.assertTrue(np.any(sim.year.sunlight == 0))
        self.assertTrue(np.any(sim.year.sunlight == 1))
        with tempfile.TemporaryDirectory() as directory:
            path = sim.export_csv(Path(directory)/'year.csv')
            with path.open() as stream:
                records = list(csv.DictReader(stream))
            self.assertEqual(len(records), 8761)
            self.assertTrue(records[-1]['utc'].startswith('2027-01-01'))
            index = int(np.argmax(sim.year.flux))
            self.assertAlmostEqual(float(records[index]['force_norm_N']),
                                   np.linalg.norm(sim.annual_loads.forces_body[index]))
        leap = Simulation(Config(year=2024, radial_cells=4))
        self.assertEqual(len(leap.hours), 8785)

    def test_invalid_physics_inputs_rejected(self):
        for kwargs in [{'mass_kg': 0}, {'altitude_m': -1}, {'specular': .9, 'diffuse': .2},
                       {'wrinkle_m': -1}, {'radial_cells': 1}, {'diameter_m': float('nan')},
                       {'hub_diameter_m': 0}, {'hub_diameter_m': 10},
                       {'hub_diameter_m': 11}, {'angular_segments': 7},
                       {'hub_specular': .8, 'hub_diffuse': .3}]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                Config(**kwargs)


if __name__ == '__main__':
    unittest.main()
