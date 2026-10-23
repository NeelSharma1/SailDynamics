"""Spatial shadow loads, contact discovery, conservative search, reproducibility."""
import csv
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from numpy.testing import assert_allclose

from sail_dynamics import Config
from sail_dynamics.environment import environment
from sail_dynamics.illumination import SpatialLoadEvaluator
from sail_dynamics.monte_carlo import StudySettings, run_study, METRICS, magnitudes
from sail_dynamics.sail import Sail, Hub
from sail_dynamics.sail.wrinkles import StochasticWrinkleField
from sail_dynamics.sampling import eclipse_refinement_times, time_grid
from sail_dynamics.frames.sail_body import SailBodyFrame
from sail_dynamics.shadows import visible_sun_fraction


class SpatialLoadTests(unittest.TestCase):
    def setUp(self):
        self.config = Config(radial_cells=4, angular_segments=16, wrinkle_m=0)
        self.evaluator = SpatialLoadEvaluator(self.config, Sail(self.config), Hub(self.config))

    def test_flat_sail_acquires_eclipse_torque(self):
        times, events = eclipse_refinement_times(self.config, 86400, .25)
        self.assertGreater(events, 0)
        env = environment(times, self.config)
        i = int(np.argmin(np.abs(env.sunlight-.5)))
        point = self.evaluator.snapshot(times[i])
        self.assertGreater(float(np.ptp(point.sunlight)), 1e-5)
        self.assertGreater(float(np.linalg.norm(point.loads.torques_body)), 1e-8)
        self.assertLess(float(np.linalg.norm(self.evaluator.sail.unit_torque)), 1e-18)
        direct_forces = point.beam_flux[:, None]*self.evaluator.unit_forces
        direct_moments = np.cross(self.evaluator.centers-np.asarray(self.config.attachment_m), direct_forces)
        assert_allclose(point.loads.forces_body, direct_forces.sum(axis=0), atol=1e-17)
        assert_allclose(point.loads.torques_body, direct_moments.sum(axis=0), atol=1e-17)

    def test_bounds_enclose_each_facet_and_all_loads(self):
        case = replace(self.config, diameter_m=100, wrinkle_m=1, wrinkle_model='stochastic')
        evaluator = SpatialLoadEvaluator(case, Sail(case), Hub(case))
        times, _ = eclipse_refinement_times(case, 86400, .5)
        env = environment(times[::3], case)
        low, high = evaluator.illumination_bounds(env)
        _, fractions = evaluator.facet_flux(env)
        self.assertTrue(np.all(fractions >= low[:, None]-1e-9))
        self.assertTrue(np.all(fractions <= high[:, None]+1e-9))
        actual = magnitudes(evaluator.loads(env))
        bounds = evaluator.magnitude_upper_bounds(env)
        self.assertTrue(np.all(actual <= bounds+1e-12))

    def test_vectorized_and_scalar_spatial_loads_agree(self):
        times, _ = eclipse_refinement_times(self.config, 86400, 1)
        chosen = times[::max(1, len(times)//12)]
        vector = self.evaluator.loads(environment(chosen, self.config))
        for i, time in enumerate(chosen):
            point = self.evaluator.snapshot(time)
            assert_allclose(vector.forces_body[i], point.loads.forces_body, atol=1e-17)
            assert_allclose(vector.torques_body[i], point.loads.torques_body, atol=1e-17)

    def test_body_frame_shadow_algebra_matches_direct_inertial_geometry(self):
        case = replace(self.config, wrinkle_m=.2, wrinkle_model='stochastic')
        evaluator = SpatialLoadEvaluator(case, Sail(case), Hub(case))
        times, _ = eclipse_refinement_times(case, 7200, .5)
        env = environment(times, case)
        offsets = np.einsum('tij,fj->tfi', SailBodyFrame(env.sun_unit).rotation_to_geocentric,
                            evaluator.centers)
        observers = env.satellite_m[:, None, :]+offsets
        sun = env.satellite_m+env.sun_unit*env.distance_m[:, None]
        direct = visible_sun_fraction(observers, sun[:, None, :])
        _, algebraic = evaluator.facet_flux(env)
        assert_allclose(algebraic, direct, atol=5e-8, rtol=0)

    def test_refinement_catches_between_minute_penumbra(self):
        times, _ = eclipse_refinement_times(self.config, 7200, .25)
        fine_times = np.arange(0, 7200, .2)
        fraction = environment(fine_times, self.config).sunlight
        partial = fine_times[(fraction > .05) & (fraction < .95)]
        self.assertGreater(len(partial), 0)
        indices = np.clip(np.searchsorted(times, partial), 1, len(times)-1)
        distance = np.minimum(np.abs(times[indices]-partial), np.abs(times[indices-1]-partial))
        self.assertLess(float(distance.max()), .251)

    def test_ten_second_refinement_preserves_short_fades_and_padding(self):
        duration = self.config.period_s
        times, events = eclipse_refinement_times(self.config, duration)
        self.assertGreater(events, 0)
        self.assertGreaterEqual(times.min(), 0)
        self.assertLessEqual(times.max(), duration)
        sunlight = environment(times, self.config).sunlight
        partial = times[(sunlight > .05) & (sunlight < .95)]
        # Both entry and exit must have an interior sample, even though each
        # finite-Sun transition is shorter than the requested 10-second step.
        self.assertGreaterEqual(len(partial), 2)
        self.assertTrue(np.any(np.diff(partial) > 60))
        for center in partial:
            near = times[np.abs(times-center) < 50]
            self.assertGreater(len(near), 8)
            self.assertLessEqual(float(np.diff(near).max()), 10)


class MonteCarloTests(unittest.TestCase):
    def test_default_is_complete_orbit_and_explicit_durations(self):
        config = Config(radial_cells=4, angular_segments=16)
        result = run_study(config, StudySettings(trials=2))
        self.assertEqual(result.minutes[0], 0)
        self.assertEqual(result.minutes[-1], config.period_s)
        self.assertTrue(np.all(np.diff(result.minutes) <= 60))
        self.assertEqual(result.settings.refine_step_s, 10)
        self.assertEqual(StudySettings(duration_orbits=2).duration_s(config), 2*config.period_s)
        self.assertEqual(StudySettings(duration_days=1).duration_s(config), 86400)
        times = result.peak_window_times()
        self.assertIn(result.peaks['assembly_torque_Nm']['elapsed_seconds'], times)
        self.assertTrue(np.all(np.diff(times) <= 10+1e-10))
        with tempfile.TemporaryDirectory() as directory:
            result.export(directory)
            report = json.loads((Path(directory)/'summary.json').read_text())
            self.assertEqual(report['duration_orbits'], 1)
            self.assertEqual(report['base_step_seconds'], 60)
            self.assertEqual(report['eclipse_step_seconds'], 10)

    def test_flat_ensemble_reuses_identical_geometry(self):
        config = Config(radial_cells=4, angular_segments=16, wrinkle_m=0)
        with tempfile.TemporaryDirectory() as directory:
            updates, messages = [], []
            result = run_study(config, StudySettings(trials=32, max_wrinkle_m=0),
                               checkpoint_dir=directory, progress=messages.append,
                               trial_progress=lambda completed, total: updates.append((completed, total)))
            self.assertEqual(updates, [(completed, 32) for completed in range(33)])
            self.assertFalse(any(message.startswith('Trial ') for message in messages))
            self.assertTrue(any('Monte Carlo summary' in message for message in messages))
            self.assertEqual(result.unique_geometry_evaluations, 1)
            self.assertIsNone(result.trials[0]['reused_from_trial'])
            for row in result.trials[1:]:
                self.assertEqual(row['reused_from_trial'], 0)
                for metric in METRICS:
                    self.assertEqual(row['max_'+metric], result.trials[0]['max_'+metric])
            # Reuse preserves the minute envelope, averages, and both winning loads.
            single = run_study(config, StudySettings(trials=2, max_wrinkle_m=0))
            assert_allclose(result.minute_envelope, single.minute_envelope, atol=0, rtol=0)
            self.assertEqual(result.peaks, single.peaks)
            assert_allclose(list(result.averages.values()), list(single.averages.values()), rtol=1e-14)
            result.export(directory)
            report = json.loads((Path(directory)/'progress.json').read_text())
            self.assertEqual(report['unique_geometry_evaluations'], 1)
            self.assertEqual(report['completed_trials'], 32)
            self.assertEqual(report['status'], 'complete')

    def test_ensemble_averages_and_assembly_only_reports(self):
        config = Config(radial_cells=4, angular_segments=16)
        settings = StudySettings(trials=3)
        with tempfile.TemporaryDirectory() as directory:
            messages = []
            result = run_study(config, settings, progress=messages.append, checkpoint_dir=directory)
            refined, _ = eclipse_refinement_times(config, config.period_s, settings.refine_step_s,
                                                   np.hypot(config.diameter_m/2, settings.max_wrinkle_m))
            times = np.unique(np.r_[result.minutes, refined])
            expected_peaks, expected_planforms = [], []
            for row in result.trials:
                case = replace(config, seed=row['pattern_seed'], wrinkle_m=row['wrinkle_m'],
                               wrinkle_model='stochastic')
                sail, hub = Sail(case), Hub(case)
                loads = SpatialLoadEvaluator(case, sail, hub).loads(environment(times, case))
                expected_peaks.append([np.linalg.norm(loads.forces_body, axis=-1).max(),
                                       np.linalg.norm(loads.torques_body, axis=-1).max()])
                rim = sail.mesh.vertices[-config.angular_segments:, :2]
                expected_planforms.append(.5*abs(np.sum(rim[:, 0]*np.roll(rim[:, 1], -1)
                                                        -rim[:, 1]*np.roll(rim[:, 0], -1))))
            self.assertEqual(METRICS, ('assembly_force_N', 'assembly_torque_Nm'))
            self.assertEqual(result.minute_envelope.shape[1], 2)
            self.assertAlmostEqual(result.averages['total_projected_area_m2'], np.mean(expected_planforms))
            flat = Sail(replace(config, wrinkle_m=0))
            self.assertAlmostEqual(result.averages['total_material_area_m2'],
                                   flat.material_area+Hub(config).projected_area)
            assert_allclose([result.averages['trial_peak_'+metric] for metric in METRICS],
                            np.mean(expected_peaks, axis=0), atol=1e-12, rtol=0)
            expected_rows = {'total_projected_area_m2': expected_planforms,
                             'trial_peak_assembly_force_N': np.array(expected_peaks)[:, 0],
                             'trial_peak_assembly_torque_Nm': np.array(expected_peaks)[:, 1]}
            for key, values in expected_rows.items():
                assert_allclose(list(result.statistics[key].values()),
                                [np.min(values), np.mean(values), np.max(values)], atol=1e-12, rtol=0)
            self.assertTrue(any('Lowest' in message and 'Average' in message and 'Highest' in message
                                for message in messages))
            self.assertTrue(any('not time averages' in message for message in messages))
            result.export(directory)
            report = json.loads((Path(directory)/'summary.json').read_text())
            self.assertEqual(report['averages'], result.averages)
            self.assertEqual(report['statistics'], result.statistics)
            self.assertEqual((Path(directory)/'summary_table.txt').read_text(), result.summary_table()+'\n')
            with (Path(directory)/'statistics.csv').open() as stream:
                rows = list(csv.DictReader(stream))
                self.assertEqual(len(rows), 4)
                for row in rows:
                    for stat in ('lowest', 'average', 'highest'):
                        self.assertEqual(float(row[stat]), result.statistics[row['quantity_SI']][stat])
            self.assertEqual(set(report['peaks']), set(METRICS))
            for peak in report['peaks'].values():
                self.assertNotIn('sail_force_body_N', peak)
                self.assertNotIn('sail_torque_body_Nm', peak)
            checkpoint = json.loads((Path(directory)/'progress.json').read_text())
            self.assertEqual(checkpoint['averages'], result.averages)
            for filename in ('trials.csv', 'minute_envelope.csv', 'peak_torque_window.csv', 'averages.csv'):
                with (Path(directory)/filename).open() as stream:
                    reader = csv.DictReader(stream)
                    self.assertFalse(any('sail_force' in field or 'sail_torque' in field
                                         for field in reader.fieldnames))
                    if filename == 'averages.csv':
                        values = next(reader)
                        for field, value in result.averages.items():
                            self.assertEqual(float(values[field]), value)
            self.assertEqual(len(list(Path(directory).glob('*_peak.npz'))), 2)

    def test_random_pattern_periodicity_bound_and_seed(self):
        r, theta = np.meshgrid(np.linspace(0, 1, 13), np.linspace(0, 2*np.pi, 49), indexing='ij')
        field = StochasticWrinkleField(.2, 7)
        height = field.heights(r, theta)
        assert_allclose(height[:, 0], height[:, -1], atol=1e-14)
        assert_allclose(height[[0, -1]], 0)
        self.assertAlmostEqual(np.max(np.abs(height)), .2)
        assert_allclose(field.heights(r, theta), height)
        self.assertGreater(np.max(np.abs(StochasticWrinkleField(.2, 8).heights(r, theta)-height)), .01)

    def test_pruned_search_matches_unpruned_grid_and_exports(self):
        config = Config(radial_cells=4, angular_segments=16)
        settings = StudySettings(trials=3, duration_days=.09, refine_step_s=.5)
        result = run_study(config, settings)
        refined, _ = eclipse_refinement_times(config, .09*86400, .5,
                                               np.hypot(config.diameter_m/2, settings.max_wrinkle_m))
        all_times = np.unique(np.r_[result.minutes, refined])
        self.assertEqual(result.trials[0]['wrinkle_m'], 0)
        self.assertEqual(result.trials[1]['wrinkle_m'], settings.max_wrinkle_m)
        self.assertTrue(np.all(np.diff(result.minutes) <= 60))
        for row in result.trials:
            case = replace(config, seed=row['pattern_seed'], wrinkle_m=row['wrinkle_m'],
                           wrinkle_model='stochastic')
            evaluator = SpatialLoadEvaluator(case, Sail(case), Hub(case))
            brute = magnitudes(evaluator.loads(environment(all_times, case))).max(axis=0)
            for j, metric in enumerate(METRICS):
                self.assertAlmostEqual(row['max_'+metric], brute[j], delta=1e-12)
                self.assertLessEqual(row['max_'+metric], row['upper_bound_'+metric]+1e-12)
        # Flat anchor: subminute penumbra produces a torque missing from most minute samples.
        self.assertGreater(result.trials[0]['max_assembly_torque_Nm'], 1e-8)
        with tempfile.TemporaryDirectory() as directory:
            result.export(directory)
            report = json.loads((Path(directory)/'summary.json').read_text())
            self.assertEqual(report['minute_epochs'], len(result.minutes))
            with (Path(directory)/'minute_envelope.csv').open() as stream:
                records = list(csv.DictReader(stream))
            self.assertEqual(len(records), len(result.minutes))
            data = np.load(Path(directory)/'assembly_torque_Nm_peak.npz')
            self.assertAlmostEqual(np.linalg.norm(data['torque_body_Nm']),
                                   result.peaks['assembly_torque_Nm']['magnitude'])
        again = run_study(config, settings)
        self.assertEqual(result.trials, again.trials)
        self.assertEqual(result.peaks, again.peaks)

    def test_invalid_study_parameters(self):
        for kwargs in ({'trials': 1}, {'min_wrinkle_m': -.1}, {'max_wrinkle_m': -.1},
                       {'refine_step_s': 0}, {'duration_days': -1}, {'seed': -1},
                       {'duration_orbits': .5}, {'duration_orbits': float('inf')},
                       {'duration_days': 1, 'duration_orbits': 1}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                StudySettings(**kwargs)


if __name__ == '__main__':
    unittest.main()
