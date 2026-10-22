"""Reproducible bounded wrinkle ensemble and sampled peak-load search.

Every pattern is static over the requested duration. Minute results are evaluated
for all patterns. All eclipse transitions receive finer sampling. Conservative
triangle-inequality bounds skip refined epochs only when they cannot increase any
of that trial's four recorded peak magnitudes. No statistical global maximum or
confidence level is inferred from the largest random sample.
"""
from dataclasses import asdict, dataclass, replace
from datetime import timedelta
import csv
import json
from pathlib import Path

import numpy as np

from .environment import environment
from .illumination import SpatialLoadEvaluator
from .sail import Sail, Hub, SailBoundary
from .sampling import time_grid, eclipse_refinement_times

METRICS = ('assembly_force_N', 'assembly_torque_Nm', 'sail_force_N', 'sail_torque_Nm')


def magnitudes(loads):
    return np.stack([np.linalg.norm(v, axis=-1) for v in
                     (loads.forces_body, loads.torques_body,
                      loads.sail_forces_body, loads.sail_torques_body)], axis=-1)


@dataclass(frozen=True)
class StudySettings:
    trials: int = 32
    seed: int = 2026
    min_wrinkle_m: float = 0.0
    max_wrinkle_m: float = .18
    duration_days: float | None = None
    refine_step_s: float = .25

    def __post_init__(self):
        if not isinstance(self.trials, int) or self.trials < 2:
            raise ValueError("Monte Carlo study requires at least two trials.")
        if not np.isfinite(self.min_wrinkle_m) or not np.isfinite(self.max_wrinkle_m):
            raise ValueError("Wrinkle limits must be finite.")
        if not 0 <= self.min_wrinkle_m <= self.max_wrinkle_m:
            raise ValueError("Require 0 <= minimum wrinkle <= maximum wrinkle.")
        if self.duration_days is not None and (not np.isfinite(self.duration_days) or self.duration_days <= 0):
            raise ValueError("Study duration must be positive and finite.")
        if not np.isfinite(self.refine_step_s) or not 0 < self.refine_step_s <= 60:
            raise ValueError("Refinement step must be in (0, 60] seconds.")
        if self.seed < 0:
            raise ValueError("Monte Carlo seed must be nonnegative.")


@dataclass
class StudyResult:
    config: object
    settings: StudySettings
    minutes: np.ndarray
    center_sunlight: np.ndarray
    minute_envelope: np.ndarray
    trials: list
    peaks: dict
    eclipse_events: int
    refinement_epochs: int
    exact_refinement_candidates: int

    def peak_config(self, metric='assembly_torque_Nm'):
        peak = self.peaks[metric]
        return replace(self.config, seed=peak['pattern_seed'], wrinkle_m=peak['wrinkle_m'],
                       wrinkle_model='stochastic')

    def export(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        summary = {
            'interpretation': 'Largest sampled magnitudes; not guaranteed physical/global maxima.',
            'torque_reference_sail_m': list(self.config.attachment_m),
            'config': asdict(self.config), 'settings': asdict(self.settings),
            'geometry': SailBoundary(self.config).metadata(),
            'minute_epochs': len(self.minutes), 'eclipse_events': self.eclipse_events,
            'refinement_epochs': self.refinement_epochs,
            'exact_refinement_candidates': self.exact_refinement_candidates,
            'search': 'All minute epochs plus contact-refined epochs, with conservative pruning.',
            'wrinkle_distribution': 'Static Gaussian spectral patterns; uniform peak amplitude, '
                                    'with minimum/maximum amplitude anchor trials.',
            'peaks': self.peaks,
            'bounds_over_sampled_patterns_and_epochs': {
                metric: max(row['upper_bound_'+metric] for row in self.trials) for metric in METRICS},
            'empirical_trial_peak_quantiles': {
                str(q): {metric: float(np.quantile([row['max_'+metric] for row in self.trials], q))
                         for metric in METRICS} for q in (.5, .95, .99)},
        }
        (directory/'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
        with (directory/'trials.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(self.trials[0]))
            writer.writeheader()
            writer.writerows(self.trials)
        with (directory/'minute_envelope.csv').open('w', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['utc', 'elapsed_seconds', 'center_sunlight_fraction',
                             *['max_'+m for m in METRICS]])
            for seconds, sunlight, values in zip(self.minutes, self.center_sunlight, self.minute_envelope):
                writer.writerow([(self.config.start+timedelta(seconds=float(seconds))).isoformat(),
                                 seconds, sunlight, *values])
        for metric in METRICS:
            config = self.peak_config(metric)
            sail, hub = Sail(config), Hub(config)
            evaluator = SpatialLoadEvaluator(config, sail, hub)
            point = evaluator.snapshot(self.peaks[metric]['elapsed_seconds'])
            np.savez_compressed(directory/f'{metric}_peak.npz',
                                sail_shape=config.sail_shape,
                                total_projected_area_m2=sail.projected_area+hub.projected_area,
                                sail_vertices=sail.mesh.vertices, sail_triangles=sail.mesh.triangles,
                                hub_vertices=hub.mesh.vertices, hub_triangles=hub.mesh.triangles,
                                beam_flux_W_m2=point.beam_flux, sunlight_fraction=point.sunlight,
                                force_body_N=point.loads.forces_body,
                                torque_body_Nm=point.loads.torques_body)
        # Save a fine time trace of the worst assembly-torque event for review.
        case = self.peak_config()
        evaluator = SpatialLoadEvaluator(case, Sail(case), Hub(case))
        center = self.peaks['assembly_torque_Nm']['elapsed_seconds']
        times = np.linspace(max(0, center-60), min(self.minutes[-1], center+60), 481)
        env = environment(times, case)
        loads = evaluator.loads(env)
        with (directory/'peak_torque_window.csv').open('w', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['utc', 'seconds_from_peak', 'center_sunlight',
                             'force_x_N', 'force_y_N', 'force_z_N', 'torque_x_Nm',
                             'torque_y_Nm', 'torque_z_Nm', *METRICS])
            for i, seconds in enumerate(times):
                writer.writerow([(case.start+timedelta(seconds=float(seconds))).isoformat(),
                                 seconds-center, env.sunlight[i], *loads.forces_body[i],
                                 *loads.torques_body[i], *magnitudes(loads)[i]])
        checkpoint_path = directory/'progress.json'
        if checkpoint_path.exists():
            checkpoint = json.loads(checkpoint_path.read_text())
            checkpoint['status'] = 'complete'
            checkpoint_path.write_text(json.dumps(checkpoint, indent=2, allow_nan=False)+'\n')
        return directory


def run_study(config, settings=None, progress=None, checkpoint_dir=None):
    settings = settings or StudySettings(max_wrinkle_m=config.wrinkle_m)
    duration = config.duration_s if settings.duration_days is None else settings.duration_days*86400
    if duration > config.duration_s:
        raise ValueError("Study duration cannot exceed the configured calendar year.")
    minutes = time_grid(duration)
    env = environment(minutes, config)
    if progress:
        progress(f'Preparing {len(minutes):,} minute epochs and locating eclipse contacts…')
    extent = np.hypot(SailBoundary(config).max_radius_m, settings.max_wrinkle_m)
    refinement, events = eclipse_refinement_times(config, duration, settings.refine_step_s, extent)
    refined_env = environment(refinement, config)
    max_incident = max(float(env.unshadowed_flux.max()),
                       float(refined_env.unshadowed_flux.max()) if len(refinement) else 0)
    if progress:
        progress(f'{events:,} eclipse events; {len(refinement):,} refined epochs '
                 f'(≤{settings.refine_step_s:g} s).')
    rng = np.random.default_rng(settings.seed)
    seeds = rng.integers(0, 2**32, settings.trials)
    amplitudes = rng.uniform(settings.min_wrinkle_m, settings.max_wrinkle_m, settings.trials)
    amplitudes[0], amplitudes[1] = settings.min_wrinkle_m, settings.max_wrinkle_m
    envelope = np.zeros((len(minutes), len(METRICS)))
    global_values = np.full(len(METRICS), -np.inf)
    peaks, trials, evaluated = {}, [], 0
    for trial, (seed, amplitude) in enumerate(zip(seeds, amplitudes)):
        case = replace(config, seed=int(seed), wrinkle_m=float(amplitude), wrinkle_model='stochastic')
        sail, hub = Sail(case), Hub(case)
        evaluator = SpatialLoadEvaluator(case, sail, hub)
        loads = evaluator.loads(env)
        values = magnitudes(loads)
        np.maximum(envelope, values, out=envelope)
        indices = np.argmax(values, axis=0)
        best_values = values[indices, np.arange(len(METRICS))].copy()
        best_times = minutes[indices].copy()
        del loads, values
        # Bound each refined epoch before costly per-facet occultation. A trial's
        # own running maxima are used, preserving its convergence/peak statistics.
        upper = evaluator.magnitude_upper_bounds(refined_env)
        eligible = np.flatnonzero(np.any(upper > best_values*(1-1e-12), axis=-1))
        for start in range(0, len(eligible), 256):
            group = eligible[start:start+256]
            # Recheck after earlier groups may have improved the running maxima.
            group = group[np.any(upper[group] > best_values*(1-1e-12), axis=-1)]
            if not len(group):
                continue
            local_env = environment(refinement[group], case)
            candidate_values = magnitudes(evaluator.loads(local_env))
            evaluated += len(group)
            local_indices = np.argmax(candidate_values, axis=0)
            local_best = candidate_values[local_indices, np.arange(len(METRICS))]
            improved = local_best > best_values
            best_values[improved] = local_best[improved]
            best_times[improved] = refinement[group[local_indices[improved]]]
        row = {'trial': trial, 'pattern_seed': int(seed), 'wrinkle_m': float(amplitude)}
        n = evaluator.sail_count
        vector_sets = (evaluator.unit_forces, evaluator.unit_moments,
                       evaluator.unit_forces[:n], evaluator.unit_moments[:n])
        for j, metric in enumerate(METRICS):
            row['max_'+metric], row[metric+'_time_s'] = float(best_values[j]), float(best_times[j])
            row['upper_bound_'+metric] = float(max_incident*np.linalg.norm(vector_sets[j], axis=-1).sum())
            if best_values[j] > global_values[j]:
                point = evaluator.snapshot(best_times[j])
                # Save the exact spatial result, with each component auditable.
                actual = float(magnitudes(point.loads)[j])
                global_values[j] = actual
                peaks[metric] = {
                    'magnitude': actual, 'trial': trial, 'pattern_seed': int(seed),
                    'wrinkle_m': float(amplitude), 'elapsed_seconds': float(best_times[j]),
                    'utc': (config.start+timedelta(seconds=float(best_times[j]))).isoformat(),
                    'assembly_force_body_N': point.loads.forces_body.tolist(),
                    'assembly_torque_body_Nm': point.loads.torques_body.tolist(),
                    'sail_force_body_N': point.loads.sail_forces_body.tolist(),
                    'sail_torque_body_Nm': point.loads.sail_torques_body.tolist(),
                    'center_sunlight_fraction': float(point.environment.sunlight),
                    'facet_sunlight_min': float(point.sunlight.min()),
                    'facet_sunlight_max': float(point.sunlight.max()),
                }
        trials.append(row)
        if checkpoint_dir is not None:
            directory = Path(checkpoint_dir)
            directory.mkdir(parents=True, exist_ok=True)
            checkpoint = {'status': 'running', 'completed_trials': len(trials),
                          'planned_trials': settings.trials, 'config': asdict(config),
                          'settings': asdict(settings), 'peaks': peaks, 'trials': trials}
            temporary = directory/'progress.json.tmp'
            temporary.write_text(json.dumps(checkpoint, indent=2, allow_nan=False)+'\n')
            temporary.replace(directory/'progress.json')
        if progress:
            progress(f'Trial {trial+1}/{settings.trials}: height {amplitude:.4f} m; '
                     f'assembly peak F={best_values[0]*1e6:.3f} µN, '
                     f'|τ|={best_values[1]*1e6:.3f} µN·m.')
    return StudyResult(config, settings, minutes, env.sunlight, envelope, trials, peaks,
                       events, len(refinement), evaluated)


def save_figures(result, directory):
    """Standalone scientific figures; no browser or GUI dependency."""
    import matplotlib.pyplot as plt
    from matplotlib.tri import Triangulation
    directory = Path(directory)
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), layout='constrained')
    fig.suptitle(f'{result.config.sail_shape.capitalize()} sail · Monte Carlo largest sampled loads')
    # Block maxima preserve brief peaks when compressing a year into a plot.
    days = (result.minutes//86400).astype(int)
    unique = np.unique(days)
    for column, metric in enumerate(METRICS[:2]):
        daily = np.array([result.minute_envelope[days == day, column].max() for day in unique])
        axes[0, column].plot(unique, daily*1e6, color='#2874b2')
        peak = result.peaks[metric]
        axes[0, column].plot(peak['elapsed_seconds']/86400, peak['magnitude']*1e6,
                             '*', color='#e75535', ms=10, label='Refined search maximum')
        axes[0, column].set(xlabel='Days from Jan 1 UTC', ylabel='µN' if column == 0 else 'µN·m',
                            title='Assembly force' if column == 0 else 'Assembly torque about attachment')
        axes[0, column].legend(fontsize=8)
        case_peaks = np.array([row['max_'+metric] for row in result.trials])*1e6
        axes[1, column].plot(np.arange(len(case_peaks))+1, case_peaks, '.', label='Trial peak')
        axes[1, column].plot(np.arange(len(case_peaks))+1, np.maximum.accumulate(case_peaks),
                             label='Running largest sample')
        axes[1, column].set(xlabel='Trial (first two anchor amplitude limits)',
                            ylabel='µN' if column == 0 else 'µN·m')
        axes[1, column].legend(fontsize=8)
    for ax in axes.ravel():
        ax.grid(alpha=.2)
    fig.savefig(directory/'study_summary.png', dpi=150)
    plt.close(fig)
    for metric in METRICS[:2]:
        case = result.peak_config(metric)
        sail, hub = Sail(case), Hub(case)
        evaluator = SpatialLoadEvaluator(case, sail, hub)
        peak = result.peaks[metric]
        point = evaluator.snapshot(peak['elapsed_seconds'])
        n = len(sail.mesh.areas)
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout='constrained')
        fig.suptitle(f'{case.sail_shape.capitalize()} · {metric}: {peak["magnitude"]*1e6:.5g} micro-SI · {peak["utc"]}\n'
                     f'Trial {peak["trial"]}, pattern seed {peak["pattern_seed"]}, height {case.wrinkle_m:.4g} m')
        tri = Triangulation(sail.mesh.vertices[:, 0], sail.mesh.vertices[:, 1], sail.mesh.triangles)
        for ax, data, title in ((axes[0, 0], point.sunlight[:n], 'Visible Sun fraction per membrane facet'),
                                (axes[0, 1], sail.mesh.vertices[:, 2], 'Prescribed wrinkle height [m]')):
            if len(data) == len(sail.mesh.vertices):
                artist = ax.tripcolor(tri, data, shading='gouraud', cmap='coolwarm')
            else:
                spread = float(np.ptp(data))
                limits = (float(data.min()), float(data.max())) if spread > 1e-7 else (0, 1)
                artist = ax.tripcolor(tri, facecolors=data, cmap='inferno', vmin=limits[0], vmax=limits[1])
                if spread > 1e-7:
                    title += '\n(local color scale)'
            ax.add_patch(plt.Circle((0, 0), case.hub_diameter_m/2, color='#46576f'))
            ax.set(aspect='equal', xlabel='Sail X [m]', ylabel='Sail Y [m]', title=title)
            bar = fig.colorbar(artist, ax=ax)
            if ax is axes[0, 0] and spread > 1e-7:
                from matplotlib.ticker import FormatStrFormatter
                bar.ax.yaxis.set_major_formatter(FormatStrFormatter('%.6f'))
        duration = result.minutes[-1]
        times = np.linspace(max(0, peak['elapsed_seconds']-60), min(duration, peak['elapsed_seconds']+60), 481)
        values = magnitudes(evaluator.loads(environment(times, case)))
        for j, ax in enumerate(axes[1]):
            ax.plot(times-peak['elapsed_seconds'], values[:, j]*1e6, label='Assembly')
            ax.plot(times-peak['elapsed_seconds'], values[:, j+2]*1e6, '--', label='Membrane')
            ax.axvline(0, color='#e75535', lw=1)
            ax.set(xlabel='Seconds from peak', ylabel='µN' if j == 0 else 'µN·m')
            ax.grid(alpha=.2)
            ax.legend()
        fig.savefig(directory/f'{metric}_peak.png', dpi=150)
        plt.close(fig)
