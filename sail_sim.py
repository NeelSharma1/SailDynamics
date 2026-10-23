"""Interactive solar-sail simulator. Run this file to open the Matplotlib explorer."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path

import numpy as np

from sail_dynamics import Config, Simulation
from sail_dynamics.constants import SOLAR_CONSTANT


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, default=2026)
    parser.add_argument("--altitude-km", type=float, default=400)
    parser.add_argument("--inclination", type=float, default=51.6, help="Orbit inclination [deg]")
    parser.add_argument("--raan", type=float, default=0, help="Ascending-node longitude [deg]")
    parser.add_argument("--phase", type=float, default=0, help="Initial orbital phase [deg]")
    parser.add_argument("--diameter", "--width", dest="diameter", type=float, default=10,
                        help="Flat material reference circle diameter [m]; wrinkles contract the outline")
    parser.add_argument("--shape", choices=("circular", "octagonal"), default="circular",
                        help="Outer sail outline; the hub and total material area are preserved")
    parser.add_argument("--hub-diameter", type=float, default=1, help="Rigid central hub diameter [m]")
    parser.add_argument("--radial-cells", "--cells", dest="radial_cells", type=int, default=12,
                        help="Radial mesh cells from hub rim to sail rim")
    parser.add_argument("--segments", type=int, default=96, help="Angular mesh segments")
    parser.add_argument("--wrinkle", type=float, default=.18, help="Peak absolute wrinkle height [m]")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--wrinkle-model", choices=("modal", "stochastic"), default="modal")
    parser.add_argument("--mass", type=float, default=20, help="Combined assembly mass [kg]")
    parser.add_argument("--specular", type=float, default=.85)
    parser.add_argument("--diffuse", type=float, default=.10)
    parser.add_argument("--hub-specular", type=float, default=0)
    parser.add_argument("--hub-diffuse", type=float, default=0)
    parser.add_argument("--attachment", type=float, nargs=3, default=(0, 0, -.8),
                        metavar=('X', 'Y', 'Z'), help="Torque reference in sail axes [m]")
    parser.add_argument("--monte-carlo", type=int, metavar="TRIALS",
                        help="Run bounded stochastic wrinkle study (at least 2 trials)")
    parser.add_argument("--mc-dir", type=Path,
                        help="Report directory; default is a unique folder under output/monte_carlo")
    parser.add_argument("--mc-seed", type=int, default=2026)
    parser.add_argument("--mc-min-wrinkle", type=float, default=0,
                        help="Minimum sampled peak wrinkle height [m]; maximum is --wrinkle")
    duration = parser.add_mutually_exclusive_group()
    duration.add_argument("--study-days", type=float,
                          help="Override study duration in days from Jan 1 UTC")
    duration.add_argument("--study-orbits", type=float,
                          help="Study duration in complete orbits (at least 1); default 1 orbit")
    parser.add_argument("--refine-step", type=float, default=10,
                        help="Maximum timestep near eclipse [s], default 10; contacts/midpoints included")
    parser.add_argument("--csv", type=Path, help="Export hourly results to this path")
    parser.add_argument("--snapshot", type=Path, help="Save a figure without opening a window")
    parser.add_argument("--hour", type=int, default=0, help="Initial hour index")
    parser.add_argument("--no-gui", action="store_true", help="Compute and optionally export only")
    args = parser.parse_args(argv)
    try:
        config = Config(year=args.year, altitude_m=args.altitude_km*1000,
                        inclination_deg=args.inclination, raan_deg=args.raan,
                        phase_deg=args.phase, diameter_m=args.diameter, sail_shape=args.shape,
                        hub_diameter_m=args.hub_diameter, radial_cells=args.radial_cells,
                        angular_segments=args.segments,
                        wrinkle_m=args.wrinkle, seed=args.seed, wrinkle_model=args.wrinkle_model,
                        mass_kg=args.mass, attachment_m=tuple(args.attachment),
                        specular=args.specular, diffuse=args.diffuse,
                        hub_specular=args.hub_specular, hub_diffuse=args.hub_diffuse)
    except ValueError as error:
        parser.error(str(error))
    replay_seconds = None
    if args.monte_carlo is not None:
        from sail_dynamics.monte_carlo import StudySettings, run_study, save_figures
        try:
            settings = StudySettings(trials=args.monte_carlo, seed=args.mc_seed,
                                     min_wrinkle_m=args.mc_min_wrinkle, max_wrinkle_m=args.wrinkle,
                                     duration_days=args.study_days, duration_orbits=args.study_orbits,
                                     refine_step_s=args.refine_step)
            if args.mc_dir is None:
                stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
                args.mc_dir = Path('output/monte_carlo')/f'{config.sail_shape}_{stamp}_{os.getpid()}'
            print(f'Study reports: {args.mc_dir.resolve()}', flush=True)
            result = run_study(config, settings, progress=lambda message: print(message, flush=True),
                               checkpoint_dir=args.mc_dir)
        except ValueError as error:
            parser.error(str(error))
        result.export(args.mc_dir)
        import matplotlib
        if args.no_gui or args.snapshot:
            matplotlib.use("Agg")
        save_figures(result, args.mc_dir)
        for metric, peak in result.peaks.items():
            print(f"Largest sampled {metric}: {peak['magnitude']:.9g} at {peak['utc']} "
                  f"(trial {peak['trial']}, seed {peak['pattern_seed']}, height {peak['wrinkle_m']:.5g} m)")
        print(f"Saved study to {args.mc_dir.resolve()}. These are sampled maxima, not guaranteed upper limits.")
        if args.no_gui and not args.snapshot and not args.csv:
            return result
        config = result.peak_config('assembly_torque_Nm')
        replay_seconds = result.peaks['assembly_torque_Nm']['elapsed_seconds']
    try:
        simulation = Simulation(config)
    except ValueError as error:
        parser.error(str(error))
    if not 0 <= args.hour < len(simulation.hours):
        parser.error(f"--hour must be between 0 and {len(simulation.hours)-1}")
    print(f"{len(simulation.hours):,} hourly samples; {len(simulation.sail.mesh.triangles):,} facets; "
          f"orbital period {config.period_s/60:.2f} minutes.")
    print(f"{config.sail_shape.capitalize()} sail; reference circle diameter {config.diameter_m:g} m; "
          f"hub diameter {config.hub_diameter_m:g} m; "
          f"membrane material area {simulation.sail.material_area:.3f} m²; "
          f"membrane projected area {simulation.sail.projected_area:.3f} m²; "
          f"hub area {simulation.hub.projected_area:.3f} m².")
    print(f"Total material area {simulation.material_area:.6g} m²; "
          f"projected area {simulation.projected_area:.6g} m²; outer scale {simulation.sail.boundary.scale:.6f}.")
    if config.sail_shape == 'octagonal':
        boundary = simulation.sail.boundary
        print(f"Octagon corner-to-corner {2*boundary.max_radius_m:.6g} m; "
              f"side-to-side {2*boundary.min_radius_m:.6g} m; "
              f"total projected area {simulation.projected_area:.6g} m².")
    print(f"Assembly full-sun SRP at 1 AU: "
          f"{np.linalg.norm(simulation.unit_force)*SOLAR_CONSTANT*1e6:.2f} µN.")
    if args.csv:
        print(f"Saved {simulation.export_csv(args.csv).resolve()}")
    if args.snapshot or not args.no_gui:
        if args.snapshot:
            import matplotlib
            matplotlib.use("Agg")
        from sail_dynamics.view import Explorer
        explorer = Explorer(simulation)
        explorer.hour_slider.set_val(args.hour if replay_seconds is None else int(round(replay_seconds/3600)))
        if replay_seconds is not None:
            explorer.orbit_slider.set_val((replay_seconds-simulation.hours[explorer.index])/60)
        if args.snapshot:
            args.snapshot.parent.mkdir(parents=True, exist_ok=True)
            explorer.fig.savefig(args.snapshot, dpi=140, facecolor=explorer.fig.get_facecolor())
            print(f"Saved {args.snapshot.resolve()}")
        elif not args.no_gui:
            explorer.show()
    return simulation


if __name__ == "__main__":
    main()
