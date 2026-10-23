# Sail Dynamics

A Python/NumPy solar radiation pressure (SRP) model with an interactive Matplotlib explorer. A circular or octagonal radial membrane extends from a rigid circular central hub to the outer sail rim. It follows a satellite on a prescribed circular 400 km Earth orbit for a calendar year, with hourly output and a finer view of each selected orbit. The mean sail plane always faces the Sun. Wrinkles change the individual triangular facets' orientations.

## Run

The project environment is already installed in `.venv` and configured in PyCharm. Run `sail_sim.py` in PyCharm or use:

```sh
.venv/bin/python sail_sim.py
```

For a fresh checkout, install with `uv sync` first. The only direct runtime dependencies are NumPy and Matplotlib. A desktop Matplotlib backend is needed for interaction. If PyCharm embeds a static plot, disable **Show plots in tool window**, then run the script in its own window.

```sh
# Change sail geometry, assembly mass, and orbital plane.
.venv/bin/python sail_sim.py --diameter 20 --hub-diameter 2 --mass 50 --wrinkle 0.3 --radial-cells 24 --segments 128 --inclination 90

# A completely flat, ideal mirror for comparison.
.venv/bin/python sail_sim.py --wrinkle 0 --specular 1 --diffuse 0 --hub-specular 1 --hub-diffuse 0

# Export without opening a window.
.venv/bin/python sail_sim.py --no-gui --csv output/year.csv

# Render a reproducible figure at a selected hour.
.venv/bin/python sail_sim.py --snapshot output/preview.png --hour 1440
```

`--diameter` sets the **flat material reference** circle diameter and `--hub-diameter` sets the rigid hub diameter in meters; the hub must be positive and fit inside the outline. Wrinkles keep the total 3D membrane area fixed and draw the outer rim inward, reducing the projected area that intercepts sunlight. The reference diameter is therefore not the deployed wrinkled diameter. `--radial-cells` controls radial resolution, while `--segments` controls angular resolution. `--width` and `--cells` remain legacy CLI aliases, now interpreted as reference circle diameter and radial cells. The Python configuration uses `diameter_m`, `hub_diameter_m`, `radial_cells`, and `angular_segments`.

`--help` lists all settings, including year, altitude, RAAN, orbital phase, wrinkle seed, and separate membrane/hub optical fractions. Dates use UTC. Normal years have 8,761 samples including both endpoints; leap years have 8,785.

## Circular and octagonal outlines at the same area

Select the outer shape with `--shape circular` (default) or `--shape octagonal`:

```sh
# Same material area and circular hub; different outer outlines.
.venv/bin/python sail_sim.py --shape circular --diameter 10 --hub-diameter 1
.venv/bin/python sail_sim.py --shape octagonal --diameter 10 --hub-diameter 1

# A one-day Monte Carlo study using the octagon.
.venv/bin/python sail_sim.py --shape octagonal --monte-carlo 32 --study-days 1 --no-gui --mc-dir output/octagonal_one_day
```

The **Shape** button beside the orbit-offset slider switches outlines in the explorer and updates geometry, loads, and plots immediately. The hub, total material area, optical properties, mass, and wrinkle settings stay the same. The hub must fit entirely inside the contracted octagon; an incompatible configuration is reported without changing the current shape.

`diameter_m` is a **flat material area reference**, rather than the corner-to-corner width of the octagon. At a fixed `--diameter` and `--segments`, both shapes have the same total material area including the hub. For the default 10 m reference circle and 96 segments, that area is approximately **78.484 m²**. In the flat case the octagon is scaled to that projected area, so its corners extend beyond the reference circle and its side centers lie inside it. With wrinkles, each outline contracts enough to keep its membrane's 3D facet area equal to its original flat area. Both shapes retain the same material area, while projected areas and contraction factors can differ. The hub geometry is unchanged.

The octagon has eight straight sides and vertices at 0°, 45°, … in sail X/Y axes. `--segments` must be a multiple of eight, so the mesh includes all corners exactly. More angular segments subdivide the same eight straight sides; they do not round the octagon. The circular hub retains the same angular discretization. The hub diameter must be smaller than the octagon's width across opposite sides. CLI output reports that width and the corner-to-corner width.

The `sail/boundary.py` object supplies nominal and contracted outlines. Eclipse-contact refinement uses the nominal corner radius as a conservative bound for all contracted patterns. Monte Carlo JSON records the flat reference geometry and each peak's actual geometry; peak NPZ files record material/projected areas and contraction scale. Hourly CSV exports include `sail_shape`, `reference_circle_diameter_m`, actual contracted `outer_corner_to_corner_m` and `outer_side_to_side_m`, material/projected areas, and `outer_scale` (1 for a flat sail). The CSV's former `sail_diameter_m` column is now named `reference_circle_diameter_m` to make its meaning explicit for both outlines.

For comparison studies, use matching ensemble seeds and wrinkle-height limits. A seed gives the same nodal heights at the same normalized radial/azimuthal positions for either shape, with different contraction, facet positions, and normals determined by the selected outline. Torque, force, and intercepted sunlight can change even though material area is preserved.

## Monte Carlo peak force and torque study

By default, each static random wrinkle pattern is simulated for **one complete orbit** (about 92.4 minutes at 400 km). Loads are sampled every minute, with **10-second or finer sampling near eclipse entry and exit**:

```sh
.venv/bin/python sail_sim.py --monte-carlo 32 --no-gui

# Explicitly select one orbit, with a specified sail/hub size.
.venv/bin/python sail_sim.py --monte-carlo 8 --study-orbits 1 --refine-step 10 --diameter 10 --hub-diameter 1.5 --no-gui --mc-dir output/one_orbit

# Optional longer study, or finer eclipse resolution for convergence checks.
.venv/bin/python sail_sim.py --monte-carlo 8 --study-days 1 --refine-step 1 --no-gui

# Sample peak heights between 0.02 and 0.3 m; reproducible ensemble seed.
.venv/bin/python sail_sim.py --monte-carlo 64 --mc-min-wrinkle .02 --wrinkle .3 --mc-seed 123 --no-gui

# Choose a different torque origin, e.g. the sail center.
.venv/bin/python sail_sim.py --monte-carlo 32 --attachment 0 0 0 --no-gui
```

Without `--no-gui`, the study also opens the normal Matplotlib explorer at the largest sampled **assembly torque**, using the winning wrinkle seed and exact event time. `--snapshot path.png` saves that view. All frame classes and geometry objects remain separate.

The stochastic model combines Gaussian spectral coefficients with random radial/azimuthal modes and phases. It includes asymmetric modes, is periodic around the hub, remains zero at both rims, and is normalized to a bounded peak nodal height. Frequencies are limited by mesh resolution. Each pattern stays fixed during its whole time simulation; the surface is not independently randomized at every minute. The maximum peak height is `--wrinkle`, and `--mc-min-wrinkle` sets the minimum. Heights are drawn uniformly between those limits, with the first two trials deliberately anchored at the minimum and maximum. The flat minimum-height trial (when minimum is zero) is particularly useful for force comparison. `--mc-seed` controls the reproducible ensemble; each pattern's own seed and height are recorded. Use `--wrinkle-model stochastic --seed <pattern_seed> --wrinkle <height>` to reproduce a particular surface outside the study.

Every trial covers the entire selected duration, including its exact endpoint. The default orbit has 94 minute-grid epochs at 400 km; loads at those times are instantaneous, not minute averages. Use `--study-orbits N` for more orbits or `--study-days D` for an explicit duration in days; these options are mutually exclusive. An annual search remains available with `--study-days 365` (366 for a leap year). A one-orbit result describes the initial Sun/orbit geometry on Jan 1 UTC; it does not establish an annual or seasonal maximum.

Orbit-minimum searches locate eclipses, including grazing penumbra; root searches then locate ingress and egress contacts. Their windows are expanded to cover the entire sail, padded by 60 seconds on either side, and sampled at `--refine-step` seconds or finer (default 10). Contact endpoints and each transition's midpoint are also included: at this altitude a penumbra can last less than 10 seconds, so regular ticks alone could miss it. The rest of the orbit, including deep eclipse, uses the minute grid. Midpoint samples prevent an entirely missed fade but do not guarantee its continuous-time peak. Reduce `--refine-step` to check peak convergence.

Identical geometry is evaluated only once and reused exactly. In particular, `--wrinkle 0 --monte-carlo 32` evaluates a single flat surface and records all 32 trial rows with `reused_from_trial` identifying duplicates. Random seeds cannot change a surface with zero wrinkle height. Nonzero patterns are still evaluated separately.

Each trial's console progress line prints its deployed **planform area (membrane + hub)** in m², alongside wrinkle height and peak loads. This is the projected area after wrinkle contraction, rather than the fixed material area.

Earth's apparent occultation is calculated at each facet center in inertial coordinates, including the wrinkle height. This gives each facet a different visible fraction of the finite Sun and permits uneven illumination to produce a moment even on a flat sail. The same calculation is used in the normal explorer and hourly exports. Full-sun and full-umbra epochs use exact uniform shortcuts. Refined epochs are skipped only when conservative illumination and triangle-inequality load bounds show that none of the trial's four running peaks can increase. The tests compare this search with evaluation of the complete unpruned schedule.

The four separate search objectives are the norms of **assembly force, assembly torque, membrane-only force, and membrane-only torque**. The hub contributes only to assembly values. All torques use the same configured `--attachment X Y Z` reference in sail coordinates, default `(0, 0, -0.8)` m. This reference is not automatically the spacecraft center of mass. Vector components, norms, UTC, trial, pattern seed, amplitude, and the range of facet illumination are retained for each winning event.

Each CLI run gets a unique folder under `output/monte_carlo`, printed before calculation, so simultaneous runs do not overwrite each other's results. Use `--mc-dir path` to choose an explicit directory; concurrent runs should use different paths. Already-running processes retain their old settings and must be restarted to use these defaults.

| File | Contents |
| --- | --- |
| `summary.json` | Settings, effective duration in seconds/orbits, sampling cadence, unique geometry count, peak values and times, reference point, winning seeds, empirical trial-peak quantiles, and conservative load bounds over the sampled patterns and scheduled epochs. |
| `progress.json` | Trial-by-trial status and peaks while the study runs; marked complete after data export. |
| `trials.csv` | Four peak magnitudes, event times, conservative bounds, material/projected areas, contraction scale, and any geometry reuse for each trial (zero-based trial indices). |
| `minute_envelope.csv` | Minute-by-minute maxima across the sampled patterns for the four objectives, plus center sunlight fraction. This envelope can select different patterns at different minutes. |
| `peak_torque_window.csv` | Force/torque components and norms at the requested refinement cadence over two minutes around the winning assembly-torque event, including the exact winning epoch and clipped to the study duration. |
| `study_summary.png` | Minute envelopes for studies shorter than a day (daily maxima for longer runs), refined winning events, and running peak convergence with trial count. |
| `assembly_force_N_peak.png`, `assembly_torque_Nm_peak.png` | Winning geometry, facet illumination, and nearby force/torque histories. |
| `<objective>_peak.npz` | Both meshes, per-facet beam irradiance/visible fraction, assembly force, and torque at each objective's winning epoch. |

These are **largest sampled magnitudes**, not guaranteed worst cases for all possible wrinkles or continuous time. The Gaussian spectral distribution is an exploratory geometry model, not a prediction of fabric behavior, and the anchor trials mean the empirical quantiles are not an unbiased risk estimate for a physical population. Increase trial count and compare results after reducing `--refine-step` and increasing mesh resolution before treating an estimate as converged. The saved conservative bounds apply only to the selected patterns and scheduled times under this optical model; they are not bounds for every physically possible sail. There is no structural/thermal response, attitude propagation, or material stress model.

## Explorer

- The **Earth view** shows the orbit, satellite, illuminated hemisphere, and direction of incoming sunlight. Drag to rotate the 3D view.
- The **3D sail** shows the radial membrane colored by the magnitude of SRP force per actual facet area in µPa. The dark central disk is the rigid hub. Blue arrows show representative membrane facet force directions and relative force per area. The green arrow shows the net membrane-plus-hub force at the attached assembly. Arrow lengths are scaled for visibility; facet and net arrows have separate scales. Sail height is visually exaggerated; axis coordinates remain meters.
- The **sunlight map** shows irradiance per actual facet area in W/m², including the hub. Wrinkles redistribute this quantity through incidence angle. The caption below the sail reports shape, total material and projected areas, hub diameter, and separate membrane/hub force magnitudes.
- The readout shows UTC, visible fraction of the Sun, Sun distance, intercepted optical power, total force, translational acceleration, force components in sail axes, torque about the attachment, and force relative to a flat sail with the same optical properties.
- The **year plot** shows instantaneous hourly force samples and the full-sun force reference. Click to select a time. The **selected orbit** plot shows orange force and green torque curves, using minute samples plus local subsecond refinement around detected sunlight changes; click it or use the continuous offset slider to inspect eclipse transitions. At the year's endpoints, the local plot is clipped to the simulated year. The backend Monte Carlo study additionally uses orbit-minimum/contact searches to catch grazing events.
- Use the **Year hour** slider, mouse wheel outside the 3D axes, or left/right arrows to move by an hour. Shift + left/right moves a day. Space or **Play year** advances the hourly samples.
- The **Wrinkle peak** slider regenerates the surface and updates the forces and year plot immediately. The pattern stays fixed while its amplitude changes. Zero gives the flat-sail limit.
- The **Hub Ø** slider changes hub diameter, rebuilding both the annular membrane and the rigid disk and updating all loads immediately. Assembly mass remains the configured value when geometry changes.
- **Export CSV** writes all hourly results for the current geometry to `output/sail_<year>_hourly.csv`. The CSV includes sail-frame and Earth-inertial assembly force vectors, magnitude, acceleration, moment about the attachment, intercepted power, sunlight fraction, Sun distance, and satellite position. Extra columns record separate membrane/hub force magnitudes, both diameters, and both projected areas. Export values are SI.

Hourly sampling undersamples a 92.41-minute orbit. These values are instantaneous loads, **not hourly averages or integrated impulses**. Use the fine orbit view to inspect eclipses. Computing an annual impulse or trajectory change would require integration with sufficiently small internal timesteps.

## Separate objects and files

Each reference frame has its own class and module. The sail, wrinkles, triangular facets, and optical calculations are also independent modules:

```text
sail_sim.py                         CLI / PyCharm entry point
sail_dynamics/
    config.py                      Immutable model settings
    constants.py                   SI constants
    frames/
        heliocentric.py            HeliocentricEclipticFrame
        geocentric.py              GeocentricEquatorialFrame
        orbital.py                 SatelliteOrbitalFrame
        sail_body.py               SailBodyFrame
    sail/
        surface.py                 Sail object and annular radial mesh construction
        boundary.py                Circular/octagonal reference and contracted outlines
        contraction.py             Fixed material area, unchanged hub, inward-moving rim
        hub.py                     Independent rigid hub object and optical response
        wrinkles.py                WrinkleField object and height generation
        facets.py                  Mesh object: connectivity, normals, areas
        optics.py                  Facet momentum exchange
    environment.py                 Sun/orbit/shadow state at any time
    shadows.py                     Earth/Sun angular-disc occultation
    illumination.py                Spatial facet shadows and conservative bounds
    loads.py                       Membrane/hub/assembly load result objects
    sampling.py                    Minute grid and eclipse contact refinement
    monte_carlo.py                 Ensemble search, peak records, reports, plots
    simulation.py                  Year sampling, assembly loads, CSV export
    view.py                        Matplotlib explorer and controls
tests/
    test_physics.py                 Analytic physics limits and transformations
    test_explorer.py                Widget / time / amplitude callback checks
    test_radial_sail.py             Annular topology, hub sizing, partitioned loads
    test_monte_carlo.py             Spatial moments, contact sampling, search bounds
    test_shapes.py                 Exact octagon edges, equal areas, unchanged hub
```

| Frame | Origin | Axes and transformations |
| --- | --- | --- |
| Heliocentric ecliptic | Sun | J2000 equinox X; ecliptic north Z. Calculates Earth's revolution. |
| Geocentric equatorial (ECI) | Earth | J2000 equinox X; celestial north Z. Translates Sun-centered positions and rotates ecliptic axes by the fixed obliquity. |
| Orbital plane | Earth | X along ascending node; Z along orbit angular momentum. Inclination and RAAN map orbital positions into ECI. |
| Sail body | Sail center | +Z points toward the Sun; X is celestial north cross +Z; Y completes the right-handed frame. Rotates vectors and translates positions into ECI. |

No Earth-fixed/spinning reference frame is used. The satellite and sail share the prescribed sail attitude; there is no independent spacecraft rotation in this model. Position transformations include origin translations, while force transformations only rotate vectors.

## Physics and assumptions

Default values: 10 m outer diameter, 1 m hub diameter, 12 radial cells and 96 angular segments (2,304 membrane triangles plus 96 flat hub triangles), 0.18 m peak absolute wrinkle height, 20 kg combined assembly mass, 51.6° orbit inclination, 0° RAAN, and 0° initial phase. The membrane has 85% specular reflection, 10% diffuse reflection, and 5% absorption. The hub is absorbing by default; use `--hub-specular` and `--hub-diffuse` to change its optical properties. These are example design inputs, rather than measured properties. The attachment is at `(0, 0, -0.8)` m in sail axes and can be changed through `Config.attachment_m`. The hub is a flat front-face optical model; its side walls, detailed satellite shape, and structural load transfer are not modeled.

The ideal flat circular membrane area is `pi/4 × (outer_diameter² - hub_diameter²)`. Straight triangular edges approximate the circular rims as regular polygons. For either outer shape, the **fixed membrane material area** is `N/8 × sin(2*pi/N) × (reference_circle_diameter² - hub_diameter²)`, where `N` is the angular segment count. With 96 segments this differs from the ideal circle by about 0.0714%. The hub uses the same inner polygon, so there is no gap or overlap. Increasing angular resolution improves the circle approximation. For a flat octagon, its circumradius is `sqrt(flat_total_area/(2*sqrt(2)))` and its apothem is that radius times `cos(pi/8)`.

`sail/contraction.py` solves for an outer-outline scale `s <= 1` so the sum of the wrinkled triangles' 3D areas equals that fixed material area. Radial rings interpolate between the unchanged hub and the scaled outline; the specified peak height stays unchanged. Circular rims remain circular and octagonal rims retain eight straight sides. The deployed projected membrane area becomes `s² × flat_total_area - hub_area`. Normals, centroids, lever arms, optical power, forces, torques, and plots all use these contracted vertices. Excessive wrinkle patterns that cannot fit around the unchanged hub with this contraction rule are rejected; the explorer rolls back invalid slider changes.

This conserves **total material area only**. Individual triangles can exchange area and their edge lengths can change, so it is not a locally inextensible fabric or elastic-equilibrium solution. Predicting physical draw-in from tension, booms, seams, and bending stiffness requires a structural wrinkle model and boundary conditions. Previous fixed-footprint results created extra material area when height was added; rerun those studies for consistent loads under this corrected geometry approximation.

Earth's revolution uses the [JPL approximate Keplerian elements](https://ssd.jpl.nasa.gov/planets/approx_pos.html) for 1800–2050, including eccentricity and a fixed ecliptic-to-equatorial obliquity. This provides seasonal Sun direction and inverse-square irradiance. The Earth/EM-barycenter distinction, planetary/lunar perturbations, and UTC–TDB difference are neglected. Earth's axial rotation is not modeled.

At each time, the model calculates the satellite-to-Sun direction and distance. Irradiance before facet projection is:

```text
I = 1361 W/m² × (1 AU / Sun distance)² × visible solar fraction
```

Shadowing uses the overlap of the apparent Earth and Sun discs, with full sunlight, umbra, and continuous penumbra. This is the common planar angular-disc overlap approximation, also used in [Orekit's SRP implementation](https://www.orekit.org/site-orekit-11.0.2/jacoco/org.orekit.forces.radiation/SolarRadiationPressure.java.html). The model uses uniform solar brightness and a spherical opaque Earth. Atmospheric refraction, limb darkening, Earth albedo, and infrared pressure are omitted. Visible fraction is calculated separately at every facet centroid, with center-to-Sun distance used for inverse-square flux. One centroid sample per triangle approximates the integral over that triangle. Optical momentum uses parallel rays along the center Sun direction: it does not integrate momentum direction over the partially visible solar disc. Mesh refinement and a more detailed extended-source optical model can change small torque estimates.

The membrane is triangulated between radial rings along spokes, with alternating cell diagonals and periodic angular connectivity. Rings interpolate from the circular hub rim to the contracted outer outline. Its inner edge attaches to the fixed hub rim. Both the inner and outer rings are flat in Z; the outer ring can move inward in X/Y.

Wrinkles remain a **temporary prescribed field** pending a replacement model. The current sinusoidal modes are periodic in azimuth and tapered to zero at both radial boundaries. They are not a calibrated stochastic or fabric mechanics model. Replace `WrinkleField.heights(radial_fraction, theta)` in `sail_dynamics/sail/wrinkles.py` when a better model is ready; `radial_fraction` runs from 0 at the hub to 1 at the outer rim, and `theta` is azimuth in radians. The returned array contains heights in meters, with the same shape as the input arrays. Heights must match at the angular seam and stay zero at both attachment boundaries for the present geometry. This interface is independent of the frames, hub, and optics.

The height field has no overhangs, so rays normal to its mean plane cannot be blocked by neighboring facets. Wrinkles are static in sail axes. There is no membrane elasticity, structural vibration, thermal deformation, or attitude-control solver.

For facet area `A`, sunward normal `n`, photon direction `k = (0,0,-1)`, and incidence cosine `mu = max(0, -k·n)`, the force is:

```text
F_facet = I A mu / c × [(1 - r_specular) k
                       - (2 r_specular mu + 2 r_diffuse / 3) n]
```

This follows incident-minus-outgoing photon momentum with specular and Lambertian diffuse reflection. The unreflected fraction is absorbed. Thermal reradiation and transmission are omitted. [NASA's reference sail model](https://ntrs.nasa.gov/citations/20150021404) describes more complete optical and thermal effects; this application implements the simpler momentum model above.

Sum both membrane and hub facet vectors for assembly force. Sum `(facet_center - attachment) × F_facet` over both objects for external moment about the attachment. The displayed acceleration is net SRP force divided by the combined assembly mass. The attachment moment is a load estimate, not a prediction of spacecraft rotation or elastic mount reaction. The dark square below the disk and its connector represent the attached assembly schematically. The flat-force ratio compares the wrinkled assembly with a flat membrane and the same hub, keeping each object's optical properties.

In full sunlight, wrinkles reduce the deployed projected area and intercepted power while keeping membrane material area fixed. They also change reflected momentum directions, force, and torque. During penumbra, individual facet visibility varies across position and height, redistributing optical power and torque. Changing hub size changes the partition of membrane/hub material, the contraction required for a given wrinkle field, and the optical load. As the mean plane tracks the Sun and wrinkles remain fixed within a trial, temporal redistribution comes from illumination and shadowing; the membrane does not spontaneously change shape.

The circular Earth orbit remains prescribed at the selected altitude. SRP, drag, J2 precession, Moon/Sun gravity, and attitude torques do not feed back into its trajectory. There is no aerodynamic load model. A year at 400 km is therefore an idealized load study, not a prediction of an unmaintained orbit's survival.

## Verification

```sh
.venv/bin/python -m unittest discover -v
```

Checks cover analytic flat absorption/mirror/diffuse limits, an independently derived tilted-mirror momentum case, conservation of 3D membrane area, inward rim contraction, reduced projected area and optical power, rejection of infeasible wrinkles, torque reference shifts, zero eclipse loads, seasonal distances, circular-orbit closure, right-handed orthonormal frame transformations and round trips, eclipse duration and smooth penumbra, leap years, full-year exports, radial seam connectivity, matching hub/membrane edges, area partition across hub sizes, hub optics, periodic wrinkle fields, spatial-shadow moments on a flat sail, facet illumination bounds, conservative-search agreement with the unpruned grid, between-minute eclipse sampling, ensemble reproducibility, study exports, and real Matplotlib slider/play/keyboard callbacks using a headless renderer.
