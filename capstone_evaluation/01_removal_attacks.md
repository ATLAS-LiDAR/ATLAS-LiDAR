# Removal attacks vs. a reactive lane-change planner

A-HFR sector removal against five target types in CARLA 0.9.14 / Town06, CenterPoint
(nuScenes weights), 8 seeds per condition. "Planner + brake" includes the emergency
AEB; "planner only" disables it to isolate the planner's own decision. "p=1.00" /
"p=0.90" is the removal sector's keep-probability (1.00 = full removal inside the
sector, 0.90 = 90% removal). "Wedge tracks elevation" re-aims the removal sector's
elevation at the target every sweep, instead of fixing it at the first active frame.

## Parked car (`Sweep_Parked_TrackElev`)

### Smoothness

| Condition | n | Mean abs jerk, m/s³ | Lateral | Longitudinal | Peak lat jerk | Peak long jerk | Mean abs jerk incl. impact | Peak lateral accel, m/s² | Overshoot, m | Collided |
|---|---|---|---|---|---|---|---|---|---|---|
| No attack, planner + brake | 8 | 1.31 (1.30-1.33) | 0.61 (0.60-0.62) | 0.89 (0.88-0.90) | 2.9 (2.7-2.9) | 13.0 (12.8-13.1) | 1.31 (1.30-1.33) | 1.70 (1.70-1.72) | 0.25 (0.25-0.25) | 0/8 |
| Always on, p=1.00, planner + brake | 8 | 4.19 (4.18-4.28) | 0.09 (0.09-0.09) | 4.16 (4.15-4.26) | 0.5 (0.5-0.6) | 108.9 (77.7-109.0) | 4.19 (4.18-4.28) | 0.02 (0.02-0.02) | - | 0/8 |
| Always on, p=1.00, wedge tracks elevation, planner + brake | 8 | 2.34 (2.32-2.36) | 0.27 (0.26-0.27) | 2.23 (2.21-2.26) | 0.5 (0.5-0.6) | 13.1 (13.0-13.2) | 5.87 (5.79-5.95) | 0.02 (0.02-0.02) | - | 8/8 |
| Always on, p=1.00, wedge tracks elevation, planner only | 8 | 2.36 (2.33-2.37) | 0.27 (0.26-0.27) | 2.26 (2.22-2.27) | 0.5 (0.5-0.6) | 13.1 (12.8-13.2) | 6.76 (6.68-7.09) | 0.02 (0.02-0.02) | - | 8/8 |

### Plan outcome

| Condition | n | Merge committed | Cancelled a merge | Stayed in lane at the car | Collided | Speed at impact, m/s | Braked | First commit, s | First cancel, s | Closest approach, m |
|---|---|---|---|---|---|---|---|---|---|---|
| No attack, planner + brake | 8 | 8/8 | 0/8 | 0/8 | 0/8 | - | 0/8 | 1.30 (1.30-1.60) | - | 3.73 (3.73-3.74) |
| Always on, p=1.00, planner + brake | 8 | 8/8 | 0/8 | 8/8 | 0/8 | - | 8/8 | 4.82 (4.80-4.90) | - | 8.28 (7.30-8.67) |
| Always on, p=1.00, wedge tracks elevation, planner + brake | 8 | 8/8 | 0/8 | 8/8 | 8/8 | 7.6 (7.6-7.6) | 8/8 | 5.75 (5.65-5.85) | - | 4.84 (4.83-4.84) |
| Always on, p=1.00, wedge tracks elevation, planner only | 8 | 8/8 | 0/8 | 8/8 | 8/8 | 7.6 (7.6-7.6) | 0/8 | 5.80 (5.65-5.85) | - | 4.78 (4.78-4.79) |

**Takeaway:** without elevation tracking, full removal beats the detector but the
brake still catches every run (0/8 collisions). Tracking the target's elevation every
sweep beats the brake outright: 8/8 collisions whether or not the brake is enabled.

## Moving car (`Sweep_Moving_Car`)

### Smoothness

| Condition | n | Mean abs jerk, m/s³ | Lateral | Longitudinal | Peak lat jerk | Peak long jerk | Mean abs jerk incl. impact | Peak lateral accel, m/s² | Overshoot, m | Collided |
|---|---|---|---|---|---|---|---|---|---|---|
| No attack, planner + brake | 8 | 1.16 (1.15-1.17) | 0.54 (0.53-0.55) | 0.80 (0.79-0.80) | 2.9 (2.8-3.0) | 13.1 (12.9-13.1) | 1.16 (1.15-1.17) | 1.70 (1.70-1.72) | 0.25 (0.25-0.25) | 0/8 |
| Always on, p=1.00, planner + brake | 8 | 7.84 (7.69-7.93) | 0.77 (0.71-0.79) | 7.55 (7.41-7.64) | 5.0 (3.6-5.7) | 108.8 (106.9-134.6) | 7.84 (7.69-7.93) | 1.69 (1.63-1.70) | 0.02 (0.01-0.06) | 0/8 |
| Always on, p=1.00, planner only | 8 | 1.60 (1.53-1.62) | 0.42 (0.34-0.43) | 1.32 (1.31-1.34) | 2.8 (2.5-2.9) | 13.0 (12.9-13.1) | 3.98 (3.00-4.75) | 1.44 (0.72-1.60) | 0.10 (0.01-0.21) | 8/8 |
| Always on, p=1.00, wedge tracks elevation, planner + brake | 8 | 1.46 (1.45-1.47) | 0.27 (0.26-0.27) | 1.32 (1.31-1.34) | 0.6 (0.5-0.6) | 13.0 (12.8-13.1) | 11.43 (9.05-13.06) | 0.02 (0.02-0.02) | - | 8/8 |
| Always on, p=0.90, planner + brake | 8 | 1.40 (1.24-7.29) | 0.66 (0.57-1.01) | 0.83 (0.80-6.73) | 2.9 (2.7-18.1) | 13.1 (12.9-92.0) | 1.40 (1.24-7.29) | 1.71 (1.66-2.66) | 0.23 (0.03-0.25) | 0/8 |
| Always on, p=0.90, planner only | 8 | 1.44 (1.24-1.92) | 0.85 (0.57-1.34) | 0.81 (0.79-0.84) | 11.8 (2.8-19.3) | 13.0 (12.8-13.3) | 1.44 (1.24-1.92) | 1.72 (1.70-2.79) | 0.23 (0.16-0.24) | 0/8 |

### Plan outcome

| Condition | n | Merge committed | Cancelled a merge | Stayed in lane at the car | Collided | Speed at impact, m/s | Braked | First commit, s | First cancel, s | Closest approach, m |
|---|---|---|---|---|---|---|---|---|---|---|
| No attack, planner + brake | 8 | 8/8 | 0/8 | 0/8 | 0/8 | - | 0/8 | 1.17 (1.10-1.25) | - | 3.54 (3.54-3.54) |
| Always on, p=1.00, planner + brake | 8 | 8/8 | 0/8 | 0/8 | 0/8 | - | 8/8 | 7.78 (7.60-7.95) | - | 3.76 (3.64-3.80) |
| Always on, p=1.00, planner only | 8 | 8/8 | 0/8 | 0/8 | 8/8 | 7.6 (7.6-7.6) | 0/8 | 7.70 (7.55-8.00) | - | 3.49 (3.27-3.86) |
| Always on, p=1.00, wedge tracks elevation, planner + brake | 8 | 8/8 | 8/8 | 8/8 | 8/8 | 7.6 (7.6-7.6) | 8/8 | 9.10 (9.05-9.10) | 10.80 (10.75-11.80) | 4.83 (4.81-4.85) |
| Always on, p=0.90, planner + brake | 8 | 8/8 | 3/8 | 0/8 | 0/8 | - | 3/8 | 4.45 (0.80-6.00) | 2.15 (1.50-6.10) | 3.67 (3.47-3.76) |
| Always on, p=0.90, planner only | 8 | 8/8 | 7/8 | 0/8 | 0/8 | - | 0/8 | 1.05 (0.60-4.55) | 2.15 (1.70-3.00) | 3.71 (3.59-3.75) |

**Takeaway:** the moving car behaves like the parked car at p=1.00 with elevation
tracking (8/8 collisions regardless of brake), but partial removal (p=0.90) is not
enough against a moving target — 0/8 collisions in both brake conditions.

## Standing pedestrian (`Sweep_Pedestrian`)

### Smoothness

| Condition | n | Mean abs jerk, m/s³ | Lateral | Longitudinal | Peak lat jerk | Peak long jerk | Mean abs jerk incl. impact | Peak lateral accel, m/s² | Overshoot, m | Collided |
|---|---|---|---|---|---|---|---|---|---|---|
| No attack, planner + brake | 8 | 1.75 (1.30-3.68) | 1.08 (0.61-2.05) | 0.96 (0.91-2.14) | 2.9 (2.9-19.5) | 13.0 (12.9-13.2) | 1.75 (1.30-2.27) | 1.73 (1.72-3.23) | 0.20 (0.14-0.25) | 1/8 |
| Always on, p=1.00, planner + brake | 8 | 3.09 (2.21-3.76) | 0.27 (0.26-0.28) | 3.00 (2.09-3.67) | 0.6 (0.5-0.9) | 14.1 (13.0-22.5) | 3.05 (2.26-8.56) | 0.02 (0.02-0.05) | 0.20 (0.17-0.20) | 8/8 |
| Always on, p=1.00, planner only | 8 | 2.20 (2.18-2.22) | 0.27 (0.26-0.27) | 2.09 (2.07-2.12) | 0.5 (0.5-0.6) | 13.0 (12.9-13.1) | 1.54 (1.51-1.58) | 0.02 (0.02-0.02) | 0.20 (0.18-0.20) | 8/8 |
| Always on, p=1.00, wedge tracks elevation, planner + brake | 8 | 2.42 (2.34-2.60) | 0.27 (0.26-0.27) | 2.32 (2.23-2.51) | 0.5 (0.5-0.6) | 13.0 (12.8-14.8) | 7.39 (6.18-10.93) | 0.02 (0.02-0.02) | 0.14 (0.01-0.60) | 8/8 |
| Always on, p=0.90, planner + brake | 8 | 3.46 (1.88-3.87) | 0.48 (0.43-1.12) | 3.30 (0.98-3.71) | 2.2 (2.1-18.6) | 13.1 (12.9-14.8) | 3.04 (1.88-6.27) | 1.31 (0.94-3.33) | 0.21 (0.05-0.25) | 6/8 |
| Always on, p=0.90, planner only | 8 | 2.34 (1.83-2.46) | 0.48 (0.30-1.07) | 2.08 (0.94-2.09) | 2.6 (2.1-18.5) | 13.0 (12.6-13.2) | 1.92 (1.66-1.98) | 1.14 (0.24-2.22) | 0.15 (0.14-0.19) | 7/8 |

### Plan outcome

| Condition | n | Merge committed | Cancelled a merge | Stayed in lane at the car | Collided | Speed at impact, m/s | Braked | First commit, s | First cancel, s | Closest approach, m |
|---|---|---|---|---|---|---|---|---|---|---|
| No attack, planner + brake | 8 | 8/8 | 8/8 | 0/8 | 1/8 | 7.6 | 1/8 | 0.70 (0.70-3.05) | 5.33 (1.85-5.70) | 3.73 (1.94-3.82) |
| Always on, p=1.00, planner + brake | 8 | 8/8 | 0/8 | 0/8 | 8/8 | 6.8 (6.2-7.6) | 8/8 | 5.40 (5.25-6.20) | - | 1.83 (1.78-2.61) |
| Always on, p=1.00, planner only | 8 | 8/8 | 0/8 | 0/8 | 8/8 | 7.6 (7.6-7.6) | 0/8 | 5.50 (5.30-6.20) | - | 1.87 (1.74-1.90) |
| Always on, p=1.00, wedge tracks elevation, planner + brake | 8 | 8/8 | 2/8 | 1/8 | 8/8 | 7.5 (7.4-7.6) | 8/8 | 5.80 (5.75-5.80) | 8.57 (8.15-9.00) | 2.21 (1.30-2.58) |
| Always on, p=0.90, planner + brake | 8 | 8/8 | 4/8 | 3/8 | 6/8 | 7.1 (7.0-7.2) | 7/8 | 4.50 (3.70-4.75) | 6.07 (5.95-6.10) | 1.71 (1.29-2.40) |
| Always on, p=0.90, planner only | 8 | 8/8 | 6/8 | 3/8 | 7/8 | 7.6 (7.6-7.6) | 0/8 | 4.72 (3.20-5.10) | 5.78 (4.45-5.95) | 1.73 (1.42-1.93) |

**Caveat specific to this target:** even the no-attack condition collides 1/8 and the
planner cancels its own merge 8/8 — CenterPoint's nuScenes weights are themselves
unreliable on CARLA-rendered pedestrians, independent of any attack. Read every
pedestrian number against that baseline, not as a clean 0% control.

## Standing cyclist (`Sweep_Cyclist`)

### Smoothness

| Condition | n | Mean abs jerk, m/s³ | Lateral | Longitudinal | Peak lat jerk | Peak long jerk | Mean abs jerk incl. impact | Peak lateral accel, m/s² | Overshoot, m | Collided |
|---|---|---|---|---|---|---|---|---|---|---|
| No attack, planner + brake | 8 | 1.30 (1.29-1.42) | 0.60 (0.59-0.73) | 0.92 (0.91-0.92) | 2.9 (2.7-3.1) | 13.1 (12.8-13.2) | 1.30 (1.29-1.42) | 1.71 (1.69-1.72) | 0.25 (0.25-0.26) | 0/8 |
| Always on, p=1.00, planner + brake | 8 | 4.94 (4.17-11.99) | 0.10 (0.09-0.20) | 4.91 (4.14-11.95) | 0.5 (0.5-0.9) | 108.7 (107.1-109.2) | 4.94 (4.17-11.53) | 0.02 (0.02-0.08) | - | 2/8 |
| Always on, p=1.00, planner only | 8 | 2.24 (2.22-2.28) | 0.27 (0.27-0.29) | 2.12 (2.10-2.16) | 0.6 (0.6-1.2) | 13.0 (12.8-13.3) | 4.98 (4.65-30.75) | 0.05 (0.05-0.17) | 0.22 (0.19-0.28) | 8/8 |
| Always on, p=1.00, wedge tracks elevation, planner + brake | 8 | 3.01 (3.00-3.03) | 0.27 (0.26-0.27) | 2.91 (2.90-2.94) | 0.5 (0.5-0.6) | 19.0 (18.8-19.3) | 4.94 (4.90-4.99) | 0.02 (0.02-0.02) | - | 8/8 |
| Always on, p=0.90, planner + brake | 8 | 7.03 (4.32-15.66) | 0.25 (0.12-0.71) | 7.00 (4.16-15.61) | 2.7 (0.6-6.0) | 132.9 (104.3-135.0) | 7.03 (4.32-12.28) | 0.28 (0.16-1.67) | 0.25 | 1/8 |
| Always on, p=0.90, planner only | 8 | 2.41 (1.38-2.49) | 0.54 (0.38-0.63) | 2.12 (0.89-2.15) | 2.7 (2.4-2.9) | 13.0 (12.8-13.2) | 4.51 (1.38-5.04) | 1.53 (0.58-1.71) | 0.23 (0.00-0.26) | 6/8 |

### Plan outcome

| Condition | n | Merge committed | Cancelled a merge | Stayed in lane at the car | Collided | Speed at impact, m/s | Braked | First commit, s | First cancel, s | Closest approach, m |
|---|---|---|---|---|---|---|---|---|---|---|
| No attack, planner + brake | 8 | 8/8 | 0/8 | 0/8 | 0/8 | - | 0/8 | 0.60 (0.60-0.60) | - | 3.70 (3.70-3.70) |
| Always on, p=1.00, planner + brake | 8 | 8/8 | 0/8 | 8/8 | 2/8 | 2.7 (2.7-2.7) | 8/8 | 5.10 (5.05-5.20) | - | 5.77 (2.92-6.70) |
| Always on, p=1.00, planner only | 8 | 8/8 | 1/8 | 3/8 | 8/8 | 7.6 (7.6-7.6) | 0/8 | 5.15 (5.05-5.15) | 10.10 | 1.52 (0.99-2.11) |
| Always on, p=1.00, wedge tracks elevation, planner + brake | 8 | 8/8 | 0/8 | 8/8 | 8/8 | 6.9 (6.9-6.9) | 8/8 | 5.60 (5.60-5.60) | - | 2.91 (2.89-2.92) |
| Always on, p=0.90, planner + brake | 8 | 8/8 | 1/8 | 7/8 | 1/8 | 2.7 | 8/8 | 4.60 (3.65-4.90) | 4.20 | 5.49 (1.51-6.84) |
| Always on, p=0.90, planner only | 8 | 8/8 | 0/8 | 2/8 | 6/8 | 7.6 (7.6-7.6) | 0/8 | 4.40 (3.05-4.85) | - | 2.01 (1.35-3.30) |

**Takeaway:** standing cyclist behaves similarly to the parked car (elevation tracking
beats the brake 8/8), but without tracking, full removal with the brake on still hits
2/8 — the brake is less reliable on the cyclist's geometry than on the car's.

## Moving cyclist (`Sweep_Cyclist_Moving`, commanded 2.5 m/s, actual ~1.6 m/s)

### Smoothness

| Condition | n | Mean abs jerk, m/s³ | Lateral | Longitudinal | Peak lat jerk | Peak long jerk | Mean abs jerk incl. impact | Peak lateral accel, m/s² | Overshoot, m | Collided |
|---|---|---|---|---|---|---|---|---|---|---|
| No attack, planner + brake | 8 | 1.15 (1.14-1.17) | 0.54 (0.53-0.55) | 0.81 (0.79-0.81) | 2.9 (2.7-3.0) | 13.1 (12.9-13.1) | 1.15 (1.14-1.17) | 1.71 (1.70-1.72) | 0.25 (0.25-0.25) | 0/8 |
| Always on, p=1.00, planner + brake | 8 | 7.48 (6.29-7.89) | 0.65 (0.61-0.83) | 7.19 (6.05-7.60) | 3.7 (3.1-10.1) | 107.3 (107.1-109.0) | 7.48 (6.29-7.89) | 1.70 (1.62-1.80) | 0.07 (0.01-0.14) | 0/8 |
| Always on, p=1.00, planner only | 8 | 1.83 (1.76-1.88) | 0.36 (0.28-0.38) | 1.62 (1.60-1.65) | 2.7 (1.4-2.9) | 13.0 (12.9-13.2) | 5.97 (2.06-10.25) | 0.72 (0.17-0.83) | 0.29 (0.21-3.05) | 8/8 |
| Always on, p=1.00, wedge tracks elevation, planner + brake | 8 | 10.75 (9.97-10.82) | 0.30 (0.26-0.39) | 10.69 (9.91-10.76) | 4.3 (3.3-6.4) | 108.9 (108.5-109.0) | 10.75 (9.97-10.82) | 0.45 (0.31-0.62) | - | 0/8 |
| Always on, p=0.90, planner + brake | 8 | 3.82 (1.25-7.79) | 0.60 (0.53-0.96) | 3.36 (0.80-7.51) | 2.9 (2.4-20.0) | 60.1 (12.8-133.0) | 3.82 (1.25-7.79) | 1.69 (1.54-3.38) | 0.23 (0.12-0.25) | 0/8 |
| Always on, p=0.90, planner only | 8 | 1.25 (1.24-1.91) | 0.58 (0.44-0.98) | 0.80 (0.80-1.61) | 2.9 (2.7-19.3) | 13.0 (12.9-13.2) | 1.25 (1.24-1.71) | 1.71 (1.36-2.52) | 0.24 (0.19-0.24) | 1/8 |

### Plan outcome

| Condition | n | Merge committed | Cancelled a merge | Stayed in lane at the car | Collided | Speed at impact, m/s | Braked | First commit, s | First cancel, s | Closest approach, m |
|---|---|---|---|---|---|---|---|---|---|---|
| No attack, planner + brake | 8 | 8/8 | 0/8 | 0/8 | 0/8 | - | 0/8 | 0.75 (0.60-1.00) | - | 3.60 (3.59-3.60) |
| Always on, p=1.00, planner + brake | 8 | 8/8 | 0/8 | 0/8 | 0/8 | - | 8/8 | 6.40 (6.35-6.55) | - | 2.91 (2.54-3.15) |
| Always on, p=1.00, planner only | 8 | 8/8 | 0/8 | 2/8 | 8/8 | 7.6 (7.6-7.6) | 0/8 | 6.40 (6.35-6.65) | - | 1.75 (0.83-2.07) |
| Always on, p=1.00, wedge tracks elevation, planner + brake | 8 | 8/8 | 8/8 | 8/8 | 0/8 | - | 8/8 | 7.00 (7.00-7.00) | 10.30 (10.20-10.35) | 3.79 (3.78-3.79) |
| Always on, p=0.90, planner + brake | 8 | 8/8 | 1/8 | 0/8 | 0/8 | - | 6/8 | 5.45 (3.60-6.25) | 5.40 | 2.72 (2.48-3.74) |
| Always on, p=0.90, planner only | 8 | 8/8 | 1/8 | 1/8 | 1/8 | 7.6 | 0/8 | 5.05 (3.30-6.10) | 4.65 | 2.72 (1.07-3.63) |

**Takeaway:** a moving cyclist is the one case where elevation tracking does *not*
beat the brake — the brake catches all 8/8 even with tracking on, because the moving
target keeps drifting out from under the fixed-length removal segment.

Values throughout are median (min-max) over seeds. Jerk is measured from the first
second to the end of the drive, or to just before the first collision.
