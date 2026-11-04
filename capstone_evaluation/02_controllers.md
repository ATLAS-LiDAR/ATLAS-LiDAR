# Does the steering controller change the attack's effect?

Same parked-car removal attack as in `01_removal_attacks.md`, repeated through three
lateral controllers (Stanley, pure pursuit, linear MPC) to check whether the attack's
effect on the *plan* depends on how the ego tracks the planned path. 8 seeds per
condition. `Sweep_Controllers`.

## Smoothness

| Condition | n | Mean abs jerk, m/s³ | Lateral | Longitudinal | Peak lat jerk | Peak long jerk | Mean abs jerk incl. impact | Peak lateral accel, m/s² | Overshoot, m | Collided |
|---|---|---|---|---|---|---|---|---|---|---|
| MPC: No attack, planner + brake | 8 | 1.33 (1.30-1.49) | 0.66 (0.65-0.66) | 0.91 (0.88-1.10) | 2.9 (2.8-3.0) | 13.1 (12.6-13.2) | 1.33 (1.30-1.49) | 1.66 (1.64-1.67) | 0.00 (0.00-0.00) | 0/8 |
| Pure pursuit: No attack, planner + brake | 8 | 1.17 (1.16-1.18) | 0.46 (0.46-0.47) | 0.89 (0.88-0.90) | 2.1 (1.9-2.2) | 13.1 (12.6-13.2) | 1.17 (1.16-1.18) | 1.17 (1.15-1.19) | 0.08 (0.08-0.08) | 0/8 |
| Stanley: No attack, planner + brake | 8 | 1.31 (1.29-1.33) | 0.61 (0.60-0.62) | 0.88 (0.87-0.90) | 2.9 (2.8-2.9) | 13.1 (12.9-13.2) | 1.31 (1.29-1.33) | 1.71 (1.70-1.72) | 0.25 (0.25-0.25) | 0/8 |
| MPC: Late start (10-20 m), p=1.00, planner only | 8 | 1.33 (1.32-1.52) | 0.66 (0.66-0.85) | 0.91 (0.89-0.91) | 2.9 (2.8-8.0) | 13.1 (12.9-13.3) | 1.33 (1.32-1.52) | 1.66 (1.65-1.67) | 0.00 (0.00-0.00) | 0/8 |
| MPC: Always on, p=1.00, planner + brake | 8 | 4.18 (4.16-4.23) | 0.10 (0.09-0.10) | 4.16 (4.13-4.20) | 0.8 (0.6-0.8) | 107.8 (106.8-109.2) | 4.18 (4.16-4.23) | 0.07 (0.05-0.08) | - | 0/8 |
| MPC: Always on, p=1.00, planner only | 8 | 2.50 (2.47-2.54) | 0.42 (0.39-0.46) | 2.24 (2.22-2.25) | 2.7 (2.5-2.7) | 12.9 (12.6-13.3) | 6.49 (6.42-6.58) | 0.72 (0.60-0.86) | - | 8/8 |
| Pure pursuit: Late start (10-20 m), p=1.00, planner only | 8 | 1.19 (1.17-1.25) | 0.48 (0.47-0.53) | 0.88 (0.87-0.90) | 2.8 (1.9-5.4) | 13.1 (12.9-13.2) | 1.19 (1.17-1.25) | 1.19 (1.15-1.49) | 0.01 (0.00-0.08) | 0/8 |
| Pure pursuit: Always on, p=1.00, planner + brake | 8 | 4.17 (4.09-5.46) | 0.13 (0.11-0.18) | 4.14 (4.06-5.43) | 1.7 (1.3-2.5) | 107.8 (101.6-114.5) | 4.17 (4.09-5.46) | 0.16 (0.12-0.28) | - | 0/8 |
| Pure pursuit: Always on, p=1.00, planner only | 8 | 2.47 (2.45-2.49) | 0.39 (0.36-0.40) | 2.25 (2.21-2.26) | 1.4 (1.3-1.7) | 13.0 (12.8-13.1) | 6.56 (6.43-6.62) | 0.75 (0.62-0.80) | - | 8/8 |
| Stanley: Late start (10-20 m), p=1.00, planner only | 8 | 1.35 (1.31-1.46) | 0.63 (0.61-0.78) | 0.89 (0.88-0.91) | 4.0 (2.7-11.1) | 13.1 (13.0-13.2) | 1.35 (1.31-1.46) | 1.71 (1.70-1.85) | 0.23 (0.18-0.25) | 0/8 |
| Stanley: Always on, p=1.00, planner + brake | 8 | 4.18 (4.17-4.28) | 0.09 (0.08-0.09) | 4.16 (4.14-4.25) | 0.5 (0.5-0.6) | 108.0 (77.7-109.2) | 4.18 (4.17-4.28) | 0.02 (0.02-0.02) | - | 0/8 |
| Stanley: Always on, p=1.00, planner only | 8 | 2.36 (2.35-2.38) | 0.28 (0.27-0.29) | 2.25 (2.22-2.27) | 1.3 (1.0-1.5) | 13.0 (12.8-13.2) | 6.23 (6.14-6.28) | 0.09 (0.05-0.15) | - | 8/8 |

## Plan outcome

| Condition | n | Merge committed | Cancelled a merge | Stayed in lane at the car | Collided | Speed at impact, m/s | Braked | First commit, s | First cancel, s | Closest approach, m |
|---|---|---|---|---|---|---|---|---|---|---|
| MPC: No attack, planner + brake | 8 | 8/8 | 0/8 | 0/8 | 0/8 | - | 1/8 | 1.33 (1.30-1.65) | - | 3.50 (3.50-3.50) |
| Pure pursuit: No attack, planner + brake | 8 | 8/8 | 0/8 | 0/8 | 0/8 | - | 0/8 | 1.35 (1.25-1.60) | - | 3.58 (3.57-3.58) |
| Stanley: No attack, planner + brake | 8 | 8/8 | 0/8 | 0/8 | 0/8 | - | 0/8 | 1.35 (1.30-1.60) | - | 3.73 (3.73-3.74) |
| MPC: Late start (10-20 m), p=1.00, planner only | 8 | 8/8 | 3/8 | 0/8 | 0/8 | - | 0/8 | 1.30 (1.30-1.50) | 4.60 (4.50-4.70) | 3.50 (3.25-3.50) |
| MPC: Always on, p=1.00, planner + brake | 8 | 8/8 | 0/8 | 8/8 | 0/8 | - | 8/8 | 4.85 (4.75-4.90) | - | 8.44 (7.89-8.99) |
| MPC: Always on, p=1.00, planner only | 8 | 8/8 | 0/8 | 8/8 | 8/8 | 7.6 (7.6-7.6) | 0/8 | 4.85 (4.80-4.90) | - | 4.78 (4.78-4.78) |
| Pure pursuit: Late start (10-20 m), p=1.00, planner only | 8 | 8/8 | 5/8 | 0/8 | 0/8 | - | 0/8 | 1.40 (1.30-1.60) | 4.65 (4.50-5.05) | 3.51 (3.30-3.58) |
| Pure pursuit: Always on, p=1.00, planner + brake | 8 | 8/8 | 0/8 | 8/8 | 0/8 | - | 8/8 | 4.85 (4.80-4.95) | - | 8.07 (6.72-8.62) |
| Pure pursuit: Always on, p=1.00, planner only | 8 | 8/8 | 0/8 | 8/8 | 8/8 | 7.6 (7.6-7.6) | 0/8 | 4.85 (4.80-4.95) | - | 4.78 (4.78-4.78) |
| Stanley: Late start (10-20 m), p=1.00, planner only | 8 | 8/8 | 5/8 | 0/8 | 0/8 | - | 0/8 | 1.38 (1.30-1.60) | 4.65 (4.50-5.00) | 3.70 (3.44-3.74) |
| Stanley: Always on, p=1.00, planner + brake | 8 | 8/8 | 0/8 | 8/8 | 0/8 | - | 8/8 | 4.85 (4.80-4.90) | - | 8.23 (7.30-8.67) |
| Stanley: Always on, p=1.00, planner only | 8 | 8/8 | 0/8 | 8/8 | 8/8 | 7.6 (7.6-7.6) | 0/8 | 4.85 (4.80-4.90) | - | 4.78 (4.78-4.78) |

**Takeaway:** across all three controllers the plan-level outcome lines up almost
exactly (same commit times, same collision/brake pattern, same closest-approach
distances) — the attack's effect on *what the planner decides* is controller-
independent. The controllers only separate on ride smoothness in the no-attack and
late-start conditions (MPC has zero overshoot but slightly higher baseline jerk; pure
pursuit has the lowest peak lateral accel).

Values are median (min-max) over seeds. Jerk is measured from the first second to the
end of the drive, or to just before the first collision.
