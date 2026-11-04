# Capstone evaluation: closed-loop ATLAS attacks in CARLA

Results from a separate, closed-loop evaluation of ATLAS-style LiDAR removal and point
injection against a CenterPoint (nuScenes weights) detector feeding a lane-change /
highway-exit planner and an emergency brake, run in CARLA 0.9.14 (Town06). This is not
the `eval/` harness in this repo (no PHiM, no OpenPCDet matrix) — it is a from-scratch
sim-in-the-loop setup built around the same attack primitives (sector removal, trace
injection), used to see what the attack does to a *planner's decisions*, not just to
detector mAP/ASR.

8 seeds per condition throughout, CARLA's deterministic RNG seeded per run.

## Files

| | |
|---|---|
| [01_removal_attacks.md](01_removal_attacks.md) | A-HFR sector removal against a parked car, a moving car, a standing pedestrian, a standing cyclist, and a moving cyclist. Planner = reactive lane-change; brake = emergency AEB. |
| [02_controllers.md](02_controllers.md) | Same removal attack (parked-car scene) run through three different steering controllers (Stanley, pure pursuit, linear MPC) to check whether the controller changes the attack's effect. |
| [03_exit_injection.md](03_exit_injection.md) | Point injection (phantom vehicle trace) against a goal-directed highway-exit planner, instead of removal against a reactive one. |

## Scenario summary

- **Removal scenarios** (`01`, `02`): a target (parked car / moving car / pedestrian /
  cyclist) sits in or near the ego's target lane. The reactive planner only merges to
  avoid it; A-HFR sector removal deletes LiDAR returns in the target's direction so the
  detector drops it, and the planner may merge into / through it instead of around it.
  "Wedge tracks elevation" = the removal sector's elevation is re-aimed at the target
  every sweep, instead of being fixed at the first active frame; this beats the
  emergency brake outright on stationary targets.
- **Controller scenario** (`02`): the parked-car removal attack repeated with three
  different lateral controllers. The attack's effect on the plan is controller-
  independent; only ride smoothness changes.
- **Exit-injection scenario** (`03`): a different kind of target and planner. There is
  no spoofed/removed object near the ego's lane — instead a goal-directed planner
  always intends to take a real highway exit unless the exit lane looks occupied. A
  phantom vehicle (a real harvested LiDAR trace, point-injected rather than removed) is
  placed in the exit lane to make the planner think it's occupied and miss the exit.

## Caveats that apply to all of the above

- Single scene per scenario (Town06), single detector (CenterPoint, nuScenes weights),
  idealized attacker (exact target/lane geometry, no sensing noise on the attacker's
  side).
- CenterPoint's own reliability on CARLA-rendered pedestrians is poor even with no
  attack (see `01_removal_attacks.md`) — a confound to read the pedestrian numbers
  against, not an attack effect.
- Jerk/smoothness metrics use a Savitzky-Golay-smoothed 3rd derivative of position
  (window 11, order 3) over the whole drive (or up to first collision).
