# Point injection vs. a goal-directed highway-exit planner

A different attack/planner pairing from the removal scenarios: point injection (a real
harvested-vehicle LiDAR trace placed via `--inject-follow`) against a goal-directed
planner (`ExitPlanner`) that always intends to take a real Town06 highway exit unless
the exit lane looks occupied for `confirm_sweeps` consecutive sweeps, with a hard
commit deadline past which taking the exit is no longer physically possible
(`s_deadline = s_gore - shift_len - lead_m - safety_buffer_m`).

The attack places one or two phantom vehicles in the exit lane (`--inject-lateral
-3.5`, 3.5 m to the ego's right, matching the exit lane's offset) to make the lane look
occupied through the planner's commit window. 8 seeds per condition, 40 runs total
(`Sweep_ExitInjection`).

| Condition | n | Took the exit | First commit, s | Missed at, s | Mean abs jerk, m/s³ | Peak jerk, m/s³ |
|---|---|---|---|---|---|---|
| No attack | 8 | 8/8 | 2.55 (2.55-2.55) | - | 1.05 (1.05-1.06) | 13.0 (12.9-13.2) |
| One phantom, whole window | 8 | 0/8 | - | 17.18 (16.75-17.50) | 1.10 (0.81-1.93) | 13.1 (12.9-19.5) |
| Two phantoms, whole window | 8 | 0/8 | - | 16.75 (16.75-16.75) | 0.81 (0.80-0.81) | 13.1 (13.0-13.2) |
| Two phantoms, first half only | 8 | 8/8 | 10.10 (10.10-10.10) | - | 1.10 (1.09-1.10) | 13.0 (12.6-13.3) |
| Two phantoms, second half only | 8 | 8/8 | 2.55 (2.55-2.55) | - | 1.05 (1.04-1.22) | 13.1 (12.6-13.2) |

Values are median (min-max) over seeds. "Took the exit" means the planner committed
to the shift before the deadline; a missed run instead reaches the `MISSED` state at
the deadline. Jerk is measured from the first second to the end of the drive.

## What this shows

- **Reliable, but only with the right timing.** One or two phantoms held in the exit
  lane for the entire commit window make the planner miss the exit in 8/8 seeds each
  — but holding the phantom for only half the window shows the attack is timing-
  gated, not occupancy-gated:
  - **First half only** (phantom removed partway through): the planner just commits
    *late*, 8/8 seeds, at the same instant every seed — as soon as the phantom clears,
    not at the clean-run time.
  - **Second half only** (phantom added partway through, after the clean-run commit
    time of 2.55 s): 8/8 seeds still took the exit at the clean-run time, because the
    planner had already committed before the attack started. There is no
    "un-commit" — once `ExitPlanner` shifts into the exit lane, a phantom appearing
    behind that decision changes nothing.
- **The first attack in this project where success makes the drive smoother, not
  rougher.** Missing the exit means the ego just keeps cruising — no swerve, no stop
  — so mean jerk stays flat (~1.0-1.1 m/s³) across every condition, including the
  missed-exit ones. Jerk cannot detect this attack; only the plan state (`MISSED` vs.
  a committed merge) reveals it.
- **Detector-reliability caveat:** the official 8-seed batch above (phantom stand-off
  `--inject-distance 12`) shows both 1- and 2-phantom conditions missing 8/8,
  reliably. An earlier, informal single-phantom test at a larger stand-off
  (`--inject-distance 15`) saw CenterPoint's detection of the phantom drop out for
  ~5 consecutive sweeps — long enough to pass the 3-sweep clear-check and let the
  planner commit anyway. One-phantom reliability at different stand-off distances is
  not yet fully settled; it is reliable in the batch actually run, but the exploratory
  result above is a genuine attacker-reliability finding worth more seeds.
- A stray emergency-brake event appeared once (1 of 8 "second-half-only" seeds, around
  t=10.15 s) after the ego had already committed and shifted lanes; it didn't change
  the outcome (that run still took the exit) and is unexplained — likely a one-off
  detector mis-localization of the phantom overlapping the already-shifted path
  corridor, not a real hazard.

## Limits

- One exit, one scene, one detector, 8 seeds per condition.
- The attacker is idealized: it knows exactly where the exit lane is and keeps the
  phantom glued to it. A real attacker would need some way to infer that this specific
  lane is the one the victim's planner is deciding whether to take.
- No collision risk by construction — missing an exit is a real safety/traffic
  problem, but it was not modeled as a crash in this scenario.
- No fine timing sweep yet, only a coarse early/late half split — the exact margin
  (how late the phantom can start and still prevent the merge) is unmeasured.
