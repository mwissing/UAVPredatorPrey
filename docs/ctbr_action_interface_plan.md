# Plan: Collective Thrust + Body Rates (CTBR) Action Interface

Status: proposal (2026-10-07), nothing implemented yet. Decision: the policies
should use an interface that real flight controllers accept, so that a later
deployment on real quadrotors is possible.

## Why change the interface

Real multicopter flight controllers are cascaded. In PX4: position ->
velocity -> acceleration/thrust vector -> attitude -> **body rates** ->
control allocation (mixer) -> motors. The rate loop runs on the flight
controller at hundreds of Hz up to about 1 kHz with gyro feedback. External
(offboard) software can enter the cascade at several levels; the lowest
standard one is **collective thrust + body-rate setpoints**
(`VehicleRatesSetpoint` over ROS 2, or MAVLink `SET_ATTITUDE_TARGET` with body
rates). The Crazyflie firmware likewise runs an attitude-rate PID on board, and
racing firmware such as Betaflight is natively rate-controlled ("acro").

Our policy currently outputs collective thrust plus body **moments**
(torques), applied directly as a wrench at the 50 Hz policy rate. The policy
therefore is the rate controller: a 50 Hz neural inner loop that has never
seen motor lag, battery sag, or inertia errors. Two consequences:

- **Fragile transfer.** The fastest loop is the one most sensitive to model
  errors, and it is learned against an idealized model.
- **Unrealistic agility.** `moment_scale = 0.01` N m per action unit is about
  twice a real Crazyflie's roll/pitch torque limit (about 0.005 N m) and ten
  times its yaw limit, with inertia of only about 1.4e-5 kg m^2. The current
  predators are tilted beyond 45 deg about 31% of the time.

With CTBR the policy decides how hard to push and how fast to rotate at
50 Hz, and a simple, well-understood rate loop handles the fast dynamics: the
classic time-scale separation of cascaded control. This is the interface used
by OPEN (Chen et al. 2024) and Swift (Kaufmann et al. 2023), and the one
compared in Kaufmann, Bauersfeld, and Scaramuzza's benchmark of control
abstractions (2022).

## Interface

Action per drone (4D, in [-1, 1] after the tanh squashing, as today):

| component | mapping |
|---|---|
| a0: collective thrust | `T = T/W * m * g * (a0 + 1) / 2` (unchanged) |
| a1, a2: roll and pitch rate setpoints | `w_sp = a * w_max_rp` |
| a3: yaw rate setpoint | `w_sp = a * w_max_yaw` |

- The thrust-to-weight asymmetry stays (prey 2.2, predators 1.9). `w_max_rp`
  and `w_max_yaw` are new agility parameters, equal for both roles at first
  (proposal: 10 rad/s roll/pitch, 3 rad/s yaw; to be checked against
  Crazyflie limits).
- Observations are unchanged; body rates are already observed.

## Inner loop in simulation

It runs in `_apply_action`, which executes at every physics step (100 Hz
today, decimation 2):

```text
tau = J * Kp * (w_sp - w) + w x (J w)       # inertia-scaled P control plus gyroscopic compensation
tau = clamp(tau, -tau_max, tau_max)          # realistic per-axis torque limits
```

- `J` and the mass are read from the `cf2x` USD at startup, not hard-coded.
- `Kp` sets the closed-loop rate time constant `1/Kp`. At 100 Hz the discrete
  loop is well damped for `Kp * dt <= 0.5`, so `Kp <= 50 1/s` (time constant
  >= 20 ms, comparable to small-quad motor lag). Raising physics to 200 Hz
  (decimation 4) allows a faster loop at about 1.5-2x simulation cost.
- Version 1 has no integral term. Optional and randomizable for robustness:
  first-order motor lag on thrust and torque, and gain or inertia scale
  errors.
- Later (deployment track): a rotor-level model with shared motor saturation
  (`isaaclab_contrib` Thruster/Multirotor), so that thrust and torque compete
  for the same motor budget as on a real drone.

## Validation before any RL

1. CPU unit tests of the controller math: step-response time constant within
   10% of `1/Kp`, zero steady-state error without disturbance, correct
   clamping, and the gyroscopic term cancelling `w x Jw`.
2. Isaac smoke without learning: a scripted outer loop (attitude hold
   producing rate setpoints) must hover and track rate steps without drift,
   OOB, or oscillation. Log the rate-tracking error.
3. The existing correctness gate and the bit-exact archived smoke on the old
   interface. The new interface is an opt-in configuration, so archived
   results remain reproducible.

## Migrating the policies

Every existing checkpoint outputs torques and cannot be reused directly.

- **Option A, from scratch** with the current curriculum: the cleanest, but
  the costliest. Run a short pilot first to estimate the time to
  anchor-level play.
- **Option B, warm start by behavior cloning** (recommended pilot): run
  anchor v2 in the old environment and record each observation together with
  the thrust and the body rates the old policy actually produced shortly
  afterwards. Pretrain the CTBR actor to output them, then fine-tune with RL
  (predator and prey separately, as in today's experiments). This keeps the
  skill already learned and reduces the switch to an adaptation problem.
  Expected caveat: the slower, torque-limited inner loop cannot reproduce the
  old, unrealistically fast rotations exactly, so some performance loss
  before fine-tuning is expected and is itself a measure of the realism gap.

## Sequence and effort

1. Controller and unit tests (no GPU).
2. Isaac hover/step smoke with the scripted outer loop.
3. Behavior-cloning pilot (a few GPU hours) or a scratch pilot, compared on
   the league protocol.
4. A new self-play lineage on CTBR toward anchor v3. This is a long run and
   needs approval.

## Open questions

- Target platform: Crazyflie (matching the simulated `cf2x` and the
  Crazyflow/JAX path) or a larger PX4 quadrotor? This sets mass, inertia,
  thrust-to-weight, rate limits, and loop gains.
- Physics rate: keep 100 Hz, or move to 200 Hz for a faster inner loop?
- Motor-level allocation now, or only in the deployment track?
