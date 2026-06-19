# Hierarchical Subgoal and MPC Planning Ideas

This note records two related control ideas for the UAV predator-prey project:

1. a hierarchical subgoal RL controller,
2. a learned dynamics/reward model with MPC-style action-sequence planning.

It is a design sketch and learning target, not a claim that either architecture
is novel.

## Idea 1: Hierarchical Subgoal RL

Use a high-level policy to choose a compact intermediate goal, then use a
low-level policy to turn the current observation and that subgoal into actions.

```text
final task goal g
current state/observation s_t

high-level actor:
    pi_high(s_t, g) -> z_t

low-level actor:
    pi_low(s_t, z_t) -> a_t
```

The subgoal `z_t` should not be a full future observation. A full observation
contains many quantities that the agent cannot directly control or predict
reliably. The subgoal should be compact, physically meaningful, and easy to
measure.

For the first version:

```text
z_t = relative target position
```

Later extensions:

```text
z_t = relative target position + desired velocity
z_t = relative target position + desired velocity + desired yaw/heading
```

## Four-Network Structure

### 1. High-Level Actor / Manager

```text
input:  s_t, g
output: z_t

pi_high(s_t, g) -> z_t
```

This network chooses the intermediate subgoal. In predator-prey, this could be
an intercept waypoint for a predator or an escape waypoint for the prey.

### 2. High-Level Critic

```text
input:  s_t, z_t, g
output: scalar value

Q_high(s_t, z_t, g) -> value
```

This critic evaluates whether the chosen subgoal helps with the real task, such
as catching the prey or surviving.

### 3. Low-Level Actor / Worker

```text
input:  s_t, z_t
output: a_t

pi_low(s_t, z_t) -> a_t
```

This network tries to realize the current subgoal using environment actions.
For a UAV, `a_t` could be a velocity command, acceleration command, body-rate
command, or motor/thrust command depending on the environment interface.

### 4. Low-Level Critic

```text
input:  s_t, a_t, z_t
output: scalar value

Q_low(s_t, a_t, z_t) -> value
```

This critic evaluates whether a low-level action helps reach the current
subgoal.

## Horizon

The manager should update the subgoal every fixed number of low-level control
steps at first.

```text
every k environment steps:
    z_t = pi_high(s_t, g)

every environment step:
    a_t = pi_low(s_t, z_t)
```

Start with a fixed `k` instead of learning it. This keeps debugging clean:

- if the system fails, first check whether the subgoal is bad or the worker
  cannot reach it,
- only after fixed horizons work should the horizon become an input or output,
- candidate values can be compared as an ablation.

Example:

```text
control rate: 50 Hz
k = 25  -> 0.5 s horizon
k = 50  -> 1.0 s horizon
k = 100 -> 2.0 s horizon
```

If the horizon is learned later, one possible form is:

```text
pi_high(s_t, g) -> z_t, k
pi_low(s_t, z_t, k) -> a_t
```

## Predator-Prey Interpretation

### Predator

Possible final goal:

```text
g = prey relative position and velocity
g = [p_prey - p_predator, v_prey - v_predator]
```

Possible subgoal:

```text
z_t = local intercept waypoint
z_t = predicted prey position after k steps
z_t = blocking position around the prey
```

For a team of predators, the subgoal may also encode role-like behavior:

```text
z_t = direct chase waypoint
z_t = left/right blocking waypoint
z_t = cut-off waypoint near arena boundary or obstacle
```

### Prey

Possible final goal:

```text
g = predator relative positions and velocities
```

The prey does not usually have a single target point. Its task objective is
defined by survival, distance from predators, staying in bounds, and avoiding
obstacles.

Possible subgoal:

```text
z_t = escape waypoint
z_t = safe point away from predators
z_t = waypoint that increases capture time
```

## Rewards

The low-level worker can receive an intrinsic reward for reaching the subgoal:

```text
r_low = - distance(current_position, z_t)
```

or, if velocity is included:

```text
r_low = - position_error - velocity_error
```

The high-level manager should receive task-level reward:

```text
r_high = environment reward over the next k steps
```

This separation matters. The worker needs dense feedback for subgoal tracking,
while the manager must learn which subgoals solve the real predator-prey task.

## Relation to Existing RL Ideas

The broad idea is not new. It is related to:

- hierarchical reinforcement learning,
- options,
- manager-worker architectures,
- Feudal RL,
- goal-conditioned RL,
- HIRO-style off-policy hierarchical RL,
- robotics waypoint planning plus low-level control.

The useful project contribution would not be "inventing hierarchy" in general.
The useful contribution could be:

- a good subgoal representation for UAV predator-prey,
- robust horizon selection,
- clean separation between manager and worker rewards,
- multi-predator coordination through subgoals,
- integration with UAV dynamics and safety constraints,
- ablations showing when hierarchy beats a flat MAPPO baseline.

## First Minimal Experiment

Before implementing this in the main training stack, test the smallest version:

```text
fixed horizon: k
subgoal: relative 2D or 3D position only
worker input: s_t, z_t
worker output: action
worker reward: negative distance to z_t
manager: scripted subgoal first, learned manager later
```

Suggested order:

1. Train or test the worker with scripted reachable subgoals.
2. Verify that the worker reaches subgoals without severe overshoot.
3. Add velocity to the subgoal only if position-only tracking is unstable.
4. Add a learned high-level manager after the worker is reliable.
5. Compare fixed horizons `k` as an ablation.
6. Compare against the flat MAPPO baseline.

The main diagnostic question is:

```text
Can the worker reliably reach compact, physically meaningful subgoals?
```

If the answer is no, a learned manager will not fix the architecture. The
low-level control problem has to be solved first.

## Idea 2: Learned Dynamics/Reward MPC

The second idea is to learn how the current world reacts to actions, then use
sampling-based planning to choose the next action.

The first thought was:

```text
R_hat(s_t, a_t) -> predicted reward
```

This is not enough for multi-step planning by itself. To score a sequence of 10
future actions, the planner also needs the predicted next state after each
action:

```text
M(s_t, a_t) -> s_{t+1}, r_t
```

Then a sampled action sequence can be rolled forward inside the learned model:

```text
s = current_obs
total = 0

for a in action_sequence:
    s_next, r = M(s, a)
    total += r
    s = s_next

execute the first action of the best sequence
```

This is an MPC pattern:

```text
plan over a short horizon
execute only the first action
observe the real next state
replan
```

## MPC Sampling Structure

Simple random shooting:

```text
sample 256 action sequences
sequence length = 10
evaluate all sequences with M(s, a)
choose the sequence with the highest predicted reward sum
execute only the first action
```

Better sampling methods:

- Cross-Entropy Method (CEM): sample sequences, keep the top fraction, refit the
  mean and standard deviation, repeat.
- MPPI: weight action sequences by predicted score instead of keeping only the
  top samples.
- Policy-guided sampling: use a policy network to suggest a good action or
  action sequence, then sample around it.
- Warm starting: shift the previous best sequence forward by one step and
  sample around that plan.

Warm starting is especially important:

```text
previous best: [a_t, a_{t+1}, ..., a_{t+9}]
next guess:    [a_{t+1}, ..., a_{t+9}, new_action]
```

## Reward Model vs Return Model

For the MPC-style idea, predicting reward is the cleaner first version:

```text
M(s, a) -> s_next, r
```

The planner itself sums the predicted rewards over the horizon.

Learning return instead would produce a Q-function:

```text
Q(s_t, a_t) = expected future return after taking action a_t
```

That is closer to actor-critic methods such as DDPG, TD3, and SAC. It can score
single actions directly:

```text
sample 256 actions
score each action with Q(s_t, a)
choose argmax action
```

But for continuous UAV actions, searching over actions with random sampling can
be inefficient. Actor-critic methods usually add an actor:

```text
pi(s_t) -> action that approximately maximizes Q(s_t, a)
```

So the distinction is:

```text
R_hat(s,a) + f_hat(s,a): planning / MPC route
Q_hat(s,a):              critic / return route
pi(s):                   fast policy route
```

## Terminal Value Extension

A short MPC horizon can be too myopic. A useful extension is a terminal value
network:

```text
V(s_t) -> expected future return from state s_t
```

Then the planner score becomes:

```text
score = r_0 + gamma r_1 + ... + gamma^9 r_9 + gamma^10 V(s_10)
```

This combines short-horizon model-based planning with a learned long-term value
estimate.

## Main Risks of Learned-Model MPC

The main risk is model exploitation: the planner may find action sequences that
look good inside the learned model but fail in the real simulator or real world.

Mitigations:

- keep the planning horizon short,
- replan every step,
- compare predicted rollouts against real rollouts,
- train on data generated by the current planner,
- use ensembles to estimate uncertainty,
- penalize uncertain model predictions,
- keep action and state constraints explicit.

## How the Two Ideas Connect

The two ideas can be combined later:

```text
manager:
    pi_high(s_t, g) -> subgoal z_t

worker/planner:
    MPC uses M(s, a) -> s_next, r
    reward includes distance to z_t
    execute first action of best sequence
```

In this combined version, the high-level policy chooses where the agent should
go, while MPC handles local control toward the subgoal.

This mirrors classical robotics:

```text
strategic planner -> waypoint/subgoal -> local MPC/controller -> action
```

For the project, the simplest comparison would be:

```text
flat MAPPO/PPO baseline
hierarchical subgoal RL
learned-model MPC
hierarchical subgoal + MPC worker
```

The key experimental question is not which idea sounds more powerful. The key
question is which decomposition makes the UAV predator-prey task easier to
learn, debug, and explain.
