# Improving PICO Teleop Tracking via Training

You can improve how well the policy tracks your PICO input in two main ways: **reward tuning** (fastest) and **recorded data + residual/fine-tune** (more work, can match deployment better).

---

## Option 1: Reward Tuning (try this first)

The policy was trained to track reference motion; if it under-tracks (e.g. small or delayed motion), the reward balance is likely favoring safety/regularization over tracking.

### Where to edit

**File:** `legged_gym/legged_gym/envs/g1/g1_mimic_future_config.py`  
**Class:** `G1MimicStuFutureCfg.rewards.scales`

### Concrete changes

1. **Increase tracking vs regularization**
   - Bump **tracking** scales (policy cares more about matching the target):
     - `tracking_joint_dof`: try `2.5` or `3.0` (default `2.0`)
     - `tracking_keybody_pos` / `tracking_keybody_pos_global`: try `2.5`–`3.0`
     - `tracking_root_linear_vel`, `tracking_root_angular_vel`: try `1.2`–`1.5`
   - Slightly **reduce** regularization so the policy can make larger corrections:
     - `action_rate`: try `-0.02` or `-0.03` (default `-0.05`)
     - `dof_acc`: keep or slightly reduce magnitude

2. **Re-enable XY root tracking** (if you want better following in the plane)
   - In `rewards.scales`, uncomment and set:
     - `tracking_root_translation_xy = 1.0` (or `0.5`–`1.5`)
   - Right now it’s commented out; enabling it encourages the base to follow XY motion.

3. **Emphasize arm joints in the tracking reward**
   - In **`legged_gym/legged_gym/envs/g1/g1_mimic_distill_config.py`** (or override in `g1_mimic_future_config.py`), the env uses `dof_err_w` (one weight per DOF) in `_reward_tracking_joint_dof` and `_reward_tracking_joint_vel`.
   - G1 order: left leg (6), right leg (6), waist (3), left arm (7), right arm (7) = 29.
   - To stress arms (indices 18–24 left, 25–31 right), override `env.dof_err_w` in `G1MimicStuFutureCfg` with e.g. arms at `1.5` or `2.0` and legs/waist at `1.0`.

4. **Motion data**
   - Add or upweight motions in `motion_data_configs/twist2_dataset.yaml` that have **larger steps** and **faster arm motion** so the policy sees targets closer to what you do on PICO.

After changing rewards, **re-train** (or resume) and re-test on PICO; iterate on scales as needed.

---

## Option 2: Record Data + Residual Policy or Fine-Tune

If reward tuning is not enough, you can better match the **deployment distribution** (PICO retargeting + your choice of lower body) by using recorded teleop.

### What you have today

- **`deploy_real/server_data_record.py`** records from Redis (e.g. vision, body/hand state, body/hand action) into episodes. So you can record **(proprio, mimic_obs, action)** when running teleop (sim2sim or sim2real with logging).
- There is **no** built-in residual policy or “merge two policies” in this repo.

### Path A: Fine-tune the existing policy on recorded data

1. **Record teleop**
   - Run teleop (PICO → GMR → Redis; keep your intentional “stationary lower body + PICO arms” if you want).
   - Record **state_body** (or equivalent proprio) and **action_body** (mimic target) and the **actual action** sent to the robot (output of the current policy). So you need either:
     - Log the policy output (action) in the low-level server and align with Redis state/action_body by timestamp, or
     - Run in sim, record (obs, action) from the sim runner.
   - Save as (obs, action) pairs or episodes.

2. **Add a BC / imitation loss to training**
   - In the PPO/DAgger runner, add an optional loss: when you have recorded (obs, action), add `alpha_bc * BC_loss(policy(obs), action)` to the PPO loss (e.g. `alpha_bc` small so PPO still dominates).
   - Or: do a **second stage**: freeze most of the policy and only fine-tune the last layer(s) on recorded (obs, action) with MSE or BC loss.

3. **Train**
   - Either continue training the current policy with the combined loss, or run a short fine-tune from the current checkpoint on recorded data only.

This way the policy sees “PICO-like” targets and your chosen action distribution, so tracking can improve without changing the deployment pipeline.

### Path B: Residual policy and merge

1. **Record teleop**
   - Same as above: get (obs, action_from_current_policy, desired_or_actual_action). “Desired” could be from a human driving the robot (e.g. joystick) or from a better reference (e.g. GMR output converted to joint targets).

2. **Train a small residual**
   - Define a small network `residual(obs)` that outputs a **delta** in action space.
   - Train it so that `base_policy(obs) + residual(obs) ≈ desired_action` (or so that applying the residual improves tracking in your reward).
   - Data: (obs, desired_action - base_policy(obs)).

3. **Deploy**
   - At inference: `action = base_policy(obs) + residual(obs)` (optionally clamp or scale the residual).

You’d add a small training script (e.g. PyTorch) that loads the base policy, builds the residual net, and trains on the recorded deltas; no change to the main PPO code unless you want to train the residual with RL later.

---

## Recommendation

- **Start with Option 1 (reward tuning):** change `g1_mimic_future_config.py` rewards (and optionally `dof_err_w` and motion data), re-train, and test on PICO. It’s the fastest and often enough.
- **If tracking is still off:** use Option 2 – record teleop, then either **fine-tune** the current policy on that data (Path A) or add a **residual** and merge at deploy (Path B). Path A reuses the existing training stack with an extra loss or a short fine-tune; Path B keeps the base policy fixed and adds a small correction net.
