# Training a Residual Policy on Top of `twist2_1017_25k.onnx`

You can use **`assets/ckpts/twist2_1017_25k.onnx`** as the fixed base policy and train a **residual network** so that at deployment:

**action = π_base(obs) + r(obs)**

The base stays frozen; only the residual is trained.

---

## 1. What You Need

| Item | Purpose |
|------|--------|
| **Base policy** | `assets/ckpts/twist2_1017_25k.onnx` — load with `onnxruntime`, input shape `(batch, 1402)`, output `(batch, 29)` |
| **Observation format** | Same as in `server_low_level_g1_real.py`: `obs = [current_mimic_obs + proprio_history + future_mimic_obs]`, total **1402** dims |
| **Action space** | **29** DOF (G1 body); residual also outputs 29 dims (delta) |
| **Residual data** | Pairs **(obs, a_desired)**. `a_desired` = desired action (e.g. from teleop / reference / better tracker). Then **delta = a_desired − π_base(obs)** is the residual target. |

---

## 2. Pipeline Overview

```
1. Load base ONNX (twist2_1017_25k.onnx) and keep it fixed.
2. Build dataset: for each (obs, a_desired), compute a_base = π_base(obs), delta = a_desired − a_base.
3. Train residual r(obs) to predict delta (e.g. MSE or BC).
4. Export r to ONNX (optional).
5. Deploy: action = π_base(obs) + r(obs); optionally clamp or scale the residual.
```

You **do** need the trained base (the ONNX) to train the residual: the residual target is **delta = a_desired − π_base(obs)**.

---

## 3. Observation and Action Specs (Match Deploy)

From `server_low_level_g1_real.py`:

- **Obs dim:** `total_obs_size = 1402`
  - `n_obs_single = 127` (35 mimic + 92 proprio)
  - `history_len = 10`
  - Layout: `[current_127, history_127*10, future_mimic_35]` → 127×11 + 35 = 1402
- **Action dim:** 29 (same scaling as deploy: `target_dof_pos = default_dof_pos + action * action_scale`)

Your residual dataset must use the **same** obs construction (and normalization, if any) as in deployment so that base and residual see the same input.

---

## 4. Data: How to Get (obs, a_desired)

You need **(observation, desired action)** pairs in the same format as the real controller.

**Option A – Teleop recording**

- Run sim2sim or sim2real with teleop (PICO → GMR → Redis).
- In the low-level server (or a logger), at each step record:
  - **obs**: the same 1402-d vector you pass to the ONNX (current + history + future mimic).
  - **a_desired**: e.g. the joint targets that correspond to “what the operator wants.” For example:
    - Use the **mimic_obs** (35D) and convert the last 29 dims (dof_pos) to action space: `a_desired = (dof_pos_target - default_dof_pos) / action_scale`, or
    - If you log “target joint position” from GMR/retargeting, convert that to the same action space (offset and scale as in deploy).
- So each sample is **(obs_1402, a_desired_29)**.

**Option B – Reference motions in sim**

- In Isaac Gym (or off-line), roll out with the **same obs construction** as deploy.
- For each step, take **a_desired** from the reference motion (e.g. ref dof_pos → action space as above).
- Record **(obs, a_desired)**. This matches “improve tracking of this motion set” and reuses your motion data.

**Option C – Human-in-the-loop corrections**

- Run the robot (or sim) with **action = π_base(obs)** and let a human apply corrections (e.g. joystick).
- Record **(obs, a_commanded)** where `a_commanded = π_base(obs) + human_correction` (or the actual commanded action). Then **a_desired = a_commanded**; delta = a_commanded − π_base(obs).

For all options, **a_desired** must be in the **same action space** as the base policy (same scale and offset as in `server_low_level_g1_real.py`).

---

## 5. Residual Network

- **Input:** same as base, e.g. **(batch, 1402)**.
- **Output:** **(batch, 29)** (the delta).
- **Architecture:** e.g. MLP: 1402 → 256 → 256 → 128 → 29 (ReLU, no activation on last layer). Keep it smaller than the base so it only captures corrections.
- **Training target:** for each batch (obs, a_desired), compute:
  - `a_base = base_policy(obs)`  (run ONNX in numpy/onnxruntime or torch with a wrapper)
  - `delta = a_desired - a_base`
  - Loss = MSE(r(obs), delta) (or L1, Huber, etc.).

You can optionally **clip or scale** delta (e.g. cap per dimension) before training so the residual doesn’t learn huge corrections; same for at deploy.

---

## 6. Loading the Base ONNX in Python

Reuse the same pattern as in `deploy_real/server_low_level_g1_real.py`:

```python
import onnxruntime as ort
import numpy as np

session = ort.InferenceSession("assets/ckpts/twist2_1017_25k.onnx", providers=["CPUExecutionProvider", "CUDAExecutionProvider"])
input_name = session.get_inputs()[0].name

def base_policy(obs):
    # obs: (N, 1402) float32
    out = session.run(None, {input_name: obs})[0]
    return out  # (N, 29)
```

For training, you can either:

- Call this in a loop over batches (no gradient through base), or
- Export the base to PyTorch and use it in the same graph (only for convenience; base params stay frozen). Easiest is to keep ONNX and use numpy/ORT in the training loop.

---

## 7. Training Loop (Pseudocode)

```text
base = load_base_onnx("assets/ckpts/twist2_1017_25k.onnx")
residual = ResidualMLP(obs_dim=1402, action_dim=29)
optimizer = Adam(residual.parameters(), lr=1e-3)

for epoch in range(epochs):
    for batch_obs, batch_a_desired in dataloader:
        with torch.no_grad():
            a_base = base(batch_obs)   # or onnxruntime run
        delta_target = batch_a_desired - a_base
        delta_pred = residual(batch_obs)
        loss = F.mse_loss(delta_pred, delta_target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

Normalize obs the same way as in deploy (e.g. same normalizer or none); if deploy uses a normalizer, use the same one when building the dataset and when running base + residual at deploy.

---

## 8. Deployment: Using Base + Residual

- **Option A – Two ONNX models**  
  - Load `twist2_1017_25k.onnx` and `residual.onnx`.  
  - Each step: `action = base_run(obs) + residual_run(obs)`.  
  - Slight overhead (two runs); no change to base checkpoint.

- **Option B – Merge into one ONNX**  
  - Build a graph: `action = base(obs) + residual(obs)`, then export that graph to ONNX so you have a single “base+residual” model.  
  - One inference call; requires wiring base and residual in one framework (e.g. PyTorch) then exporting.

- **Option C – Fuse in Python (sim2real server)**  
  - In `server_low_level_g1_real.py`, load both base and residual (e.g. both ONNX).  
  - Replace `raw_action = self.policy(obs_tensor)` with:
    - `a_base = self.base_policy(obs_tensor)`
    - `a_residual = self.residual_policy(obs_tensor)`
    - `raw_action = a_base + a_residual` (and optionally clamp).

---

## 9. Checklist

- [ ] Base ONNX path: `assets/ckpts/twist2_1017_25k.onnx`
- [ ] Obs dim 1402, action dim 29; obs layout matches deploy.
- [ ] Dataset: (obs, a_desired) with a_desired in same action space as deploy.
- [ ] Residual target: delta = a_desired − π_base(obs); train r(obs) → delta.
- [ ] Deploy: action = π_base(obs) + r(obs); optional clamp/scale on residual.

Using your existing ONNX as the fixed base and training only the residual is the right way to “improve” the policy with a residual technique; the ONNX you have is exactly what you need for that.
