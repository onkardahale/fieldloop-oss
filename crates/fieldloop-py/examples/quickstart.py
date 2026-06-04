import uuid

import fieldloop


cap = fieldloop.Capture("tenant-a", "robot-1", capacity=1024)

ctx = cap.register_context(
    policy_version="pick@v1+abc123abc123",
    model_hash="sha256:model-hash",
    embodiment="warehouse_pick_arm",
    task_id="pick-can",
)

episode_id = str(uuid.uuid4())
rollout_id = cap.log_step(
    episode_id,
    step_index=0,
    ctx=ctx,
    inference_us=180,
)

rows = cap.drain()

print("rollout_id:", rollout_id)
print("drained:", len(rows))
print("dropped:", cap.dropped())
