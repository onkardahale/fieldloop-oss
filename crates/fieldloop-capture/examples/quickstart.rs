use fieldloop_capture::Capture;
use fieldloop_types::{EpisodeId, PolicyVersion, RobotId, RobotIdentity, TenantId};

fn main() {
    let robot = RobotIdentity::new(TenantId::new("tenant-a"), RobotId::new("robot-1"));
    let (capture, drain) = Capture::new(robot, 1024);

    let ctx = capture.register_context(
        PolicyVersion::new("pick@v1+abc123abc123"),
        "sha256:model-hash",
        "warehouse_pick_arm",
        "pick-can",
    );

    let rollout_id = capture.log_step(EpisodeId::new(), 0, ctx, 180);
    let rollouts = drain.drain_available();

    println!("rollout_id = {rollout_id}");
    println!("drained = {}", rollouts.len());
    println!("dropped = {}", capture.dropped());
}
