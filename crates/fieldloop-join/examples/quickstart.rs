//! Fieldloop quickstart: the field-learning JOIN, end to end, with no database.
//!
//! Run it: `cargo run -p fieldloop-join --example quickstart`
//!
//! It builds three policy rollouts (decisions a robot made), three real-world signals that
//! arrived later (a takeover, a collision, and silence), runs the attribution cascade, and prints
//! which outcome bound to which decision — by what method and at what confidence. This is the open
//! workbench's core value (local decision→outcome attribution) on in-memory data.

use fieldloop_config::Config;
use fieldloop_join::{AttributeOptions, Heartbeat, attribute};
use fieldloop_types::{
    BootId, BoundedBlob, EpisodeId, Feedback, FeedbackTarget, FeedbackValue, MonoClock,
    OutcomeEvent, OutcomeKind, PayloadRef, PolicyVersion, RobotId, RobotIdentity, Rollout,
    RolloutId, TenantId,
};

const MS: u64 = 1_000_000; // nanoseconds per millisecond

fn rollout(robot: &RobotIdentity, boot: BootId, step: u32, t_ns: u64) -> Rollout {
    Rollout::new(
        robot.clone(),
        EpisodeId::new(),
        step,
        MonoClock::new(boot, t_ns, t_ns as i64),
        PolicyVersion::new("pick@v1.2.0"),
        "sha256:demo".to_string(),
        "ur5e".to_string(), // matches the embodiment whose attribution windows Config::example declares
        "bin_pick".to_string(),
        PayloadRef::none(),
        PayloadRef::none(),
        BoundedBlob::empty(),
        0,
    )
}

fn main() {
    // The config declares, per robot type, how far back each outcome kind may bind. The bundled
    // example covers `ur5e` with a 250ms (clock-safe) collision window and a 5s takeover window.
    let config = Config::example();
    let robot = RobotIdentity::new(TenantId::new("acme-warehouse"), RobotId::new("arm-01"));
    let boot = BootId::new();

    // Three decisions, 2s apart.
    let (t0, t1, t2) = (1_000 * MS, 3_000 * MS, 5_000 * MS);
    let r_takeover = rollout(&robot, boot, 0, t0);
    let r_collision = rollout(&robot, boot, 1, t1);
    let r_clean = rollout(&robot, boot, 2, t2);
    let rollouts = vec![r_takeover.clone(), r_collision.clone(), r_clean.clone()];

    // The outcomes that arrived later, with no rollout id attached — the cascade reconstructs the link.
    let outcomes = vec![
        // A human took over 300ms after the first decision -> binds temporally (within 5s).
        OutcomeEvent::new(
            robot.clone(),
            MonoClock::new(boot, t0 + 300 * MS, (t0 + 300 * MS) as i64),
            OutcomeKind::TeleopTakeover,
            BoundedBlob::empty(),
        ),
        // A collision 100ms after the second decision -> binds (within the 250ms clock-safe window).
        OutcomeEvent::new(
            robot.clone(),
            MonoClock::new(boot, t1 + 100 * MS, (t1 + 100 * MS) as i64),
            OutcomeKind::Collision,
            BoundedBlob::empty(),
        ),
    ];

    // Heartbeats blanket the third decision's window with no adverse outcome -> a synthesized
    // "nothing happened here" success (the absence is itself evidence the run went fine).
    let mut heartbeats = Vec::new();
    let mut t = t2;
    while t <= t2 + 2_000 * MS {
        heartbeats.push(Heartbeat {
            robot: robot.clone(),
            clock: MonoClock::new(boot, t, t as i64),
        });
        t += 400 * MS;
    }

    let opts = AttributeOptions {
        heartbeat_period_ns: Some(500 * MS),
        ..Default::default()
    };

    let feedback = attribute(&config, &rollouts, &outcomes, &heartbeats, &opts);

    // Render each binding: which decision an outcome attached to, how, and how sure.
    let label = |id: RolloutId| -> &'static str {
        if id == r_takeover.id {
            "decision#0 (takeover expected)"
        } else if id == r_collision.id {
            "decision#1 (collision expected)"
        } else if id == r_clean.id {
            "decision#2 (clean expected)"
        } else {
            "unknown"
        }
    };
    let value = |fb: &Feedback| -> String {
        match &fb.value {
            FeedbackValue::Boolean { value } => {
                if *value {
                    "success".into()
                } else {
                    "FAILURE".into()
                }
            }
            FeedbackValue::FailureClass { class } => format!("FAILURE ({class:?})"),
            other => format!("{other:?}"),
        }
    };

    println!(
        "Fieldloop quickstart — {} rollouts in, {} bindings out\n",
        rollouts.len(),
        feedback.len()
    );
    println!("[");
    for (i, fb) in feedback.iter().enumerate() {
        let target = match fb.target {
            FeedbackTarget::Rollout(id) => label(id),
            _ => "episode",
        };
        let comma = if i + 1 < feedback.len() { "," } else { "" };
        println!(
            "  {{ \"bound_to\": \"{target}\", \"metric\": \"{}\", \"outcome\": \"{}\", \
             \"method\": \"{:?}\", \"confidence\": {:.2} }}{comma}",
            fb.metric_name,
            value(fb),
            fb.join_method,
            fb.join_confidence,
        );
    }
    println!("]");
    println!(
        "\nEach binding is a decision→outcome link the cascade reconstructed — the input the gate \
         and RCA build on. No id was attached to any outcome; the join inferred every link."
    );
}
