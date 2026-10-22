---
format: aep.planning-md/3
id: review-result:m1-vulkan-profile-adversary-1
kind: review-result
status: active
title: Adversary review of m1-vulkan-profile at a041325
relations:
- reviews: story:m1-vulkan-profile
revision: 1
---
# Adversary review of story:m1-vulkan-profile at a041325

unit: story:m1-vulkan-profile, clone at a041325 (base ba1370c)
verdict: NEEDS-CHANGE
cases: `task ci` executed 1232 → 1233; on the RTX 3090, 5 cases added, 2 red
origin: introduced 1 / pre-existing 0 / undecided 0

## Cases added

| case | asserts | result |
|---|---|---|
| `vulkan_profile_adversary::step_counts_around_the_command_buffer_boundary_time_every_dispatch` | 1, 21, 255, 256, 257, 512, 513 steps: calls = steps, totals > 0, steady state only above 20 steps, submits and dispatches equal to an unprofiled run, losses and parameters bit-equal, no validation messages | green |
| `vulkan_profile_adversary::the_step_sites_do_not_count_one_interval_twice` | 600 steps: sum of step-site means ≤ `steady_us_per_step` (168.166 ≤ 172.744) | green |
| `vulkan_profile_adversary::the_forward_lines_are_the_dispatch_steps_of_the_forward_plan` | acceptance 1 and 3 as written | red: `left: [0, 2, 4, 6] right: [0, 2, 4]` |
| `train_profile_adversary::profile_on_the_cpu_is_refused_in_every_flag_order` | 5 flag orders exit 2 naming `'--profile'` | green |
| `train_profile_adversary::a_forward_line_names_a_dispatch_step_of_plan_forward` | each `forward <p> <id>` line is a dispatch of `plan forward` | red: `forward 6 74ad23b6-… names no dispatch of plan forward` |

## Finding

`crates/modelspec-runtime/src/vulkan/runner.rs:327`: the `forward` sites are `actions[..loss_end]`,
the step plan up to the loss (4 dispatches, step-plan positions 0, 2, 4, 6), where acceptance 1 and
3 name the forward plan's 3 Dispatch steps (32,403 dispatches; the run prints 32,404), and the
guide (`website/docs/guides/inspect-a-run.md:59`) says the position is in "its plan".

## Attacked and not broken

Query pool size and reset across command buffers (0 to 1800 steps); attribution (step-site sum
below steady time by 3 to 5 µs of barriers and copies); bit-equal losses and parameters and
byte-equal stdout with and without `--profile`; 1 host wait either way; CPU refusal in 5 flag
orders; a single counter wrap. Not covered: a device without timestamp queries, a counter
narrower than 64 bits, start stamps on other vendors.

```findings
- file: crates/modelspec-runtime/src/vulkan/runner.rs
  line: 327
  category: acceptance
  severity: warning
  verdict: NEEDS-CHANGE
  origin: introduced
  message: forward lines are the step plan up to the loss (4 sites, step-plan positions, 32404 dispatches) where acceptance 1 and 3 name the forward plan's 3 Dispatch steps (32403), and `forward 6` names no dispatch of `plan forward` in ir --stage kernel
```

## Resolution (coordinator)

The plan word becomes `final` (`PlanName::Final`), its positions documented as step-plan
positions in DESIGN.md § 8, the guide (§ 3 and the field table) and the story's acceptance 1 and 3
(revision 11). The two red cases were rewritten to that contract and kept in the unit:
`the_final_lines_are_the_step_plan_dispatches_up_to_the_loss` and
`a_final_line_names_a_dispatch_step_of_plan_step`.
