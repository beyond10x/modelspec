---
format: aep.planning-md/3
id: story:observatory-ui-keeps-up-with-fast-runs
kind: story
status: draft
title: The observatory UI keeps up with a fast run
relations:
- serves: vision:modelspec
revision: 1
---
# The observatory UI keeps up with a fast run

## Why

A CPU run of `examples/mnist_mlp.mspec` on MNIST, served with `--serve 127.0.0.1:8081 --ui ui/dist`,
publishes about 1400 StepCompleted events per second (300000 steps, 2026-10-23). Watching it in
Brave on the same machine, the operator said: "crazy, ui cannot keep up with how fast you push stuff
in there". A Vulkan run publishes its 72000 steps at once after its host wait, which the UI also
has to absorb.

## Acceptance

- The UI renders at most one chart update per animation frame, whatever the event rate, and the
  loss chart's last point is the newest step received.
- A browser test feeds 100000 StepCompleted events in one burst and checks that the page shows the
  last step within a stated time and stays responsive to a click.
- `task ci` exits 0.
