---
format: aep.planning-md/3
id: resource-blocker:disk-under-25g-unit-4-m1-h
kind: resource-blocker
status: cleared
title: Disk under 25G free holds unit 4 of wave m1-h
relations:
- blocks: story:observatory-ui-live-loss-chart
revision: 2
transitions:
- {from: "open", to: "cleared", at: "2026-10-23T13:09:36Z", actor: "human:timo", revision: 2}
---
# Disk under 25G free holds unit 4 of wave m1-h

- `df -h /` read 24G free (98%) on 2026-10-23, under the 25G stop line of the orchestrator's brief.
- The unit 4 adversary's build was stopped. modelspec held 2.0G (`modelspec-impl-m1h-ui-chart/target`);
  the running builds were other repositories' sessions.
- b10x-chef-orchestrator, verbatim: "Ack, keep it stopped. I send \"resume\" when / is at 30G or more
  (df). From now b10x-build-slot itself pauses running builds under 25G and resumes at 30G, so run
  builds through it as before."
- Clears on the orchestrator's "resume".
