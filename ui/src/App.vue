<script setup lang="ts">
import { computed, onBeforeUnmount, ref, shallowRef } from 'vue';
import MetricChart from './components/MetricChart.vue';
import { apply, charts, emptyRun } from './lib/series.ts';
import { subscribe } from './lib/stream.ts';

// The run is folded in place and redrawn at most once per animation frame: a fast run sends
// hundreds of events a second, and one render per event would fall behind the stream.
const run = shallowRef(emptyRun());
const version = ref(0);
const interrupted = ref(false);
let pending = false;

function redraw(): void {
  pending = false;
  version.value += 1;
}

const close = subscribe('events', {
  event(event) {
    interrupted.value = false;
    if (apply(run.value, event) && !pending) {
      pending = true;
      requestAnimationFrame(redraw);
    }
  },
  interrupted() {
    interrupted.value = true;
  },
});
onBeforeUnmount(close);

const state = computed(() => {
  void version.value;
  return run.value.state;
});
const series = computed(() => {
  void version.value;
  return charts(run.value);
});
// A run of `steps` steps from `start_step` records the loss at every step from `start_step` to
// `start_step + steps`, both included.
const progress = computed(() => {
  void version.value;
  const r = run.value;
  if (r.steps === null || r.startStep === null) {
    return null;
  }
  const step = r.loss.points.at(-1)?.step ?? r.startStep;
  return `step ${step} of ${r.startStep + r.steps}`;
});
const runId = computed(() => {
  void version.value;
  return run.value.trainingRunId;
});
const failure = computed(() => {
  void version.value;
  return run.value.failure;
});
</script>

<template>
  <main :data-run-state="state">
    <header>
      <h1>ModelSpec observatory</h1>
      <p class="status">
        <span class="state" :class="state">{{ state }}</span>
        <span v-if="progress">{{ progress }}</span>
        <span v-if="interrupted" class="interrupted">connection lost, reconnecting</span>
      </p>
      <p v-if="runId" class="run">run {{ runId }}</p>
    </header>
    <p v-if="state === 'failed'" class="failure" role="alert">{{ failure }}</p>
    <section class="charts">
      <MetricChart v-for="s in series" :key="s.name" :series="s" :version="version" />
    </section>
  </main>
</template>
