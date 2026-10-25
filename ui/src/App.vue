<script setup lang="ts">
import { computed, onBeforeUnmount, ref, shallowRef } from 'vue';
import MetricChart from './components/MetricChart.vue';
import ParameterPanel from './components/ParameterPanel.vue';
import { LOSS, apply, charts, emptyRun, sampleAt } from './lib/series.ts';
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
const parameters = computed(() => {
  void version.value;
  return run.value.parameters;
});
const sampled = computed(() => {
  void version.value;
  return run.value.samples.map((s) => s.step);
});
// The step whose histograms are drawn; `null` follows the latest sample as the run goes on.
const selected = ref<number | null>(null);
const shown = computed(() => {
  void version.value;
  return sampleAt(run.value, selected.value);
});
</script>

<template>
  <main :data-run-state="state">
    <header data-ux-region="header">
      <h1>ModelSpec observatory</h1>
      <p class="status">
        <span class="state" :class="state">{{ state }}</span>
        <span v-if="progress">{{ progress }}</span>
        <span v-if="interrupted" class="interrupted">connection lost, reconnecting</span>
      </p>
      <p v-if="runId" class="run">run {{ runId }}</p>
    </header>
    <p v-if="state === 'failed'" class="failure" role="alert" data-ux-region="failure">{{ failure }}</p>
    <section class="charts">
      <MetricChart
        v-for="s in series"
        :key="s.name"
        :series="s"
        :version="version"
        :region="s.name === LOSS ? 'loss-chart' : 'metric-chart'"
      />
    </section>
    <section v-if="parameters.length > 0" class="parameters" data-ux-region="parameters">
      <label class="pick">
        histograms at
        <select v-model="selected" data-step-select>
          <option :value="null">latest sampled step</option>
          <option v-for="step in sampled" :key="step" :value="step">step {{ step }}</option>
        </select>
      </label>
      <ParameterPanel
        v-for="p in parameters"
        :key="p.path"
        :series="p"
        :sample="shown"
        :version="version"
      />
    </section>
  </main>
</template>
