<script setup lang="ts">
import { computed } from 'vue';
import type { ParameterStatistics } from '../generated/observatory.ts';
import type { ParameterSeries, Sample } from '../lib/series.ts';
import HistogramChart from './HistogramChart.vue';
import MetricChart from './MetricChart.vue';

// One parameter: the norm of its values and of its gradient per sampled step, and the histograms
// of the sample the page shows.
const props = defineProps<{ series: ParameterSeries; sample: Sample | null; version: number }>();

function entry(subject: ParameterStatistics['subject']): ParameterStatistics | null {
  void props.version;
  return props.sample?.statistics.find((s) => s.path === props.series.path && s.subject === subject) ?? null;
}

const value = computed(() => entry('Value'));
const gradient = computed(() => entry('Gradient'));
</script>

<template>
  <section class="parameter" :data-parameter="series.path" data-ux-region="parameter-panel">
    <h2>{{ series.path }}</h2>
    <MetricChart
      :series="series.value"
      :version="version"
      :tag="{ 'data-norm': series.path, 'data-subject': 'Value' }"
    />
    <MetricChart
      :series="series.gradient"
      :version="version"
      :tag="{ 'data-norm': series.path, 'data-subject': 'Gradient' }"
    />
    <HistogramChart
      v-if="sample && value"
      :path="series.path"
      subject="Value"
      :step="sample.step"
      :histogram="value.histogram"
    />
    <HistogramChart
      v-if="sample && gradient"
      :path="series.path"
      subject="Gradient"
      :step="sample.step"
      :histogram="gradient.histogram"
    />
    <p v-else-if="sample" class="note">no gradient at step {{ sample.step }}</p>
  </section>
</template>
