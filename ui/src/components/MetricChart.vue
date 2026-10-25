<script setup lang="ts">
import { computed } from 'vue';
import { FRAME, domain, label, line, project } from '../lib/chart.ts';
import type { Series } from '../lib/series.ts';

// `tag` holds the attributes that say which chart this is, `data-metric` and the series name
// unless given: a reader of `svg[data-metric]` reads the loss and metric charts and no other.
// `region` is the chart's `data-ux-region` mark for the UI loop's recorder and gates (`ui/ux/`).
const props = defineProps<{ series: Series; version: number; tag?: Record<string, string>; region?: string }>();

const tag = computed(() => props.tag ?? { 'data-metric': props.series.name });

const last = computed(() => {
  void props.version;
  return props.series.points.at(-1) ?? null;
});
const range = computed(() => {
  void props.version;
  return domain(props.series.points);
});
const d = computed(() => {
  void props.version;
  return line(FRAME, props.series.points);
});
const axes = computed(() => {
  const r = range.value;
  if (r === null) {
    return null;
  }
  const { x, y } = project(FRAME, r);
  return {
    left: x(r.steps[0]),
    right: x(r.steps[1]),
    top: y(r.values[1]),
    bottom: y(r.values[0]),
    labels: {
      first: String(r.steps[0]),
      last: String(r.steps[1]),
      low: label(r.values[0]),
      high: label(r.values[1]),
    },
  };
});
</script>

<template>
  <figure class="chart" :data-ux-region="region">
    <figcaption>
      <span class="name">{{ series.name }}</span>
      <span v-if="last" class="last">step {{ last.step }}: {{ label(last.value) }}</span>
    </figcaption>
    <svg
      :viewBox="`0 0 ${FRAME.width} ${FRAME.height}`"
      role="img"
      :aria-label="`${series.name} per step`"
      v-bind="tag"
      :data-points="series.points.length"
      :data-last-step="last === null ? undefined : last.step"
      :data-last-value="last === null ? undefined : String(last.value)"
    >
      <g v-if="axes" class="axes">
        <line :x1="axes.left" :x2="axes.right" :y1="axes.bottom" :y2="axes.bottom" />
        <line :x1="axes.left" :x2="axes.left" :y1="axes.top" :y2="axes.bottom" />
        <text :x="axes.left - 6" :y="axes.top" class="value">{{ axes.labels.high }}</text>
        <text :x="axes.left - 6" :y="axes.bottom" class="value">{{ axes.labels.low }}</text>
        <text :x="axes.left" :y="axes.bottom + 18" class="step">{{ axes.labels.first }}</text>
        <text :x="axes.right" :y="axes.bottom + 18" class="step end">{{ axes.labels.last }}</text>
      </g>
      <path data-line :d="d" />
    </svg>
  </figure>
</template>
