<script setup lang="ts">
import { computed } from 'vue';
import type { Histogram, StatisticsSubject } from '../generated/observatory.ts';
import { FRAME, label } from '../lib/chart.ts';
import { bars, nonFinite } from '../lib/histogram.ts';

const props = defineProps<{
  path: string;
  subject: StatisticsSubject;
  step: number;
  histogram: Histogram;
}>();

const drawn = computed(() => bars(FRAME, props.histogram.counts));
const outside = computed(() => nonFinite(props.histogram));
const edges = computed(() => {
  const e = props.histogram.bin_edges;
  if (e.length === 0) {
    return null;
  }
  return { low: label(e[0] ?? null), high: label(e.at(-1) ?? null) };
});
const title = computed(() => (props.subject === 'Value' ? 'values' : 'gradient'));
</script>

<template>
  <figure class="chart histogram" data-ux-region="histogram">
    <figcaption>
      <span class="name">{{ path }} {{ title }}</span>
      <span class="last">step {{ step }}<template v-if="outside">, {{ outside }} outside the bins</template></span>
    </figcaption>
    <svg
      :viewBox="`0 0 ${FRAME.width} ${FRAME.height}`"
      role="img"
      :aria-label="`histogram of ${path} ${title} at step ${step}`"
      :data-histogram="path"
      :data-subject="subject"
      :data-step="step"
      :data-counts="histogram.counts.join(',')"
    >
      <g class="axes">
        <line
          :x1="FRAME.margin.left"
          :x2="FRAME.width - FRAME.margin.right"
          :y1="FRAME.height - FRAME.margin.bottom"
          :y2="FRAME.height - FRAME.margin.bottom"
        />
        <template v-if="edges">
          <text :x="FRAME.margin.left" :y="FRAME.height - FRAME.margin.bottom + 18" class="step">
            {{ edges.low }}
          </text>
          <text :x="FRAME.width - FRAME.margin.right" :y="FRAME.height - FRAME.margin.bottom + 18" class="step end">
            {{ edges.high }}
          </text>
        </template>
        <text v-else :x="FRAME.margin.left" :y="FRAME.height - FRAME.margin.bottom + 18" class="step">
          no finite element
        </text>
      </g>
      <rect
        v-for="(bar, i) in drawn"
        :key="i"
        data-bin
        :x="bar.x"
        :y="bar.y"
        :width="bar.width"
        :height="bar.height"
      >
        <title>{{ bar.count }}</title>
      </rect>
    </svg>
  </figure>
</template>
