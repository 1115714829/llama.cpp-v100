<script lang="ts">
 import type { HistoryPoint } from '$lib/stores/telemetry.svelte';
 import { t } from '$lib/stores/locale.svelte';
 let { title, points, end, field }: {title: string; points: HistoryPoint[]; end?: number; field: 'prefill_tps' | 'decode_tps'} = $props();
 let visible = $derived(points.filter(p => end != null && p.t_ms >= end - 60000 && p.t_ms <= end).sort((a,b) => a.t_ms - b.t_ms));
 let max = $derived(Math.max(1, ...visible.map(p => p[field] ?? 0)));
 let segments = $derived.by(() => {
  const lines: {x:number; y:number}[][] = [];
  let line: {x:number; y:number}[] = [];
  let last = -Infinity;
  for (const p of visible) {
   const v = p[field];
   if (v == null || p.t_ms - last > 1500) { if (line.length) lines.push(line); line = []; }
   if (v != null) line.push({x:52 + 400 * (1 - (end! - p.t_ms) / 60000), y:122 - 100 * v / max});
   last = p.t_ms;
  }
  if (line.length) lines.push(line);
  return lines;
 });
</script>
<figure style:--trace={field === 'prefill_tps' ? 'var(--live-prefill)' : 'var(--live-decode)'}>
 <figcaption>{title}<small>tok/s · {t('过去 60 秒','Last 60 s')}</small></figcaption>
 <svg viewBox="0 0 470 150" role="img" aria-label={title}>
  {#each [0,1,2,3] as i (i)}
   <line x1="52" x2="452" y1={122-i*100/3} y2={122-i*100/3}/><text x="44" y={125-i*100/3} text-anchor="end">{segments.length ? +(max*i/3).toFixed(1) : '—'}</text>
  {/each}
  {#each [60,45,30,15,0] as s (s)}<text x={52+(60-s)*400/60} y="141" text-anchor="middle">{s ? `−${s}s` : t('现在','Now')}</text>{/each}
  {#each segments as line, i (i)}
   {#if line.length === 1}<circle cx={line[0].x} cy={line[0].y} r="2.5"/>{:else}<polyline points={line.map(p => `${p.x},${p.y}`).join(' ')}/>{/if}
  {:else}<text x="252" y="72" text-anchor="middle">{t('暂无有效样本','No valid samples')}</text>{/each}
 </svg>
</figure>
<style>
 figure{margin:0;border:1px solid var(--border);background:var(--card);border-radius:.95rem;padding:.9rem}figcaption{display:flex;justify-content:space-between;align-items:center;font-weight:700}small,text{color:var(--muted-foreground);font-size:.72rem;font-weight:400}svg{width:100%;display:block}line{stroke:var(--border)}text{fill:var(--muted-foreground)}polyline{fill:none;stroke:var(--trace);stroke-width:2;stroke-linejoin:round;stroke-linecap:round}circle{fill:var(--trace)}
</style>
