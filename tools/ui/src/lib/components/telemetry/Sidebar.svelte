<script lang="ts">
 import { onMount } from 'svelte';
 import { t } from '$lib/stores/locale.svelte';
 import { telemetry } from '$lib/stores/telemetry.svelte';
 import Trend from './Trend.svelte';
 let { openDetails }: { openDetails: () => void } = $props();
 let collapsed = $state(false);
 onMount(() => { try { collapsed = localStorage.getItem('telemetry-sidebar') === 'closed'; } catch {} });
 function toggle() { collapsed = !collapsed; try { localStorage.setItem('telemetry-sidebar', collapsed ? 'closed' : 'open'); } catch {} }
 let data = $derived(telemetry.snapshot);
 let inf = $derived(data?.inference);
 const f = (x?: number | null) => x == null ? '—' : x.toLocaleString(undefined,{maximumFractionDigits:1});
 const gib = (x?: number | null) => f(x == null ? null : x/2**30);
 let ram = $derived(data?.host.mem_total_bytes == null || data.host.mem_available_bytes == null ? null : data.host.mem_total_bytes - data.host.mem_available_bytes);
</script>
<aside class="monitor-rail" class:collapsed aria-label={t('本机实时状态','Local status')}>
 <button class="fold" aria-expanded={!collapsed} onclick={toggle}>{collapsed ? t('展开本机状态','Show local status') : t('收起','Collapse')}</button>
 {#if !collapsed}
 <h2>{t('本机实时状态','Local status')}</h2>
 <button class="details" onclick={openDetails}>{t('展开详情','Open details')} →</button>
 <div class="reading"><span>{t('预填充','Prefill')} <b>{f(inf?.prefill_tps)} tok/s</b></span><span>{t('吐字','Decode')} <b>{f(inf?.decode_tps)} tok/s</b></span></div>
 <Trend title={t('预填充','Prefill')} points={inf?.history ?? []} end={data?.sampled_at_ms} field="prefill_tps"/>
 <Trend title={t('吐字','Decode')} points={inf?.history ?? []} end={data?.sampled_at_ms} field="decode_tps"/>
 <div class="reading"><b>{t('主机内存','Host memory')}</b><span>{gib(ram)} / {gib(data?.host.mem_total_bytes)} GiB</span></div>
 <h3>{t('逐卡状态','GPU status')}</h3>
 {#each data?.gpus ?? [] as g (g.id)}
 <section class="rail-gpu">
  <div class="reading"><b>GPU {g.id}</b><span>{g.name}</span><b>{f(g.power_w)} W</b></div>
  <div class="reading"><span>{gib(g.mem_used_bytes)} / {gib(g.mem_total_bytes)} GiB</span><span>{f(g.gpu_util_pct)}% · {f(g.temp_c)} °C</span></div>
  {#if g.mem_used_bytes != null && g.mem_total_bytes}<progress aria-label={t('显存','VRAM')} value={g.mem_used_bytes} max={g.mem_total_bytes}></progress>{/if}
 </section>
 {:else}<small>{data && data.nvml_status !== 'ok' ? t('NVML 不可用','NVML unavailable') : t('暂无 GPU 采样','No GPU samples')}</small>{/each}
 {/if}
</aside>
<style>
 .monitor-rail{display:none} @media(min-width:1280px){.monitor-rail{display:block;position:sticky;top:5.5rem;align-self:flex-start;width:clamp(17rem,24vw,24rem);flex:none;max-height:calc(100dvh - 6.5rem);overflow:auto;margin:1rem 1.2rem 0 0;background:var(--card);border:1px solid var(--border);border-radius:1rem;padding:.9rem}.collapsed{width:auto}}
 .fold{float:right}.fold,.details{font-size:.75rem;border-radius:999px;padding:.3rem .55rem;background:var(--secondary)}.details{color:var(--live-decode);margin:.5rem 0}.reading{display:flex;justify-content:space-between;gap:.4rem;font-size:.75rem;margin:.35rem 0}h2,h3{font-weight:700}h3{margin:.6rem 0}.rail-gpu{border-top:1px solid var(--border);padding:.35rem 0}progress{width:100%;height:.32rem;accent-color:var(--live-prefill);display:block}.monitor-rail :global(figure){padding:.4rem;margin:.4rem 0}
</style>
