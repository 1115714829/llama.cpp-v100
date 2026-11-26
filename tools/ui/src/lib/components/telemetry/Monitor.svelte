<script lang="ts">
 import { Activity, Moon, Sun } from '@lucide/svelte';
 import { mode, setMode } from 'mode-watcher';
 import { locale, t } from '$lib/stores/locale.svelte';
 import { telemetry } from '$lib/stores/telemetry.svelte';
 import { settingsStore } from '$lib/stores';
 import Trend from './Trend.svelte';
 let dialog: HTMLDialogElement;
 export function openDetails() { dialog.showModal(); }
 let data = $derived(telemetry.snapshot);
 let inf = $derived(data?.inference);
 let phase = $derived.by(() => {
  if (telemetry.state !== 'ready') return ({ connecting: t('连接中','Connecting'), loading: t('加载中','Loading'), unauthorized: t('遥测未授权','Telemetry unauthorized'), unavailable: t('遥测不可用','Telemetry unavailable'), offline: t('连接中断','Disconnected') })[telemetry.state];
  return ({ loading: t('加载中','Loading'), idle: t('空闲','Idle'), prefill: t('预填充','Prefill'), decode: t('吐字','Decoding'), mixed: t('并发混合','Mixed') })[inf!.phase];
 });
 const n = (v?: number | null) => v == null ? '—' : new Intl.NumberFormat(locale.language, {maximumFractionDigits: 1}).format(v);
 const gib = (v?: number | null) => n(v == null ? null : v / 2 ** 30);
 const sum = (values: (number | null)[] = []) => values.length && values.every(v => v != null) ? values.reduce<number>((a, v) => a + (v ?? 0),0) : null;
 const speed = (v?: number | null) => v == null ? '—' : `${n(v)} tok/s`;
 let ramUsed = $derived(data?.host.mem_total_bytes == null || data.host.mem_available_bytes == null ? null : data.host.mem_total_bytes - data.host.mem_available_bytes);
 $effect(() => telemetry.connect(String(settingsStore.config.apiKey ?? '').trim()));
</script>

<header class="local-bar">
 <div class="local-brand"><span class="brand-icon"><Activity size={22}/></span><div><strong>{t('本地推理台','Local Inference')}</strong><small>LLAMA.CPP-V100</small></div></div>
 <span class="local-model" title={data?.model.display_name}>{data?.model.display_name || 'llama.cpp-v100'}</span>
 <button class="local-status" onclick={openDetails} aria-haspopup="dialog">
  <span class="phase" data-ready={telemetry.state === 'ready'}>● {phase}</span>
  <span class="rate prefill"><small>{t('实时预填充','Live prefill')}</small><b>{speed(inf?.prefill_tps)}</b>{#if inf?.prefill_tps == null && inf?.last_prefill_tps != null}<small>{t('上次','Last')}: {speed(inf.last_prefill_tps)}</small>{/if}</span>
  <span class="rate decode"><small>{t('实时吐字','Live decode')}</small><b>{speed(inf?.decode_tps)}</b>{#if inf?.decode_tps == null && inf?.last_decode_tps != null}<small>{t('上次','Last')}: {speed(inf.last_decode_tps)}</small>{/if}</span>
  <span class="local-gpu"><small>{t('功耗 · 显存','Power · VRAM')}</small><b>{n(sum(data?.gpus.map(g => g.power_w)))} W · {gib(sum(data?.gpus.map(g => g.mem_used_bytes)))} / {gib(sum(data?.gpus.map(g => g.mem_total_bytes)))} GiB</b></span>
 </button>
 <button class="local-control" onclick={() => locale.toggle()} title={t('切换到英语','Switch to Chinese')}>{locale.language === 'en' ? '中文' : 'EN'}</button>
 <button class="local-control" onclick={() => setMode(mode.current === 'dark' ? 'light' : 'dark')} title={t('切换日间 / 夜间','Switch light / dark')}>{#if mode.current === 'dark'}<Moon size={15}/>{t('夜间','Dark')}{:else}<Sun size={15}/>{t('日间','Light')}{/if}</button>
</header>

<dialog class="local-drawer" bind:this={dialog} aria-labelledby="telemetry-title">
 <div class="drawer-head"><div><h2 id="telemetry-title">{t('本机实时性能','Local performance')}</h2><small>{data?.model.display_name || 'llama.cpp-v100'} · {phase}</small></div><button class="local-control" onclick={() => dialog.close()}>{t('关闭面板','Close panel')}</button></div>
 <div class="metric-grid">
  <article><small>{t('实时预填充','Live prefill')}</small><strong class="prefill">{speed(inf?.prefill_tps)}</strong><small>{t('上次','Last')}: {speed(inf?.last_prefill_tps)}</small></article>
  <article><small>{t('实时吐字','Live decode')}</small><strong class="decode">{speed(inf?.decode_tps)}</strong><small>{t('上次','Last')}: {speed(inf?.last_decode_tps)}</small></article>
  <article><small>{t('首字延迟','Time to first token')}</small><strong>{n(inf?.ttft_ms)} ms</strong></article>
  <article><small>{t('投机接受率','Draft acceptance')}</small><strong>{inf?.draft_proposed && inf.draft_acceptance_rate != null ? `${n(100 * inf.draft_acceptance_rate)}%` : '—'}</strong><small>{n(inf?.draft_accepted)} / {n(inf?.draft_proposed)}</small></article>
 </div>
 <div class="trend-grid">
  <Trend title={t('预填充速度','Prefill speed')} points={inf?.history ?? []} end={data?.sampled_at_ms} field="prefill_tps"/>
  <Trend title={t('吐字速度','Decode speed')} points={inf?.history ?? []} end={data?.sampled_at_ms} field="decode_tps"/>
 </div>
 <h3>{t('本机 GPU','Local GPUs')} <small>{data?.nvml_status === 'ok' ? data.gpus.length : ''}</small></h3>
 {#if data && data.nvml_status !== 'ok'}<p>{data.nvml_status === 'error' ? t('NVML 采集出错','NVML sampling failed') : t('NVML 不可用','NVML unavailable')}</p>{:else if !data || !data.gpus.length}<p>{t('暂无 GPU 采样','No GPU samples')}</p>{/if}
 <div class="gpu-grid">
 {#each data?.gpus ?? [] as g (g.id)}
  <article class="gpu-card"><strong>GPU {String(g.id).padStart(2,'0')}</strong><small>{g.name}{g.status === 'ok' ? '' : ` · ${g.status === 'partial' ? t('部分可用','Partial') : t('采集错误','Error')}`}</small>
  <div class="reading"><span>{t('显存','VRAM')}</span><b>{gib(g.mem_used_bytes)} / {gib(g.mem_total_bytes)} GiB</b></div>
  {#if g.mem_used_bytes != null && g.mem_total_bytes != null && g.mem_total_bytes > 0}<progress aria-label={t('显存','VRAM')} value={g.mem_used_bytes} max={g.mem_total_bytes}></progress>{/if}
  <div class="reading"><span>{t('利用率','Utilization')}</span><b>{n(g.gpu_util_pct)}%</b></div>
  {#if g.gpu_util_pct != null}<progress class="util" aria-label={t('利用率','Utilization')} value={g.gpu_util_pct} max="100"></progress>{/if}
  <div class="reading"><span>{n(g.temp_c)} °C</span><b>{n(g.power_w)} / {n(g.enforced_limit_w)} W</b></div></article>
 {/each}
 </div>
 <div class="trend-grid resources">
  <article><div class="reading"><b>{t('主机内存','Host memory')}</b><span>{gib(ramUsed)} / {gib(data?.host.mem_total_bytes)} GiB</span></div>{#if ramUsed != null && data?.host.mem_total_bytes}<progress aria-label={t('主机内存','Host memory')} value={ramUsed} max={data.host.mem_total_bytes}></progress>{/if}</article>
  <article><b>{t('活跃槽上下文','Active slot context')} · {n(inf?.active_slots)}</b>
   {#each (data?.slots ?? []).filter(s => s.state !== 'idle') as s (s.id)}
    <div class="reading"><span>#{s.id} · {s.state === 'prefill' ? t('预填充','Prefill') : t('吐字','Decode')}</span><span>{n(s.context_used_tokens)} / {n(s.context_capacity_tokens)} tok</span></div>
    {#if s.context_capacity_tokens > 0}<progress class="util" aria-label={`${t('槽位','Slot')} ${s.id}`} value={s.context_used_tokens} max={s.context_capacity_tokens}></progress>{/if}
   {:else}<small>{t('暂无活跃槽','No active slots')}</small>{/each}
  </article>
 </div>
</dialog>

<style>
 .local-bar{background:var(--card);border-bottom:1px solid var(--border);display:flex;gap:clamp(.3rem,1.4vw,1.3rem);align-items:center;padding:.65rem clamp(.65rem,2vw,1.8rem);min-height:4.6rem;container-type:inline-size}
 .local-brand{display:flex;gap:.62rem;align-items:center;white-space:nowrap;font-size:1.05rem}.brand-icon{display:grid;place-items:center;width:2.3rem;height:2.3rem;border-radius:.65rem;color:#102335;background:linear-gradient(135deg,#9cf2df,#80adff)}
 small{display:block;color:var(--muted-foreground);font-size:.72rem}.local-brand small{font-family:monospace;font-size:.63rem;letter-spacing:.05em}.local-model{min-width:0;max-width:19ch;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;color:var(--muted-foreground)}
 .local-status{display:flex;align-items:center;justify-content:space-evenly;gap:clamp(.5rem,2vw,2rem);margin-left:auto;min-width:0;text-align:left}.local-control,.phase{border-radius:999px;background:var(--secondary);padding:.43rem .75rem;white-space:nowrap;font-size:.78rem;font-weight:650}.phase[data-ready='true']{color:var(--live-decode)}.local-control{display:flex;align-items:center;gap:.3rem}.local-control:hover,.local-status:hover .phase{background:var(--accent)}
 .local-status b{white-space:nowrap;font-size:.9rem}.prefill{color:var(--live-prefill)}.decode{color:var(--live-decode)}
 .local-drawer{position:fixed;inset:0 0 0 auto;margin:0;width:min(100%,82rem);max-width:none;height:100dvh;max-height:none;padding:clamp(1rem,3vw,2.5rem);border:1px solid var(--border);background:var(--background);color:var(--foreground);overflow:auto}.local-drawer::backdrop{background:#050c18a6}.drawer-head{display:flex;justify-content:space-between;align-items:center;gap:1rem;margin-bottom:1rem}.drawer-head h2{font-size:clamp(1.45rem,3vw,2rem);font-weight:750}h3{font-size:1.15rem;font-weight:700;margin:.8rem 0;display:flex;align-items:center;gap:.6rem}
 article{background:var(--card);border:1px solid var(--border);border-radius:.95rem;padding:.9rem;min-width:0}.metric-grid,.trend-grid,.gpu-grid{display:grid;gap:.8rem}.metric-grid{grid-template-columns:repeat(auto-fit,minmax(9.5rem,1fr))}.metric-grid strong{display:block;font-size:1.36rem}.trend-grid{grid-template-columns:repeat(auto-fit,minmax(min(100%,24rem),1fr));margin-top:.8rem}.gpu-grid{grid-template-columns:repeat(auto-fit,minmax(min(100%,9rem),1fr))}.gpu-card small{overflow-wrap:anywhere}.reading{display:flex;justify-content:space-between;gap:.35rem;flex-wrap:wrap;margin:.45rem 0;font-size:.78rem}progress{appearance:none;display:block;width:100%;height:.3rem;border:none;border-radius:999px;overflow:hidden;background:var(--secondary);color:var(--live-prefill);accent-color:var(--live-prefill)}progress::-webkit-progress-bar{background:var(--secondary)}progress::-webkit-progress-value{background:currentColor}progress::-moz-progress-bar{background:currentColor}.util{color:var(--live-decode);accent-color:var(--live-decode)}
 @container (max-width: 950px){.local-model{display:none}}
 @container (max-width: 720px){.local-gpu,.rate small:last-child{display:none}}
 @container (max-width: 535px){.local-brand div{display:none}.local-control{padding:.4rem}.local-status{gap:.35rem}.phase{padding:.3rem}.local-status b{font-size:.72rem}}
</style>
