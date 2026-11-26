import { base } from '$app/paths';

type Reading = number | null;
export type HistoryPoint = { t_ms: number; prefill_tps: Reading; decode_tps: Reading; active_slots: number };
export type Telemetry = {
 schema_version: 1; epoch: number; seq: number; sampled_at_ms: number;
 nvml_status: 'ok' | 'unavailable' | 'error'; model: { display_name: string };
 inference: { phase: 'loading' | 'idle' | 'prefill' | 'decode' | 'mixed'; active_slots: number;
 prefill_tps: Reading; decode_tps: Reading; last_prefill_tps: Reading; last_decode_tps: Reading;
 ttft_ms: Reading; draft_proposed: number; draft_accepted: number; draft_acceptance_rate: Reading; history: HistoryPoint[] };
 slots: { id: number; state: 'idle' | 'prefill' | 'decode'; context_used_tokens: number; context_capacity_tokens: number }[];
 gpus: { id: number; name: string; mem_used_bytes: Reading; mem_total_bytes: Reading; gpu_util_pct: Reading; temp_c: Reading; power_w: Reading; enforced_limit_w: Reading; status: 'ok' | 'partial' | 'error' }[];
 host: { mem_total_bytes: Reading; mem_available_bytes: Reading };
};

class TelemetryStore {
 snapshot = $state.raw<Telemetry | null>(null);
 state = $state<'connecting' | 'ready' | 'loading' | 'unauthorized' | 'unavailable' | 'offline'>('connecting');

 connect(key: string) {
  let disposed = false;
  let timer: ReturnType<typeof setTimeout> | undefined;
  let pending: AbortController | undefined;
  this.snapshot = null;
  this.state = 'connecting';
  const update = async () => {
   if (disposed || document.hidden || pending) return;
   const request = new AbortController();
   pending = request;
   const timeout = setTimeout(() => request.abort(), 5000);
   try {
    const response = await fetch(`${base}/v1/v100/status`, {
     cache: 'no-store', signal: request.signal,
     headers: key ? { Authorization: `Bearer ${key}` } : {}
    });
    if (disposed || document.hidden) return;
    if (!response.ok) {
     this.snapshot = null;
     this.state = response.status === 401 || response.status === 403 ? 'unauthorized' :
      response.status === 503 ? 'loading' : response.status === 404 ? 'unavailable' : 'offline';
     return;
    }
    const data: Telemetry = await response.json();
    if (disposed || request.signal.aborted || document.hidden) return;
    if (data?.schema_version !== 1 || !data.model || !data.host || !Array.isArray(data.gpus) || !Array.isArray(data.slots) || !Array.isArray(data.inference?.history) || !['loading', 'idle', 'prefill', 'decode', 'mixed'].includes(data.inference.phase)) {
     this.snapshot = null;
     this.state = 'unavailable';
     return;
    }
    this.snapshot = data;
    this.state = 'ready';
   } catch {
    if (!disposed && !document.hidden) {
     this.snapshot = null;
     this.state = 'offline';
    }
   } finally {
    clearTimeout(timeout);
    pending = undefined;
    if (!disposed && !document.hidden) timer = setTimeout(update, 1000);
   }
  };
  const visibility = () => {
   clearTimeout(timer);
   if (document.hidden) pending?.abort();
   else void update();
  };
  document.addEventListener('visibilitychange', visibility);
  void update();
  return () => {
   disposed = true;
   clearTimeout(timer);
   pending?.abort();
   document.removeEventListener('visibilitychange', visibility);
  };
 }
}
export const telemetry = new TelemetryStore();
