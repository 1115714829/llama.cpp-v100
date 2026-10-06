#!/bin/sh
# Container entrypoint: starts llama-server with the arguments given to `docker run`.
# LLAMA_NUMA_MEMBIND=<nodes> (for example 0,8) starts it under `numactl --membind=<nodes>`.
if [ -n "${LLAMA_NUMA_MEMBIND:-}" ]; then
    exec numactl --membind="${LLAMA_NUMA_MEMBIND}" /app/llama-server "$@"
fi
exec /app/llama-server "$@"
