#!/usr/bin/env bash
# build an expert routing profile from a calibration corpus
#
# 用校准语料跑一遍 llama-perplexity，收集每层每个专家的路由计数，拼成一份画像文件
# 供 --expert-profile 使用
#
# the counts are written when the model unloads, so the tool has to exit normally.
# -lv 4 is needed because the model shape lines (print_info) are INFO level and the
# default log threshold drops them / 模型形状行是 INFO 级别，默认日志门槛会丢掉，所以加 -lv 4
# Sending a signal or killing it loses the counts.
#
# sample usage:
#   bash scripts/v100-expert-profile.sh -m model.gguf -f corpus.txt -o profile.txt
#   bash scripts/v100-expert-profile.sh -m model.gguf -f corpus.txt -o profile.txt \
#       -c 4096 -n 200 -- -sm tensor -ngl 999 -ctk q8_0

set -euo pipefail

PROGNAME=$(basename "$0")

function usage() {
    cat <<EOF
usage: $PROGNAME -m <model.gguf> -f <corpus.txt> -o <profile.txt> [-c <ctx>] [-n <chunks>]
       [-b <bin-dir>] [--] <extra llama-perplexity args>

  用一段校准文本生成专家路由画像 / build an expert routing profile from a calibration text

  -m <model.gguf>   model to profile, required / 要采样的模型
  -f <corpus.txt>   calibration text, one document per line, required / 校准文本，每行一段
  -o <profile.txt>  output profile, read back by --expert-profile, required / 画像输出路径
  -c <ctx>          context size for the measurement, default 2048 / 采样时的上下文长度
  -n <chunks>       chunks to process, default all / 处理的 chunk 数
  -b <bin-dir>      directory holding llama-perplexity, default build/bin / 可执行文件目录
  -- <extra args>   passed to llama-perplexity unchanged / 原样传给 llama-perplexity 的参数

  example / 例子:
    bash $PROGNAME -m model.gguf -f corpus.txt -o profile.txt -- -sm tensor -ngl 999
EOF
    exit 2
}

# --- arguments / 参数 ---

MODEL=""
CORPUS=""
OUT=""
CTX=2048
CHUNKS=""
BIN_DIR="build/bin"
EXTRA=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        -m|--model)   MODEL="${2:-}";  shift 2 ;;
        -f|--file)    CORPUS="${2:-}"; shift 2 ;;
        -o|--output)  OUT="${2:-}";    shift 2 ;;
        -c|--ctx)     CTX="${2:-}";    shift 2 ;;
        -n|--chunks)  CHUNKS="${2:-}"; shift 2 ;;
        -b|--bin-dir) BIN_DIR="${2:-}"; shift 2 ;;
        -h|--help)    usage ;;
        --)           shift; EXTRA=("$@"); break ;;
        *)            EXTRA+=("$1"); shift ;;
    esac
done

[[ -n "$MODEL"  ]] || usage
[[ -n "$CORPUS" ]] || usage
[[ -n "$OUT"    ]] || usage
[[ -f "$MODEL"  ]] || { echo "error: model '$MODEL' not found / 模型不存在"; exit 2; }
[[ -f "$CORPUS" ]] || { echo "error: corpus '$CORPUS' not found / 校准文本不存在"; exit 2; }
[[ "$CTX" =~ ^[0-9]+$ ]] || { echo "error: -c needs a positive integer / -c 需要正整数"; exit 2; }
if [[ -n "$CHUNKS" ]]; then
    [[ "$CHUNKS" =~ ^([0-9]+|-1)$ ]] || { echo "error: -n needs an integer or -1 / -n 需要整数或 -1"; exit 2; }
fi
[[ -d "$(dirname -- "$OUT")" ]] || { echo "error: output directory does not exist / 输出目录不存在"; exit 2; }

PPL_BIN="$BIN_DIR/llama-perplexity"
[[ -x "$PPL_BIN" ]] || { echo "error: '$PPL_BIN' is missing or not executable / 找不到可执行的 llama-perplexity"; exit 3; }

WORK_DIR=$(mktemp -d)
trap 'rm -rf "$WORK_DIR"' EXIT

LOG="$WORK_DIR/perplexity.log"
RAW="$WORK_DIR/routing"
ROWS="$WORK_DIR/rows"
SORTED="$WORK_DIR/sorted"

# --- run the measurement / 采样 ---

echo "model   : $MODEL" / 模型
echo "corpus  : $CORPUS" / 校准文本
echo "ctx     : $CTX" / 上下文长度
echo "chunks  : ${CHUNKS:-all}" / chunk 数
echo "binary  : $PPL_BIN" / 可执行文件
echo

# a dump left over from an earlier run would mix two measurements into one profile
# 上一次的残留计数会混进这次的画像，先清掉
rm -f -- "$RAW" "$RAW".dev*

PPL_ARGS=(-m "$MODEL" -f "$CORPUS" -lv 4 -c "$CTX" --dump-expert-routing "$RAW")
if [[ -n "$CHUNKS" ]]; then
    PPL_ARGS+=(--chunks "$CHUNKS")
fi
if [[ ${#EXTRA[@]} -gt 0 ]]; then
    PPL_ARGS+=("${EXTRA[@]}")
fi

# pipefail keeps the tool's exit code here, and the counts only reach the disk on a normal
# exit / pipefail 保证退出码不被 tee 吞掉，计数只在正常退出时写出
if ! "$PPL_BIN" "${PPL_ARGS[@]}" 2>&1 | tee "$LOG"; then
    echo "error: llama-perplexity failed / 采样失败, see $LOG" >&2
    exit 4
fi

# --- read the model shape / 读取模型规模 ---

# the tool colorizes its log when asked to; drop the escape codes before matching
clean_log() {
    sed -E 's/\x1B\[[0-9;]*[a-zA-Z]//g' "$LOG"
}

N_LAYER=$(clean_log  | sed -nE 's/.*print_info: n_layer += *([0-9]+).*/\1/p'  | tail -n 1)
N_EXPERT=$(clean_log | sed -nE 's/.*print_info: n_expert += *([0-9]+).*/\1/p' | tail -n 1)

[[ -n "$N_LAYER"  ]] || { echo "error: no n_layer in the log / 日志里没有 n_layer"; exit 5; }
[[ -n "$N_EXPERT" ]] || { echo "error: no n_expert in the log / 日志里没有 n_expert"; exit 5; }
if (( N_LAYER < 1 || N_EXPERT < 1 )); then
    echo "error: n_layer=$N_LAYER n_expert=$N_EXPERT, not an MoE model / 不是 MoE 模型" >&2
    exit 5
fi

echo
echo "n_layer  : $N_LAYER"  / 层数
echo "n_expert : $N_EXPERT" / 每层专家数

# --- collect the counts / 汇总计数 ---

# one store per device writes its own file next to the raw path when the machine has
# more than one CUDA device; the numbers inside are already global, so the files merge
# 多卡时每张卡写一份，文件里的专家号已经是全局的，直接合并
DUMPS=()
if [[ -f "$RAW" ]]; then
    DUMPS+=("$RAW")
fi
for f in "$RAW".dev*; do
    [[ -e "$f" ]] || continue
    DUMPS+=("$f")
done

if [[ ${#DUMPS[@]} -eq 0 ]]; then
    echo "error: llama-perplexity wrote no routing counts / 没有写出路由计数" >&2
    exit 5
fi
echo "dumps    : ${#DUMPS[@]} file(s) / 计数文件数"

cat -- "${DUMPS[@]}" | grep -vE '^[[:space:]]*(#|$)' > "$ROWS" || true   # 去掉以 # 开头的说明行和空行 / drop comment and blank lines

if [[ ! -s "$ROWS" ]]; then
    echo "error: llama-perplexity wrote no routing counts / 没有写出路由计数" >&2
    exit 5
fi

# rows outside the model shape or rows that are not "layer expert count" mean the tool
# and the gguf disagree; stop instead of writing a profile that silently disagrees
# 行号越界或格式不对，说明工具与 gguf 不一致，直接报错，不写画像
if ! awk -v nl="$N_LAYER" -v ne="$N_EXPERT" '
    (NF != 3 && NF != 4) || $1 !~ /^[0-9]+$/ || $2 !~ /^[0-9]+$/ || $3 !~ /^[0-9]+$/ || (NF == 4 && $4 !~ /^[0-9]+$/) { print "malformed: " $0; bad = 1; next }
    ($1 >= nl || $2 >= ne)                                   { print "out of range: " $0; bad = 1 }
    END { exit bad ? 1 : 0 }
' "$ROWS"; then
    echo "error: routing counts do not fit $N_LAYER layers x $N_EXPERT experts / 计数行不合法或越界" >&2
    exit 6
fi

# a (layer, expert) pair belongs to exactly one device; a repeat means the split is not
# what the log says, so keep the larger count and tell the user
# 同一个专家只应属于一张卡，重复说明拆分与日志不符，取较大值并提示
DUP=$(awk '{ k = $1 SUBSEP $2; if (k in best) dup = 1; if ($3 > best[k]) best[k] = $3 } END { print dup + 0 }' "$ROWS")
if [[ "$DUP" == "1" ]]; then
    echo "warning: an expert was counted on more than one device, kept the larger count" \
         / "警告：有专家被多张卡各记了一次，已取较大值" >&2
fi

# 2.0.0: the tool writes a 4th column "launches" (how many launches routed to the expert);
# keep the larger value of each column and carry it through / 第 4 列 launches 同样取较大值并带到画像里
awk '{ k = $1 SUBSEP $2; if ($3 > best[k]) best[k] = $3; if (NF == 4) { has4 = 1; if ($4 > bl[k]) bl[k] = $4 } }
     END { for (k in best) { split(k, p, SUBSEP); if (has4) print p[1], p[2], best[k], bl[k] + 0; else print p[1], p[2], best[k] } }' \
    "$ROWS" | sort -k1,1n -k2,2n > "$SORTED"

N_ROW=$(( N_LAYER * N_EXPERT ))
N_PAIR=$(grep -c '^[0-9]' "$SORTED" || true)
if (( N_PAIR < N_ROW )); then
    echo "note: $N_PAIR of $N_ROW experts were routed, the rest are written as 0" \
         / " $N_PAIR/$N_ROW 个专家被路由到，其余按 0 写入"
fi

# --- write the profile / 写出画像 ---

# the tool leaves out experts it never routed; a dense profile makes the row count
# checkable and keeps one line per expert / 工具不写没被路由到的专家，这里补齐成稠密画像
# build it aside and move it into place once the row count passes, so a failed check
# cannot leave a half written profile behind / 先写到临时文件，校验通过再落位
{
    if awk 'NF == 4 { f = 1 } END { exit f ? 0 : 1 }' "$SORTED"; then
        echo "# expert routing profile: layer expert count launches"
    else
        echo "# expert routing profile: layer expert count"
    fi
    echo "# generated by $PROGNAME from $(basename -- "$CORPUS")"
    cat "$SORTED"
} | awk -v nl="$N_LAYER" -v ne="$N_EXPERT" '
    BEGIN { for (i = 0; i < nl; i++) for (j = 0; j < ne; j++) { c[i, j] = 0; l[i, j] = 0 } }
    $1 ~ /^#/ { if ($0 ~ /launches/) has4 = 1; print; next }
    { c[$1, $2] = $3; if (NF == 4) l[$1, $2] = $4 }
    END { for (i = 0; i < nl; i++) for (j = 0; j < ne; j++) { if (has4) print i, j, c[i, j], l[i, j]; else print i, j, c[i, j] } }
' > "$WORK_DIR/profile"

N_WRITTEN=$(grep -c '^[0-9]' "$WORK_DIR/profile" || true)
if (( N_WRITTEN != N_ROW )); then
    echo "error: wrote $N_WRITTEN rows, expected $N_LAYER x $N_EXPERT = $N_ROW" \
         / " 写出行数 $N_WRITTEN，应为 $N_ROW" >&2
    exit 6
fi

mv -f -- "$WORK_DIR/profile" "$OUT"

# --- summary / 摘要 ---

echo
echo "profile  : $OUT" / 画像路径
echo "rows     : $N_WRITTEN = $N_LAYER x $N_EXPERT / 行数 = 层数 x 专家数"

awk -v counts="$WORK_DIR/counts" '
    $1 !~ /^#/ {
        v = $3
        total += v
        print v > counts
        if (lo[$1] == "" || v < lo[$1]) lo[$1] = v
        if (v > hi[$1])        hi[$1] = v
    }
    END {
        lmin = ""; lmax = ""
        for (l in lo) {
            if (lmin == "" || lo[l] < lmin) lmin = lo[l]
            if (lmax == "" || hi[l] > lmax) lmax = hi[l]
        }
        printf "%d %d %d\n", total, lmin, lmax
    }
' "$OUT" > "$WORK_DIR/stats"
sort -n "$WORK_DIR/counts" > "$WORK_DIR/counts.sorted"

read -r TOTAL LAYER_MIN LAYER_MAX < "$WORK_DIR/stats"
N_SEEN=$(grep -c '^[0-9]' "$WORK_DIR/counts.sorted" || true)

quantile() { # quantile <p> -> the value at quantile p of the sorted counts / 取该分位数的值
    awk -v p="$1" -v n="$N_SEEN" 'NR == int(p * n + 0.999999) { print; exit }' "$WORK_DIR/counts.sorted"
}

echo "total hits      : $TOTAL / 路由总次数"
echo "experts routed  : $N_SEEN / $N_ROW / 被路由到的专家数"
echo "per-layer min   : $LAYER_MIN / 每层命中最少的专家"
echo "per-layer max   : $LAYER_MAX / 每层命中最多的专家"
echo "p50 / p90 / p99 : $(quantile 0.50) / $(quantile 0.90) / $(quantile 0.99) / 分位数"

exit 0