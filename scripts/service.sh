#!/usr/bin/env bash
#
# 启动 / 关闭 RSI-Jev 服务（本机默认：MLX 后端的 v6.1-VL 4B 8-bit）。
#
#   ./scripts/service.sh                         # 后台启动，等 /health 通了才返回
#   ./scripts/service.sh --port 8001 --effort high
#   ./scripts/service.sh -f                      # 前台运行，Ctrl-C 退出
#   ./scripts/service.sh status                  # 查看是否在跑
#   ./scripts/service.sh logs                    # 跟踪日志
#   ./scripts/service.sh stop                    # 停止（本脚本启动的服务）
#   ./scripts/service.sh stop --force            # 停止端口上任何 rsi-jev 服务，
#                                                #   包括不是本脚本启动的
#
# 环境变量（均可被同名命令行参数覆盖）：
#   MODEL   模型，别名或 HF repo id        默认 v6.1-vl-4b-mlx-8bit
#   ACCEPT  请求体里额外接受的名字（逗号分隔），默认同 MODEL；
#           置空则只认别名 jev-latest 和正式名 rsi-jev-<model>
#   BACKEND 后端，mlx 或 torch             默认 mlx
#   HOST    监听地址                       默认 127.0.0.1
#   PORT    监听端口                       默认 8000
#   PY      python 解释器                  默认取 /opt/anaconda3/bin/python
#   LOG     日志文件                       默认 /tmp/rsi-jev-<port>.log
#   WAIT    等待就绪的秒数                 默认 300
#
# 其余不认识的参数原样透传给 `rsi-jev serve`，例如 --effort high --api-key xxx。
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

MODEL="${MODEL:-v6.1-vl-4b-mlx-8bit}"
ACCEPT="${ACCEPT-$MODEL}"     # 注意是 `-` 不是 `:-`：ACCEPT="" 表示不要额外名字
BACKEND="${BACKEND:-mlx}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
PY="${PY:-/opt/anaconda3/bin/python}"
WAIT="${WAIT:-300}"

[[ -x "$PY" ]] || PY="$(command -v python3)"
LOG="${LOG:-/tmp/rsi-jev-${PORT}.log}"
PIDFILE="${PIDFILE:-/tmp/rsi-jev-${PORT}.pid}"

FOREGROUND=0
PASSTHROUGH=()

# ---------- 日志 ----------
log()  { printf '%s\n' "$*"; }
die()  { printf '错误：%s\n' "$*" >&2; exit 1; }

# ---------- 辅助 ----------
runner_running() {   # 返回 0 表示 PIDFILE 里的进程还活着
    [[ -f "$PIDFILE" ]] || return 1
    local pid
    pid="$(cat "$PIDFILE" 2>/dev/null || true)"
    [[ -n "$pid" ]] || return 1
    kill -0 "$pid" 2>/dev/null
}

port_holder() {      # 打印占用 PORT 的进程，没有则退出码 1
    lsof -nP -iTCP:"$PORT" -sTCP:LISTEN 2>/dev/null | sed -n '2,$p'
}

port_holder_pid() {  # 取占用者的第一个 pid
    port_holder | awk 'NR==1{print $2}'
}

is_jev_process() {   # 判定 pid 是不是本项目的服务，避免误杀别人的进程
    local cmd
    cmd="$(ps -o command= -p "$1" 2>/dev/null || true)"
    [[ "$cmd" == *"serve.py"* || "$cmd" == *"serve/server.py"* || "$cmd" == *"rsi-jev serve"* ]]
}

health_ok() {
    curl -sf -o /dev/null --max-time 2 "http://${HOST}:${PORT}/health"
}

# ---------- 子命令 ----------
cmd_status() {
    if runner_running; then
        log "运行中：pid $(cat "$PIDFILE") · http://${HOST}:${PORT} · 日志 $LOG"
        health_ok && log "健康检查：通过" || log "健康检查：未通过（可能仍在加载）"
    else
        log "未运行（pid 文件 ${PIDFILE}）"
        local holder
        if holder="$(port_holder)"; then
            printf '注意：端口 %s 仍被占用，但不是本脚本启动的：\n%s\n' "$PORT" "$holder"
        fi
        return 1
    fi
}

cmd_logs() {
    [[ -f "$LOG" ]] || die "日志不存在：$LOG"
    tail -f "$LOG"
}

cmd_stop() {
    local force="${1:-0}" pid
    if ! runner_running; then
        # 没有 pid 文件，但端口被占：多半是别处启动的（例如直接 rsi-jev serve）。
        # 这种情况不擅自 kill，先报出来；确认是本项目服务后再 --force。
        [[ -f "$PIDFILE" ]] && mv "$PIDFILE" "$PIDFILE.stale"
        local holder
        if holder="$(port_holder)"; then
            pid="$(port_holder_pid)"
            printf '没有 pid 文件（%s），但端口 %s 被占用：\n%s\n' \
                   "$PIDFILE" "$PORT" "$holder" >&2
            if ! is_jev_process "$pid"; then
                printf '这不是本项目的服务进程，脚本不做处理，请自行确认。\n' >&2
                return 1
            fi
            if [[ "$force" != 1 ]]; then
                printf '确认要停它，加 --force：%s stop --force\n' "$0" >&2
                return 1
            fi
            log "强制停止 pid ${pid} …"
        else
            log "没有在跑的服务"
            return 0
        fi
    else
        pid="$(cat "$PIDFILE")"
        log "停止 pid $pid …"
        [[ "$force" == 1 ]] && log "（--force 已给出，但进程本来就是本脚本启动的）"
    fi
    kill "$pid" 2>/dev/null || true
    for _ in $(seq 1 50); do          # 最多等 5 秒
        kill -0 "$pid" 2>/dev/null || break
        sleep 0.1
    done
    if kill -0 "$pid" 2>/dev/null; then
        log "宽限期结束，强制结束"
        kill -9 "$pid" 2>/dev/null || true
    fi
    mv "$PIDFILE" "$PIDFILE.stale" 2>/dev/null || true
    log "已停止"
}

cmd_start() {
    runner_running && die "服务已在运行（pid $(cat "$PIDFILE")），先执行 stop"
    if port_holder >/dev/null; then
        printf '错误：端口 %s 已被占用：\n' "$PORT" >&2
        port_holder >&2
        exit 1
    fi
    [[ -d "$ROOT/rsijev" ]] || die "找不到 $ROOT/rsijev，请在仓库内运行本脚本"

    local -a cmd=("$PY" "$ROOT/scripts/serve.py" --ckpt "$MODEL"
                  --backend "$BACKEND" --host "$HOST" --port "$PORT")
    if [[ -n "$ACCEPT" ]]; then
        local name
        local -a names=()
        IFS=',' read -r -a names <<<"$ACCEPT"
        for name in "${names[@]}"; do
            [[ -n "$name" ]] && cmd+=(--accept-model "$name")
        done
    fi
    [[ ${#PASSTHROUGH[@]} -gt 0 ]] && cmd+=("${PASSTHROUGH[@]}")

    log "模型   $MODEL"
    log "后端   $BACKEND"
    log "地址   http://${HOST}:${PORT}"
    log "日志   $LOG"
    log "命令   ${cmd[*]}"
    log ""

    if [[ "$FOREGROUND" == 1 ]]; then
        exec "${cmd[@]}"
    fi

    : >"$LOG"                                    # 每次启动清空日志，避免长期堆积
    # stdin 也从 /dev/null 来：否则后台服务会继承调用方的管道，`... | tail` 之类的
    # 管道要等到服务退出才拿到 EOF（实测卡满超时）。
    nohup "${cmd[@]}" </dev/null >>"$LOG" 2>&1 &
    local pid=$!
    printf '%s\n' "$pid" >"$PIDFILE"
    log "已启动，pid ${pid}，等待就绪（最多 ${WAIT}s）…"

    local waited=0
    while (( waited < WAIT )); do
        if ! kill -0 "$pid" 2>/dev/null; then
            printf '\n错误：服务进程已退出，日志末尾：\n' >&2
            tail -n 30 "$LOG" >&2
            exit 1
        fi
        if health_ok; then
            printf '\n就绪：http://%s:%s（耗时 %ss）\n' "$HOST" "$PORT" "$waited"
            log "试试：curl -s http://${HOST}:${PORT}/v1/limits"
            log "界面：python scripts/demo_web.py 或 rsi-jev demo $MODEL"
            return 0
        fi
        sleep 1
        waited=$((waited + 1))
        (( waited % 10 == 0 )) && printf '  已等待 %ss（模型加载 + 预热中）…\n' "$waited"
    done

    printf '\n错误：等待 %ss 仍未就绪，日志末尾：\n' "$WAIT" >&2
    tail -n 30 "$LOG" >&2
    exit 1
}

usage() { sed -n '3,20p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; }

# ---------- 解析参数 ----------
case "${1:-start}" in
    start)  shift || true ;;
    stop)   shift
            case "${1:-}" in
                -f|--force) shift; cmd_stop 1; exit $? ;;
                *)          cmd_stop 0; exit $? ;;
            esac ;;
    status) shift; cmd_status; exit $? ;;
    logs)   shift; cmd_logs; exit $? ;;
    -h|--help) usage; exit 0 ;;
    -*)     ;;                                   # 以选项开头即视为 start 的参数
    *)      die "未知子命令：$1（可用：start / stop / status / logs）" ;;
esac

while [[ $# -gt 0 ]]; do
    case "$1" in
        -f|--foreground) FOREGROUND=1; shift ;;
        -h|--help)       usage; exit 0 ;;
        *)               PASSTHROUGH+=("$1"); shift ;;
    esac
done

cmd_start
