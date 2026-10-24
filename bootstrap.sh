#!/usr/bin/env bash
#
# bootstrap.sh — 服务器端一把梭：预检 → 装 → 起服务 → 冒烟测试 → 打包结果
#
# 用法（在仓库目录里）：
#   bash bootstrap.sh                 # 全流程，跑完把 tar.gz 拉回本地听
#   bash bootstrap.sh --keep          # 跑完不关服务，留着给 Legado / curl 用
#   bash bootstrap.sh --port 8090     # 换端口
#   bash bootstrap.sh --reinstall     # 强制重跑 install.sh（幂等，已下载的会跳过）
#   bash bootstrap.sh --skip-smoke    # 只装和起服务
#
# 产物：
#   $APP_DIR/bootstrap-<时间>.tar.gz   含 smoke_out/*.wav + logs/engine.log
#
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

APP_DIR="${APP_DIR:-$HOME/sherpa-onnx-deploy}"
PORT="${PORT:-8080}"
MODEL_NAME="sherpa-onnx-zipvoice-distill-int8-zh-en-emilia"
VOCOS_NAME="vocos_24khz.onnx"

FORCE_INSTALL=0
KEEP=0
SKIP_SMOKE=0
PYTHON_BIN="${PYTHON_BIN:-}"

while [ $# -gt 0 ]; do
  case "$1" in
    --port)        PORT="$2"; shift 2 ;;
    --dir)         APP_DIR="$2"; shift 2 ;;
    --python)      PYTHON_BIN="$2"; shift 2 ;;
    --reinstall)   FORCE_INSTALL=1; shift ;;
    --keep)        KEEP=1; shift ;;
    --skip-smoke)  SKIP_SMOKE=1; shift ;;
    -h|--help)     sed -n '2,20p' "$0"; exit 0 ;;
    *) echo "未知参数: $1（-h 查看用法）" >&2; exit 2 ;;
  esac
done

step() { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }
log()  { printf '\033[1;32m[boot]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[warn]\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31m[error]\033[0m %s\n' "$*" >&2; exit 1; }

ENGINE_PID=""
cleanup() {
  if [ -n "$ENGINE_PID" ] && kill -0 "$ENGINE_PID" 2>/dev/null; then
    if [ "$KEEP" = "1" ]; then
      log "服务仍在运行（pid $ENGINE_PID，端口 $PORT）—— --keep 生效"
    else
      kill "$ENGINE_PID" 2>/dev/null || true
      log "已停止测试用的服务进程"
    fi
  fi
}
trap cleanup EXIT

# ---------------------------------------------------------------- 0. 预检
step "环境预检"
ARCH="$(uname -m)"
[ "$ARCH" = "aarch64" ] || [ "$ARCH" = "x86_64" ] || warn "架构 $ARCH 不在实测范围内，可能没有预编译 wheel"
log "架构 $ARCH  |  $(nproc) 核  |  内存 $(free -m | awk 'NR==2{printf "%.1fG", $2/1024}')"

# Python 版本：sherpa-onnx 目前只到 cp312，3.13 没有 wheel
if [ -z "$PYTHON_BIN" ]; then
  for c in python3.12 python3.11 python3.10; do
    command -v "$c" >/dev/null 2>&1 && { PYTHON_BIN="$c"; break; }
  done
fi

# 系统只有 3.13/3.14 时自动补一个 3.12（sherpa-onnx 没有 3.13 的 wheel）
if [ -z "$PYTHON_BIN" ] && command -v apt-get >/dev/null 2>&1; then
  if [ "$(id -u)" = "0" ] || { command -v sudo >/dev/null 2>&1 && sudo -n true 2>/dev/null; }; then
    warn "没找到 python3.10~3.12，尝试自动安装 python3.12（约 30MB）"
    APT=(apt-get)
    [ "$(id -u)" != "0" ] && APT=(sudo -n apt-get)
    "${APT[@]}" update -qq || warn "apt-get update 未成功，用已有索引继续"
    "${APT[@]}" install -y -qq --no-install-recommends python3.12 python3.12-venv \
      || warn "自动安装 python3.12 失败"
    for c in python3.12 python3.11 python3.10; do
      command -v "$c" >/dev/null 2>&1 && { PYTHON_BIN="$c"; break; }
    done
  fi
fi
[ -n "$PYTHON_BIN" ] || die "找不到 python3.10~3.12。sherpa-onnx 没有 3.13 的 wheel，请先装：apt install -y python3.12 python3.12-venv"
PYVER="$("$PYTHON_BIN" -c 'import sys;print("%d.%d"%sys.version_info[:2])')"
case "$PYVER" in
  3.10|3.11|3.12) log "Python $PYVER ($PYTHON_BIN)" ;;
  *) die "Python $PYVER 没有对应的 sherpa-onnx wheel，改用 python3.12" ;;
esac

FREE_GB="$(df -Pm "$HOME" | awk 'NR==2{printf "%d", $4/1024}')"
[ "${FREE_GB:-0}" -ge 8 ] || die "家目录只剩 ${FREE_GB}GB，模型+依赖约需 4GB，先清点空间"
log "磁盘剩余 ${FREE_GB}GB"

if command -v timedatectl >/dev/null 2>&1; then
  if ! timedatectl status 2>/dev/null | grep -qi "synchronized: yes"; then
    warn "系统时钟未同步。edge-tts 会因此报 403（zipvoice 不受影响）。建议：sudo timedatectl set-ntp true"
  fi
fi

# ---------------------------------------------------------------- 1. 安装
step "安装（venv + 依赖 + 模型 + 音色库）"
VENV_PY="$APP_DIR/venv/bin/python"
# 注意：不能只看 venv/bin/python 存在就跳过。上次中断会留下"有解释器没有 pip"的半成品，
# 跳过安装后会在后面莫名其妙地失败（No module named pip）。必须验证 pip 真的可用。
if [ -x "$VENV_PY" ] && "$VENV_PY" -m pip --version >/dev/null 2>&1 && [ "$FORCE_INSTALL" = "0" ]; then
  log "已安装过且 venv 可用，跳过。要重装加 --reinstall"
else
  [ -e "$APP_DIR/venv" ] && warn "venv 存在但不可用，将重建"
  bash install.sh --dir "$APP_DIR" --python "$PYTHON_BIN"
fi

[ -x "$VENV_PY" ] || die "$VENV_PY 不存在，安装没成功"
"$VENV_PY" -m pip --version >/dev/null 2>&1 || die "venv 里没有可用的 pip，删掉 $APP_DIR/venv 后重跑"

# ---------------------------------------------------------------- 2. 起服务
step "启动 TTS 服务（端口 $PORT）"
if command -v ss >/dev/null 2>&1 && ss -ltn 2>/dev/null | grep -q ":$PORT "; then
  die "端口 $PORT 已被占用，换一个：bash bootstrap.sh --port 8090"
fi

mkdir -p "$APP_DIR/logs"
nohup "$VENV_PY" "$APP_DIR/engine.py" \
  --port "$PORT" \
  --model-dir "$APP_DIR/models/$MODEL_NAME" \
  --vocoder "$APP_DIR/models/$VOCOS_NAME" \
  --voices-dir "$APP_DIR/voices" \
  --cache-dir "$APP_DIR/cache" \
  > "$APP_DIR/logs/engine.log" 2>&1 &
ENGINE_PID=$!

log "等待 /health（最多 90 秒，首次加载模型慢）"
ok=0
for i in $(seq 1 90); do
  if curl -fsS --max-time 2 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
    ok=1
    log "服务就绪（${i}s）"
    break
  fi
  if ! kill -0 "$ENGINE_PID" 2>/dev/null; then
    echo "----- engine.log 尾部 -----" >&2
    tail -30 "$APP_DIR/logs/engine.log" >&2
    die "服务进程退出了，看上面的日志"
  fi
  sleep 1
done
[ "$ok" = "1" ] || { tail -30 "$APP_DIR/logs/engine.log" >&2; die "90 秒内没起来"; }

curl -fsS "http://127.0.0.1:$PORT/health" && echo
curl -fsS "http://127.0.0.1:$PORT/voices" && echo

# ---------------------------------------------------------------- 3. 冒烟
RESULT_TAR=""
if [ "$SKIP_SMOKE" = "0" ]; then
  step "冒烟测试（2~4 分钟，会打印真实 RTF）"
  set +e
  "$VENV_PY" "$APP_DIR/smoke_test.py" --base-url "http://127.0.0.1:$PORT" --out "$APP_DIR/smoke_out"
  SMOKE_RC=$?
  set -e
  [ "$SMOKE_RC" = "0" ] || warn "冒烟测试返回 $SMOKE_RC，看上面的输出"

  step "打包结果"
  RESULT_TAR="$APP_DIR/bootstrap-$(date +%Y%m%d-%H%M).tar.gz"
  tar czf "$RESULT_TAR" -C "$APP_DIR" smoke_out logs 2>/dev/null || warn "打包失败，可直接 scp smoke_out/"
  log "已打包 $(basename "$RESULT_TAR")  $(du -h "$RESULT_TAR" 2>/dev/null | cut -f1)"
fi

# ---------------------------------------------------------------- 4. 怎么拿回来
PUBIP="$(curl -fsS --max-time 3 https://api.ipify.org 2>/dev/null || echo "<服务器IP>")"
cat <<EOF

$(printf '\033[1;32m全部完成\033[0m')

在你自己电脑上执行（把 ~ 换成实际路径）：
  scp root@$PUBIP:$( [ -n "$RESULT_TAR" ] && echo "~${RESULT_TAR#$HOME}" || echo "~${APP_DIR#$HOME}/smoke_out" ) .

服务信息
  健康检查   http://$PUBIP:$PORT/health
  音色列表   http://$PUBIP:$PORT/voices
  试听一句   http://$PUBIP:$PORT/tts?text=%E4%BD%A0%E5%A5%BD%E4%B8%96%E7%95%8C&voice=<音色名>
  日志       $APP_DIR/logs/engine.log

$( [ "$KEEP" = "1" ] && echo "服务保持运行中，重启机器后不会自动起——要常驻请配 systemd（还没做）。" || echo "测试进程已退出。要长期跑：cd $APP_DIR && nohup venv/bin/python engine.py --port $PORT > logs/engine.log 2>&1 &" )
EOF
