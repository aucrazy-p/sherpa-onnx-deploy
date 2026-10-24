#!/usr/bin/env bash
#
# install.sh — ZipVoice + sherpa-onnx 中文零样本克隆 TTS 一键安装
#
# 目标平台：Linux aarch64（Oracle Ampere A1 实测）/ x86_64
# 做什么：建 venv → 装 sherpa-onnx → 下模型 → 建音色库 → 自检
#
# 用法：
#   bash install.sh                     # 默认装到 ~/sherpa-onnx-deploy，模型走 GitHub
#   bash install.sh --dir /opt/tts      # 自定义目录
#   bash install.sh --source modelscope # 模型走 ModelScope（国内网络更快）
#   bash install.sh --skip-models       # 只装 Python 环境
#   bash install.sh --index-url https://pypi.tuna.tsinghua.edu.cn/simple
#   bash install.sh --no-auto-deps      # 缺系统包时不自动 apt install，直接报错退出
#
# 会自动补装的系统包（缺了才装，有 root 或免密 sudo 时）：
#   python3.X-venv（ensurepip）、curl、bzip2、ca-certificates、tar
#
set -euo pipefail

# ---------------------------------------------------------------- 常量
SHERPA_ONNX_VERSION="${SHERPA_ONNX_VERSION:-1.13.8}"
MODEL_NAME="sherpa-onnx-zipvoice-distill-int8-zh-en-emilia"
VOCOS_NAME="vocos_24khz.onnx"

# 官方 release 地址（已实测 200）
GH_BASE="https://github.com/k2-fsa/sherpa-onnx/releases/download"
GH_MODEL_TAR="$GH_BASE/tts-models/$MODEL_NAME.tar.bz2"
GH_VOCOS="$GH_BASE/vocoder-models/$VOCOS_NAME"        # 注意是 vocoder-models，不是 tts-models

# ModelScope 镜像（国内直连，无需代理）
MS_BASE="https://modelscope.cn/models"
MS_MODEL="$MS_BASE/roarson/$MODEL_NAME/resolve/master"
MS_VOCOS="$MS_BASE/roarson/vocos_24khz.onnx/resolve/master/$VOCOS_NAME"   # repo id 必须带 .onnx

MODEL_TAR_SIZE=109162785
VOCOS_MIN_SIZE=40000000

# ---------------------------------------------------------------- 参数
APP_DIR="$HOME/sherpa-onnx-deploy"
PYTHON_BIN="${PYTHON_BIN:-python3}"
SOURCE="github"
SKIP_MODELS=0
INDEX_URL="${PIP_INDEX_URL:-}"
AUTO_DEPS="${AUTO_DEPS:-1}"

while [ $# -gt 0 ]; do
  case "$1" in
    --dir)         APP_DIR="$2"; shift 2 ;;
    --python)      PYTHON_BIN="$2"; shift 2 ;;
    --source)      SOURCE="$2"; shift 2 ;;
    --index-url)   INDEX_URL="$2"; shift 2 ;;
    --skip-models) SKIP_MODELS=1; shift ;;
    --no-auto-deps) AUTO_DEPS=0; shift ;;
    -h|--help)     sed -n '2,18p' "$0"; exit 0 ;;
    *) echo "未知参数: $1（-h 查看用法）" >&2; exit 2 ;;
  esac
done

case "$SOURCE" in github|modelscope) ;; *) echo "--source 只能是 github 或 modelscope" >&2; exit 2 ;; esac

log()  { printf '\033[1;32m[install]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[warn]\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31m[error]\033[0m %s\n' "$*" >&2; exit 1; }
step() { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }

# ---------------------------------------------------------------- 预检
step "预检"

[ "$(uname -s)" = "Linux" ] || warn "当前不是 Linux（$(uname -s)），本脚本按 Linux 编写，可能失败"
ARCH="$(uname -m)"
log "架构: $ARCH   内核: $(uname -r)"

case "$ARCH" in
  aarch64|arm64|x86_64|amd64) ;;
  *) die "不支持的架构: $ARCH（sherpa-onnx 只提供 aarch64 / x86_64 wheel）" ;;
esac

command -v "$PYTHON_BIN" >/dev/null 2>&1 || die "找不到 $PYTHON_BIN，请先 apt install python3"

PY_VER="$("$PYTHON_BIN" -c 'import sys;print("%d.%d"%sys.version_info[:2])')"
PY_FULL="$("$PYTHON_BIN" -c 'import sys;print(sys.version.split()[0])')"
log "Python: $PY_FULL ($(command -v "$PYTHON_BIN"))"

"$PYTHON_BIN" - "$PY_VER" <<'PY' || exit 1
import sys
maj, minr = map(int, sys.argv[1].split("."))
if (maj, minr) < (3, 8):
    sys.exit("Python 版本过低（%s），sherpa-onnx 需要 >= 3.8" % sys.argv[1])
if (maj, minr) >= (3, 14):
    sys.exit("Python %s 太新，sherpa-onnx %s 大概率没有对应 wheel；请改用 python3.10/3.12" % (sys.argv[1], "1.13.8"))
if (maj, minr) == (3, 13):
    print("[warn] Python 3.13 的 sherpa-onnx wheel 不一定齐全，装不上就换 3.12", file=sys.stderr)
PY

# ---------------------------------------------------------------- 系统依赖
# 目标：用户在干净系统上只需要 bash install.sh 一条命令，不手动 apt install。
step "系统依赖"

PM=""
for c in apt-get dnf yum apk; do
  if command -v "$c" >/dev/null 2>&1; then
    case "$c" in apt-get) PM=apt ;; *) PM="${c}" ;; esac
    break
  fi
done
log "包管理器: ${PM:-未识别}   身份: $( [ "$(id -u)" = 0 ] && echo root || echo "uid=$(id -u)" )"

run_root() {
  if [ "$(id -u)" = "0" ]; then
    "$@"
  elif command -v sudo >/dev/null 2>&1 && sudo -n true 2>/dev/null; then
    sudo -n "$@"
  else
    return 127
  fi
}

# apt 有 dpkg 锁时需要重试（云镜像首次开机常有 unattended-upgrade 在跑）
pkg_install() {
  [ $# -gt 0 ] || return 0
  local tries=0 rc=0
  while [ "$tries" -lt 3 ]; do
    tries=$((tries + 1))
    rc=0
    case "$PM" in
      apt)
        # NEEDRESTART_MODE=a：Ubuntu 22.04+ 装了 needrestart 后 apt 会弹交互，非交互环境会卡住
        run_root env DEBIAN_FRONTEND=noninteractive NEEDRESTART_MODE=a \
          apt-get update -qq || warn "apt-get update 未成功，尝试用已有索引继续"
        run_root env DEBIAN_FRONTEND=noninteractive NEEDRESTART_MODE=a \
          apt-get install -y -qq --no-install-recommends "$@" || rc=$?
        ;;
      dnf) run_root dnf install -y -q "$@" || rc=$? ;;
      yum) run_root yum install -y -q "$@" || rc=$? ;;
      apk) run_root apk add --no-cache "$@" || rc=$? ;;
      *)   rc=127 ;;
    esac
    [ "$rc" = 0 ] && return 0
    [ "$rc" = 127 ] && return 127
    [ "$tries" -lt 3 ] && { warn "安装失败（第 $tries 次），3 秒后重试：$*"; sleep 3; }
  done
  return "$rc"
}

DEPS=()
NEED_VENV=0
if ! "$PYTHON_BIN" -c 'import ensurepip' >/dev/null 2>&1; then
  NEED_VENV=1
  case "$PM" in
    apt) DEPS+=("python${PY_VER}-venv") ;;
    *)   DEPS+=(python3-venv) ;;
  esac
fi
command -v curl >/dev/null 2>&1 || DEPS+=(curl)
command -v tar  >/dev/null 2>&1 || DEPS+=(tar)
if [ "$SKIP_MODELS" = "0" ]; then
  command -v bzip2 >/dev/null 2>&1 || DEPS+=(bzip2)     # 解 tar.bz2 必需
fi
case "$PM" in
  apt) [ -s /etc/ssl/certs/ca-certificates.crt ] || DEPS+=(ca-certificates) ;;
  apk) [ -s /etc/ssl/certs/ca-certificates.crt ] || DEPS+=(ca-certificates) ;;
esac

if [ ${#DEPS[@]} -eq 0 ]; then
  log "系统依赖齐全，无需安装"
else
  log "缺失: ${DEPS[*]}"
  if [ "$AUTO_DEPS" != "1" ]; then
    die "缺少系统包 ${DEPS[*]}；去掉 --no-auto-deps 让脚本自动安装，或手动装"
  fi
  if [ -z "$PM" ]; then
    die "无法识别包管理器（apt/dnf/yum/apk 都没有），请手动安装: ${DEPS[*]}"
  fi

  # 权限问题先判，否则会被后面"换包名重试"的失败信息掩盖成"包装不上"
  if [ "$(id -u)" != "0" ]; then
    if ! command -v sudo >/dev/null 2>&1; then
      die "当前非 root 且没有 sudo，无法自动装包。请用 root 运行，或手动安装: ${DEPS[*]}"
    elif ! sudo -n true 2>/dev/null; then
      die "sudo 需要输入密码（脚本非交互）。请用 root 运行，或先手动执行 sudo -v 再重跑。缺失: ${DEPS[*]}"
    fi
  fi

  if ! pkg_install "${DEPS[@]}"; then
    # apt 上 python3.X-venv 有时没有对应版本（如自编译 Python），退回通用名再试一次
    HAS_PY=0
    for d in "${DEPS[@]}"; do case "$d" in python*-venv) HAS_PY=1 ;; esac; done
    if [ "$PM" = "apt" ] && [ "$HAS_PY" = "1" ]; then
      DEPS2=()
      for d in "${DEPS[@]}"; do
        case "$d" in python*-venv) DEPS2+=(python3-venv python3-pip) ;; *) DEPS2+=("$d") ;; esac
      done
      warn "python${PY_VER}-venv 装不上，改用通用包名重试：${DEPS2[*]}"
      pkg_install "${DEPS2[@]}" || die "系统依赖安装失败。请手动执行：apt-get install -y ${DEPS2[*]}"
    else
      die "系统依赖安装失败，请手动执行后重跑。缺失: ${DEPS[*]}"
    fi
  fi

  # 装完必须真的可用，否则早失败好过晚失败
  if [ "$NEED_VENV" = "1" ] && ! "$PYTHON_BIN" -c 'import ensurepip' >/dev/null 2>&1; then
    die "已安装 venv 相关包，但 $PYTHON_BIN 仍无法使用 ensurepip；请确认包名与 Python 版本匹配（当前 $PY_FULL）"
  fi
  command -v curl >/dev/null 2>&1 || die "curl 仍未安装成功"
  log "系统依赖已就绪"
fi

FREE_MB="$(df -Pm "$(dirname "$APP_DIR")" 2>/dev/null | awk 'NR==2{print $4}')"
if [ -n "${FREE_MB:-}" ] && [ "$FREE_MB" -lt 3000 ]; then
  die "$(dirname "$APP_DIR") 剩余空间只有 ${FREE_MB}MB，至少需要 3GB"
fi

mkdir -p "$APP_DIR"
APP_DIR="$(cd "$APP_DIR" && pwd)"
VENV="$APP_DIR/venv"
MODELS="$APP_DIR/models"
VOICES="$APP_DIR/voices"
CACHE="$APP_DIR/cache"
log "安装目录: $APP_DIR"

# ---------------------------------------------------------------- Python 环境
step "创建 venv 并安装依赖"

# 上一次失败可能留下半成品 venv（有 bin/python 但没有 pip），必须先识别再重建，
# 否则 `[ ! -x $VENV/bin/python ]` 会判定"已存在"而跳过，后面 pip 直接报 No module named pip。
if [ -d "$VENV" ] && [ "$VENV" != "$APP_DIR" ] && [ "$(basename "$VENV")" = "venv" ] \
   && [ -x "$VENV/bin/python" ] && ! "$VENV/bin/python" -m pip --version >/dev/null 2>&1; then
  warn "发现上次中断留下的不完整 venv（pip 不可用），删除重建"
  rm -rf "$VENV"
fi

if [ ! -x "$VENV/bin/python" ]; then
  "$PYTHON_BIN" -m venv "$VENV" || die "venv 创建失败（请确认已安装 python${PY_VER}-venv）"
fi
"$VENV/bin/python" -m pip --version >/dev/null 2>&1 \
  || die "venv 里没有可用的 pip（$VENV）。删掉 $VENV 后重跑本脚本即可"
# shellcheck disable=SC1091
. "$VENV/bin/activate"

PIP_ARGS=()
if [ -n "$INDEX_URL" ]; then PIP_ARGS+=(-i "$INDEX_URL"); fi

python -m pip install -q --upgrade pip ${PIP_ARGS[@]+"${PIP_ARGS[@]}"}
log "安装 sherpa-onnx==$SHERPA_ONNX_VERSION numpy soundfile fastapi uvicorn"
python -m pip install -q \
  "sherpa-onnx==$SHERPA_ONNX_VERSION" \
  "numpy>=1.24" \
  "soundfile>=0.12" \
  "fastapi>=0.110" \
  "uvicorn>=0.29" \
  ${PIP_ARGS[@]+"${PIP_ARGS[@]}"} || die "pip 安装失败；若报 'No matching distribution'，多半是 Python 版本太新，换 python3.12 重来"

python - <<'PY'
import sherpa_onnx, numpy, soundfile, fastapi, uvicorn
print("[install] sherpa-onnx =", sherpa_onnx.__version__ if hasattr(sherpa_onnx, "__version__") else "?")
print("[install] onnxruntime 可用:", end=" ")
try:
    import onnxruntime; print(onnxruntime.__version__)
except Exception as e:
    print("否 ->", e)
PY

# ---------------------------------------------------------------- 脚本就位
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ "$SCRIPT_DIR" != "$APP_DIR" ]; then
  for f in engine.py smoke_test.py; do
    [ -f "$SCRIPT_DIR/$f" ] || die "$SCRIPT_DIR 下没有 $f（install.sh 要和这两个文件放在一起）"
    cp -f "$SCRIPT_DIR/$f" "$APP_DIR/$f"
  done
  log "已把 engine.py / smoke_test.py 复制到 $APP_DIR"
fi

# ---------------------------------------------------------------- 模型
if [ "$SKIP_MODELS" = "0" ]; then
  step "下载模型（$( [ "$SOURCE" = github ] && echo 'GitHub' || echo 'ModelScope' )）"
  mkdir -p "$MODELS"
  cd "$MODELS"

  # fetch <url> <输出文件> <校验方式: tar|size|none>   失败返回非 0，不直接退出
  fetch() {
    local url="$1" out="$2" mode="${3:-none}"
    if [ -s "$out" ]; then
      log "已存在，跳过: $out"
    else
      log "下载 $(basename "$out")  <-  $url"
      if ! curl -fL --retry 3 --retry-delay 2 --connect-timeout 20 -# -o "$out.part" "$url"; then
        rm -f "$out.part"
        warn "下载失败: $url"
        return 1
      fi
      mv "$out.part" "$out"
    fi
    case "$mode" in
      tar)  tar -tjf "$out" >/dev/null 2>&1 || { warn "$out 不是有效的 tar.bz2（可能不完整），已删除"; rm -f "$out"; return 1; } ;;
      size) [ "$(stat -c %s "$out")" -ge "$VOCOS_MIN_SIZE" ] || { warn "$out 只有 $(stat -c %s "$out") 字节，明显不对（vocoder 约 51MB），已删除"; rm -f "$out"; return 1; } ;;
    esac
    return 0
  }

  if [ "$SOURCE" = "github" ]; then
    fetch "$GH_MODEL_TAR" "$MODEL_NAME.tar.bz2" tar || die "模型包下载失败，可换源重试：bash install.sh --source modelscope"
    if [ ! -d "$MODEL_NAME" ]; then
      log "解压 $MODEL_NAME.tar.bz2"
      tar -xjf "$MODEL_NAME.tar.bz2"
      rm -f "$MODEL_NAME.tar.bz2"
    fi
    fetch "$GH_VOCOS" "$VOCOS_NAME" size || die "vocoder 下载失败（缺它就合成不出来）"
    # 注意：不要再下 lexicon-en-us.txt。那是 kokoro 的 en-us 音素词典，token 集和 ZipVoice 的
    # tokens.txt 对不上，塞进去只会让 sherpa-onnx 在加载时刷几万行 "Unknown token"。
    # ZipVoice 用模型包里自带的 lexicon.txt + espeak-ng-data，解压后就在 $MODEL_NAME/ 下。
  else
    mkdir -p "$MODEL_NAME/test_wavs"
    for f in encoder.int8.onnx decoder.int8.onnx tokens.txt lexicon.txt \
             test_wavs/news-female.wav test_wavs/news-female-2.wav \
             test_wavs/leijun-1.wav test_wavs/prompt.txt; do
      fetch "$MS_MODEL/$f" "$MODEL_NAME/$f" none || die "下载失败: $f（可换源：bash install.sh --source github）"
    done
    fetch "$MS_VOCOS" "$VOCOS_NAME" size || die "vocoder 下载失败（缺它就合成不出来）"
  fi

  step "校验模型文件"
  MODEL_DIR="$MODELS/$MODEL_NAME"
  MISSING=0
  for f in "$MODEL_DIR/tokens.txt" "$MODEL_DIR/encoder.int8.onnx" "$MODEL_DIR/decoder.int8.onnx" "$MODELS/$VOCOS_NAME"; do
    if [ -s "$f" ]; then
      log "OK  $(du -h "$f" | cut -f1)  ${f#"$MODELS"/}"
    else
      warn "缺失 $f"; MISSING=1
    fi
  done
  # 词典 + espeak 数据：英文要靠它们音素化，缺了英文会读成逐字母或直接跳过（中文不受影响）
  for f in "$MODEL_DIR/lexicon.txt"; do
    [ -s "$f" ] && log "OK  $(du -h "$f" | cut -f1)  ${f#"$MODELS"/}" \
      || warn "缺 lexicon.txt（模型包里自带，检查解压是否完整）"
  done
  if [ -d "$MODEL_DIR/espeak-ng-data" ]; then
    log "OK  espeak-ng-data/ （$(ls "$MODEL_DIR/espeak-ng-data" | wc -l) 项）"
  else
    warn "缺 espeak-ng-data/：英文没有音素化兜底，会退化成逐字母读。用 github 源重装可补齐"
  fi
  [ "$MISSING" = "0" ] || die "模型文件不全，重跑 install.sh（已下载的会跳过）"
else
  warn "已跳过模型下载（--skip-models）"
fi

# ---------------------------------------------------------------- 音色库
step "初始化音色库 voices/"

mkdir -p "$VOICES" "$CACHE"
python - "$MODELS/$MODEL_NAME" "$VOICES" <<'PY'
import shutil, sys
from pathlib import Path

model_dir, voices_dir = Path(sys.argv[1]), Path(sys.argv[2])
prompt = model_dir / "test_wavs" / "prompt.txt"
if not prompt.is_file():
    print("[warn] 没找到 test_wavs/prompt.txt，跳过内置音色")
    sys.exit(0)

n = 0
for line in prompt.read_text(encoding="utf-8").splitlines():
    line = line.strip()
    if not line:
        continue
    name, _, text = line.partition(" ")
    wav = model_dir / "test_wavs" / name
    if not wav.is_file():
        print(f"[warn] 缺参考音频 {wav}")
        continue
    dst_wav = voices_dir / name
    dst_txt = dst_wav.with_suffix(".txt")
    if not dst_wav.exists():
        shutil.copy2(wav, dst_wav)
    if not dst_txt.exists():
        dst_txt.write_text(text.strip() + "\n", encoding="utf-8")
    n += 1
print(f"[install] 内置音色 {n} 个 -> {voices_dir}")
PY

# ---------------------------------------------------------------- 自检
step "自检（真正加载模型，能暴露 vocoder 不匹配等问题）"
python "$APP_DIR/engine.py" --check \
  --model-dir "$MODELS/$MODEL_NAME" \
  --vocoder "$MODELS/$VOCOS_NAME" \
  --voices-dir "$VOICES"

# ---------------------------------------------------------------- 收尾
cat <<EOF

$(printf '\033[1;32m安装完成\033[0m')

目录结构
  $APP_DIR/engine.py       TTS 服务
  $APP_DIR/smoke_test.py   冒烟测试
  $MODELS/                 模型
  $VOICES/                 音色库（*.wav + *.txt 同名成对）

下一步
  1) 起服务（前台，先看日志）
       cd $APP_DIR && venv/bin/python engine.py --port 8080

  2) 另开一个终端跑冒烟测试（约 2-4 分钟，会打印真实 RTF 并把 wav 存到 smoke_out/）
       cd $APP_DIR && venv/bin/python smoke_test.py

  3) 把 smoke_out/ 里的 wav 拉回本地听音质
       scp -r <服务器>:~$(basename "$APP_DIR")/smoke_out ./

  挂 systemd / 加 WebUI / 接 Legado 都等听完再决定。
EOF
