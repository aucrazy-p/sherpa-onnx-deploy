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
GH_LEXICON_EN="$GH_BASE/tts-models/lexicon-en-us.txt"

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

while [ $# -gt 0 ]; do
  case "$1" in
    --dir)         APP_DIR="$2"; shift 2 ;;
    --python)      PYTHON_BIN="$2"; shift 2 ;;
    --source)      SOURCE="$2"; shift 2 ;;
    --index-url)   INDEX_URL="$2"; shift 2 ;;
    --skip-models) SKIP_MODELS=1; shift ;;
    -h|--help)     sed -n '2,20p' "$0"; exit 0 ;;
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

command -v curl >/dev/null 2>&1 || die "缺少 curl，请先 apt install -y curl"
if [ "$SKIP_MODELS" = "0" ] && ! command -v bzip2 >/dev/null 2>&1; then
  die "缺少 bzip2（解 tar.bz2 需要），请先 apt install -y bzip2"
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

if [ ! -x "$VENV/bin/python" ]; then
  "$PYTHON_BIN" -m venv "$VENV" || die "venv 创建失败；Debian/Ubuntu 需要 apt install -y python3-venv"
fi
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
    # 英文字典可选：6MB，缺失只影响英文发音
    fetch "$GH_LEXICON_EN" "lexicon-en-us.txt" none || warn "lexicon-en-us.txt 没下成功，中文不受影响"
  else
    mkdir -p "$MODEL_NAME/test_wavs"
    for f in encoder.int8.onnx decoder.int8.onnx tokens.txt lexicon.txt \
             test_wavs/news-female.wav test_wavs/news-female-2.wav \
             test_wavs/leijun-1.wav test_wavs/prompt.txt; do
      fetch "$MS_MODEL/$f" "$MODEL_NAME/$f" none || die "下载失败: $f（可换源：bash install.sh --source github）"
    done
    fetch "$MS_VOCOS" "$VOCOS_NAME" size || die "vocoder 下载失败（缺它就合成不出来）"
    [ -s "lexicon-en-us.txt" ] || warn "ModelScope 源没有 lexicon-en-us.txt，英文发音会受影响（可另从 GitHub 补）"
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
  --lexicon "$MODELS/lexicon-en-us.txt" \
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
