#!/usr/bin/env python3
"""engine.py — ZipVoice + sherpa-onnx 零样本音色克隆 TTS 服务

音色 = 一对文件：voices/<名字>.wav（参考音频，建议 3~10 秒单人干净人声）
                  voices/<名字>.txt（该 wav 的逐字转写，必须和音频说的一模一样，含标点）
改完文件不用重启，服务会在下次请求时自动重载。

用法
  python engine.py --port 8080                      启服务
  python engine.py --check                          只加载模型 + 试合成一句，自检用
  python engine.py --check --no-synth               只加载模型不合成

接口（给 Legado 用第一个就够）
  GET /tts?text=<要读的文本>&voice=<音色名>&speed=1.0&num_steps=4    -> audio/wav
  GET /voices                                                        音色列表
  GET /health                                                        状态 + 模型信息
  GET /                                                              接口索引

Legado 里填：http://<服务器>:8080/tts?text={{speakText}}&voice=news-female
"""
from __future__ import annotations

import argparse
import hashlib
import io
import logging
import re
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf

import sherpa_onnx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
import uvicorn

LOG = logging.getLogger("tts")

# 句子结束符：在这些位置断句，避免一次喂太长
_SPLIT_RE = re.compile(r"[^。！？!?…；;\n]*[。！？!?…；;\n]+|[^。！？!?…；;\n]+$")
_SUB_SPLIT_RE = re.compile(r"[^，,、：:\s]+[，,、：:\s]*|[，,、：:\s]+")


# --------------------------------------------------------------------------- 文本处理
def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("“", "「").replace("”", "」")
    text = re.sub(r"[ \t\u3000]+", " ", text)
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()


def _hard_split(sent: str, limit: int) -> List[str]:
    """把超长单句在逗号/空格处切开，兜底按字硬切。"""
    out: List[str] = []
    cur = ""
    for piece in _SUB_SPLIT_RE.findall(sent):
        if not piece:
            continue
        if len(cur) + len(piece) <= limit:
            cur += piece
        else:
            if cur.strip():
                out.append(cur.strip())
            cur = piece
            while len(cur) > limit:  # 单段本身超长
                out.append(cur[:limit])
                cur = cur[limit:]
    if cur.strip():
        out.append(cur.strip())
    return [s for s in out if s]


def split_text(text: str, target: int = 50, hard_max: int = 100) -> List[str]:
    """按句切分并合并到 target 字左右，单块不超过 hard_max。"""
    chunks: List[str] = []
    cur = ""
    for sent in _SPLIT_RE.findall(text):
        sent = sent.strip()
        if not sent:
            continue
        if len(sent) > hard_max:
            if cur:
                chunks.append(cur)
                cur = ""
            chunks.extend(_hard_split(sent, hard_max))
            continue
        if not cur:
            cur = sent
        elif len(cur) + len(sent) <= target:
            cur += sent
        else:
            chunks.append(cur)
            cur = sent
    if cur:
        chunks.append(cur)
    return chunks


def to_wav_bytes(samples: np.ndarray, sample_rate: int) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, samples, sample_rate, format="WAV", subtype="PCM_16")
    return buf.getvalue()


# --------------------------------------------------------------------------- 音色库
@dataclass
class Voice:
    name: str
    wav: Path
    text: str
    duration: float = 0.0
    sample_rate: int = 0
    mtime_ns: int = 0


class VoiceRegistry:
    """扫 voices/*.wav + 同名 .txt，目录一变就重载（热加载）。"""

    def __init__(self, directory: Path):
        self.dir = Path(directory)
        self._voices: Dict[str, Voice] = {}
        self._audio: Dict[str, Tuple[int, np.ndarray, int]] = {}
        self._sig: Optional[tuple] = None
        self.rescan(force=True)

    def _signature(self) -> tuple:
        if not self.dir.is_dir():
            return ()
        items = []
        for p in sorted(self.dir.glob("*.wav")):
            try:
                st = p.stat()
            except FileNotFoundError:
                continue
            items.append((p.name, st.st_mtime_ns, st.st_size))
        return tuple(items)

    def rescan(self, force: bool = False) -> bool:
        sig = self._signature()
        if not force and sig == self._sig:
            return False
        voices: Dict[str, Voice] = {}
        for name, mtime_ns, _size in sig:
            wav = self.dir / name
            txt = wav.with_suffix(".txt")
            if not txt.is_file():
                LOG.warning("音色 %s 缺少同名转写 %s，跳过（转写必须和音频内容一致）", name, txt.name)
                continue
            ref_text = txt.read_text(encoding="utf-8").strip()
            if not ref_text:
                LOG.warning("音色 %s 的转写文件是空的，跳过", name)
                continue
            voices[wav.stem] = Voice(name=wav.stem, wav=wav, text=ref_text, mtime_ns=mtime_ns)
        self._voices = voices
        self._audio.clear()
        self._sig = sig
        if voices:
            LOG.info("音色库已加载 %d 个: %s", len(voices), ", ".join(voices))
        else:
            LOG.warning("音色库为空（%s 下没有成对的 .wav/.txt）", self.dir)
        return True

    def names(self) -> List[str]:
        self.rescan()
        return list(self._voices)

    def get(self, name: str) -> Voice:
        self.rescan()
        v = self._voices.get(name)
        if v is None:
            raise KeyError(name)
        return v

    def audio(self, name: str) -> Tuple[np.ndarray, int]:
        v = self.get(name)
        cached = self._audio.get(name)
        if cached and cached[0] == v.mtime_ns:
            return cached[1], cached[2]
        samples, sr = sf.read(str(v.wav), dtype="float32", always_2d=True)
        samples = np.ascontiguousarray(samples[:, 0])  # 只取第一声道
        peak = float(np.abs(samples).max()) if samples.size else 0.0
        if peak < 1e-4:
            raise ValueError(f"参考音频 {v.wav.name} 基本是静音（peak={peak:.5f}）")
        v.duration = len(samples) / sr
        v.sample_rate = sr
        if v.duration > 12:
            LOG.warning("音色 %s 参考音频 %.1f 秒，偏长（官方建议 3~10 秒，过长又慢又糊）", name, v.duration)
        self._audio[name] = (v.mtime_ns, samples, sr)
        return samples, sr


# --------------------------------------------------------------------------- 引擎
@dataclass
class SynthResult:
    wav: bytes
    sample_rate: int
    chars: int
    chunks: int
    audio_seconds: float
    elapsed: float
    cached: bool = False

    @property
    def rtf(self) -> float:
        return self.elapsed / self.audio_seconds if self.audio_seconds > 0 else 0.0


class ZipVoiceEngine:
    def __init__(
        self,
        model_dir: Path,
        vocoder: Path,
        lexicon_extra: Optional[Path] = None,
        voices_dir: Path = Path("voices"),
        num_threads: int = 2,
        num_steps: int = 4,
        speed: float = 1.0,
        chunk_chars: int = 50,
        gap: float = 0.15,
        cache_dir: Optional[Path] = None,
        max_chars: int = 1000,
    ):
        self.model_dir = Path(model_dir)
        self.vocoder = Path(vocoder)
        self.voices_dir = Path(voices_dir)
        self.num_threads = num_threads
        self.default_num_steps = num_steps
        self.default_speed = speed
        self.chunk_chars = max(20, chunk_chars)
        self.hard_max = self.chunk_chars * 2
        self.gap = gap
        self.max_chars = max_chars
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self._lock = threading.Lock()
        self.started_at = time.time()
        self.stats = {"requests": 0, "cache_hits": 0, "audio_seconds": 0.0, "synth_seconds": 0.0}

        self.voices = VoiceRegistry(self.voices_dir)
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        self._api = self._detect_api()
        self.tts, self.model_info = self._build()
        self.model_info["api"] = self._api
        self._cache_salt = self._fingerprint()

    def _detect_api(self) -> str:
        """1.13.x 用 GenerationConfig 传参考音频；更老的版本只能用 generate(prompt_text=...)"""
        try:
            if hasattr(sherpa_onnx.GenerationConfig, "reference_audio"):
                return "generation_config"
        except AttributeError:
            pass
        LOG.warning(
            "当前 sherpa-onnx 没有 GenerationConfig.reference_audio，退回旧版 generate(prompt_text=...) 接口。"
            "建议升级：pip install -U 'sherpa-onnx>=1.13.8'"
        )
        return "legacy"

    # ---------------------------------------------------------------- 模型
    def _pick(self, *names: str) -> Optional[Path]:
        for n in names:
            p = self.model_dir / n
            if p.is_file():
                return p
        return None

    def _build(self):
        tokens = self._pick("tokens.txt")
        encoder = self._pick("encoder.int8.onnx", "encoder.onnx", "encoder.fp16.onnx")
        decoder = self._pick("decoder.int8.onnx", "decoder.onnx")
        if not tokens or not encoder or not decoder or not self.vocoder.is_file():
            missing = [
                str(p)
                for p, ok in (
                    (self.model_dir / "tokens.txt", tokens),
                    (self.model_dir / "encoder.int8.onnx", encoder),
                    (self.model_dir / "decoder.int8.onnx", decoder),
                    (self.vocoder, self.vocoder.is_file()),
                )
                if not ok
            ]
            LOG.error("模型文件不全，缺：\n  %s", "\n  ".join(missing))
            raise SystemExit(2)

        if any(k in self.vocoder.name.lower() for k in ("22khz", "univ")):
            LOG.warning(
                "vocoder 看起来是 %s：ZipVoice 的 decoder 输出 100 mel，22kHz-univ 版是 80 mel，"
                "用错会合成出纯噪音。正确文件是 vocos_24khz.onnx（约 51.6MB）",
                self.vocoder.name,
            )

        lexicons = []
        main_lex = self._pick("lexicon.txt")
        if main_lex:
            lexicons.append(str(main_lex))
        if self.lexicon_extra and Path(self.lexicon_extra).is_file():
            lexicons.append(str(Path(self.lexicon_extra)))

        cfg = sherpa_onnx.OfflineTtsConfig(
            model=sherpa_onnx.OfflineTtsModelConfig(
                zipvoice=sherpa_onnx.OfflineTtsZipvoiceModelConfig(
                    tokens=str(tokens),
                    encoder=str(encoder),
                    decoder=str(decoder),
                    vocoder=str(self.vocoder),
                    lexicon=",".join(lexicons),
                ),
                num_threads=self.num_threads,
                debug=False,
                provider="cpu",
            ),
            max_num_sentences=1,
        )
        if not cfg.validate():
            LOG.error("模型配置校验失败，请看上面的错误信息")
            raise SystemExit(2)

        t0 = time.perf_counter()
        LOG.info("加载模型 ...（首次加载要读 ~130MB）")
        tts = sherpa_onnx.OfflineTts(cfg)
        load_s = time.perf_counter() - t0
        info = {
            "model_dir": str(self.model_dir),
            "vocoder": str(self.vocoder),
            "sample_rate": int(tts.sample_rate),
            "num_threads": self.num_threads,
            "num_steps": self.default_num_steps,
            "lexicons": [Path(p).name for p in lexicons],
            "load_seconds": round(load_s, 2),
        }
        LOG.info(
            "模型就绪，%.2fs | 输出采样率 %d Hz | 线程 %d | num_steps %d",
            load_s, tts.sample_rate, self.num_threads, self.default_num_steps,
        )
        return tts, info

    def _fingerprint(self) -> str:
        h = hashlib.md5()
        for p in (self.model_dir / "encoder.int8.onnx", self.model_dir / "decoder.int8.onnx", self.vocoder):
            if p.is_file():
                h.update(f"{p.name}:{p.stat().st_size}".encode())
        h.update(f"v{self.chunk_chars}".encode())
        return h.hexdigest()[:8]

    # ---------------------------------------------------------------- 缓存
    def _cache_path(self, text: str, voice: str, speed: float, num_steps: int) -> Optional[Path]:
        if not self.cache_dir:
            return None
        v = self.voices.get(voice)
        key = hashlib.md5(
            f"{self._cache_salt}|{voice}|{v.mtime_ns}|{num_steps}|{speed:.2f}|{text}".encode()
        ).hexdigest()
        return self.cache_dir / key[:2] / f"{key}.wav"

    # ---------------------------------------------------------------- 合成
    def synthesize(
        self,
        text: str,
        voice: Optional[str] = None,
        speed: Optional[float] = None,
        num_steps: Optional[int] = None,
        use_cache: bool = True,
    ) -> SynthResult:
        text = normalize_text(text or "")
        if not text:
            raise ValueError("text 为空")
        if self.max_chars and len(text) > self.max_chars:
            raise ValueError(
                f"文本 {len(text)} 字，超过上限 {self.max_chars}。"
                "请让阅读器按段落/句子切分后再请求，或用 --max-chars 放宽（长文本会合成很久）"
            )

        names = self.voices.names()
        if not names:
            raise ValueError(f"音色库为空，往 {self.voices_dir} 里放 <名字>.wav + <名字>.txt")
        if not voice:
            voice = names[0]
        ref_audio, ref_sr = self.voices.audio(voice)
        ref_text = self.voices.get(voice).text

        speed = float(self.default_speed if speed is None else speed)
        num_steps = int(self.default_num_steps if num_steps is None else num_steps)
        num_steps = max(1, min(num_steps, 32))
        speed = max(0.2, min(speed, 3.0))

        # 缓存命中直接返回：小说朗读每段只读一次，这一项收益最大
        cpath = self._cache_path(text, voice, speed, num_steps) if use_cache else None
        if cpath and cpath.is_file():
            data = cpath.read_bytes()
            with sf.SoundFile(io.BytesIO(data)) as f:
                dur = len(f) / f.samplerate
                sr = f.samplerate
            self.stats["cache_hits"] += 1
            self.stats["requests"] += 1
            self.stats["audio_seconds"] += dur
            LOG.info("cache HIT  voice=%s chars=%d dur=%.2fs", voice, len(text), dur)
            return SynthResult(data, sr, len(text), 0, dur, 0.0, cached=True)

        chunks = split_text(text, self.chunk_chars, self.hard_max)
        if not chunks:
            raise ValueError("文本切分后为空")

        t0 = time.perf_counter()
        parts: List[np.ndarray] = []
        sample_rate = int(self.tts.sample_rate)
        with self._lock:  # OfflineTts 单实例不是线程安全的
            for ch in chunks:
                if self._api == "generation_config":
                    gen = sherpa_onnx.GenerationConfig()
                    gen.reference_audio = ref_audio
                    gen.reference_sample_rate = ref_sr
                    gen.reference_text = ref_text
                    gen.num_steps = num_steps
                    gen.speed = speed
                    gen.extra["min_char_in_sentence"] = "30"
                    out = self.tts.generate(ch, gen)
                else:  # 旧接口（sherpa-onnx < 1.13）
                    out = self.tts.generate(
                        ch,
                        prompt_text=ref_text,
                        prompt_samples=ref_audio,
                        sample_rate=ref_sr,
                        speed=speed,
                        num_steps=num_steps,
                    )
                if out is None or len(out.samples) == 0:
                    raise RuntimeError(f"合成失败（空音频），分块内容：{ch[:40]!r}")
                sample_rate = int(out.sample_rate)
                if parts:
                    parts.append(np.zeros(int(sample_rate * self.gap), dtype=np.float32))
                parts.append(np.asarray(out.samples, dtype=np.float32))
        elapsed = time.perf_counter() - t0

        audio = np.concatenate(parts) if len(parts) > 1 else parts[0]
        dur = len(audio) / sample_rate
        wav = to_wav_bytes(audio, sample_rate)

        if cpath:
            cpath.parent.mkdir(parents=True, exist_ok=True)
            cpath.write_bytes(wav)

        self.stats["requests"] += 1
        self.stats["audio_seconds"] += dur
        self.stats["synth_seconds"] += elapsed
        res = SynthResult(wav, sample_rate, len(text), len(chunks), dur, elapsed)
        LOG.info(
            "synth voice=%s chars=%d chunks=%d steps=%d speed=%.2f | elapsed=%.2fs audio=%.2fs RTF=%.2f",
            voice, len(text), len(chunks), num_steps, speed, elapsed, dur, res.rtf,
        )
        return res


# --------------------------------------------------------------------------- HTTP
app = FastAPI(title="ZipVoice TTS", docs_url=None, redoc_url=None)
ENGINE: Optional[ZipVoiceEngine] = None


def _pick_param(q, *keys: str) -> Optional[str]:
    for k in keys:
        v = q.get(k)
        if v:
            return v
    return None


def _as_float(raw: Optional[str], default: float) -> float:
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        raise ValueError(f"参数不是数字: {raw!r}")


@app.get("/")
def index():
    return {
        "service": "zipvoice-tts",
        "endpoints": {
            "GET /tts": "?text=...&voice=...&speed=1.0&num_steps=4&nocache=0  -> audio/wav（Legado 用这个）",
            "GET /voices": "音色列表",
            "GET /health": "状态",
        },
    }


@app.get("/health")
def health():
    if ENGINE is None:
        return JSONResponse({"ok": False, "error": "engine not initialized"}, status_code=503)
    names = ENGINE.voices.names()
    st = ENGINE.stats
    return {
        "ok": True,
        "uptime_seconds": round(time.time() - ENGINE.started_at, 1),
        "model": ENGINE.model_info,
        "voices": names,
        "cache": {"enabled": ENGINE.cache_dir is not None, "hits": st["cache_hits"]},
        "stats": {
            "requests": st["requests"],
            "audio_seconds": round(st["audio_seconds"], 1),
            "avg_rtf": round(st["synth_seconds"] / st["audio_seconds"], 3) if st["audio_seconds"] else None,
        },
    }


@app.get("/voices")
def voices():
    if ENGINE is None:
        return JSONResponse({"ok": False, "error": "engine not initialized"}, status_code=503)
    out = []
    for name in ENGINE.voices.names():
        v = ENGINE.voices.get(name)
        try:
            samples, sr = ENGINE.voices.audio(name)
            dur = len(samples) / sr
        except Exception as e:  # 单个音色坏了不影响列表
            LOG.warning("音色 %s 加载失败: %s", name, e)
            continue
        out.append(
            {
                "name": name,
                "duration": round(dur, 2),
                "sample_rate": sr,
                "ref_text": v.text,
                "chars": len(v.text),
            }
        )
    return {"voices": out, "default": out[0]["name"] if out else None}


@app.get("/tts")
def tts(request: Request):
    if ENGINE is None:
        return JSONResponse({"ok": False, "error": "engine not initialized"}, status_code=503)
    q = request.query_params
    text = _pick_param(q, "text", "speakText", "txt", "t") or ""
    voice = _pick_param(q, "voice", "speaker", "v")
    try:
        speed = _as_float(q.get("speed"), ENGINE.default_speed)
        num_steps = int(_as_float(q.get("num_steps"), ENGINE.default_num_steps))
        use_cache = str(q.get("nocache", "0")).lower() not in ("1", "true", "yes")
        res = ENGINE.synthesize(text, voice=voice, speed=speed, num_steps=num_steps, use_cache=use_cache)
    except KeyError as e:
        return JSONResponse(
            {"ok": False, "error": f"音色不存在: {e.args[0]}", "voices": ENGINE.voices.names()},
            status_code=404,
        )
    except ValueError as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=400)
    except Exception as e:  # noqa: BLE001
        LOG.exception("合成异常")
        return JSONResponse({"ok": False, "error": f"{type(e).__name__}: {e}"}, status_code=500)

    return Response(
        content=res.wav,
        media_type="audio/wav",
        headers={
            "X-RTF": f"{res.rtf:.3f}",
            "X-Elapsed": f"{res.elapsed:.3f}",
            "X-Audio-Seconds": f"{res.audio_seconds:.3f}",
            "X-Chars": str(res.chars),
            "X-Chunks": str(res.chunks),
            "X-Cache": "hit" if res.cached else "miss",
            "Cache-Control": "no-store",
        },
    )


# --------------------------------------------------------------------------- CLI
def build_engine(args) -> ZipVoiceEngine:
    return ZipVoiceEngine(
        model_dir=Path(args.model_dir),
        vocoder=Path(args.vocoder),
        lexicon_extra=Path(args.lexicon) if args.lexicon else None,
        voices_dir=Path(args.voices_dir),
        num_threads=args.threads,
        num_steps=args.num_steps,
        speed=args.speed,
        chunk_chars=args.chunk_chars,
        gap=args.gap,
        cache_dir=Path(args.cache_dir) if args.cache_dir else None,
        max_chars=args.max_chars,
    )


def main() -> int:
    here = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description="ZipVoice + sherpa-onnx TTS 服务", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--model-dir", default=str(here / "models" / "sherpa-onnx-zipvoice-distill-int8-zh-en-emilia"))
    p.add_argument("--vocoder", default=str(here / "models" / "vocos_24khz.onnx"))
    p.add_argument("--lexicon", default=str(here / "models" / "lexicon-en-us.txt"), help="英文字典，可缺省")
    p.add_argument("--voices-dir", default=str(here / "voices"))
    p.add_argument("--cache-dir", default=str(here / "cache"), help="wav 缓存目录，空字符串=关闭")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--threads", type=int, default=2, help="onnxruntime 线程数，2 核给 2")
    p.add_argument("--num-steps", type=int, default=4, help="distill 模型 4 步；8 步更慢可能更稳")
    p.add_argument("--speed", type=float, default=1.0)
    p.add_argument("--chunk-chars", type=int, default=50, help="每块目标字数，越小越稳但越慢")
    p.add_argument("--gap", type=float, default=0.15, help="块间静音秒数")
    p.add_argument("--max-chars", type=int, default=1000, help="单次请求字符上限，0=不限")
    p.add_argument("--check", action="store_true", help="只加载模型并试合成一句，然后退出")
    p.add_argument("--no-synth", action="store_true", help="配合 --check：不试合成")
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO),
                        format="%(asctime)s %(levelname)-5s %(message)s", datefmt="%H:%M:%S")

    global ENGINE
    ENGINE = build_engine(args)

    if args.check:
        names = ENGINE.voices.names()
        print("\n--- 自检 ---")
        for k, v in ENGINE.model_info.items():
            print(f"  {k:14} {v}")
        print(f"  voices         {len(names)} 个: {', '.join(names) if names else '(空)'}")
        if names and not args.no_synth:
            text = "自检成功, 这是一段测试语音."
            print(f"\n  试合成：「{text}」 voice={names[0]}")
            res = ENGINE.synthesize(text, voice=names[0], use_cache=False)
            out = Path(__file__).resolve().parent / "check.wav"
            out.write_bytes(res.wav)
            print(
                f"  耗时 {res.elapsed:.2f}s / 音频 {res.audio_seconds:.2f}s -> RTF {res.rtf:.2f}"
                f"  ({res.chars} 字, {res.chunks} 块)"
            )
            print(f"  已保存 {out}（拉回本地听一下，确认不是噪音/电流声）")
        print("\n自检通过。")
        return 0

    LOG.info("监听 http://%s:%d   音色目录 %s", args.host, args.port, ENGINE.voices_dir)
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level.lower(), access_log=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
