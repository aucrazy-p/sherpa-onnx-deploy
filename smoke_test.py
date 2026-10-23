#!/usr/bin/env python3
"""smoke_test.py — 对已在跑的服务做冒烟测试，输出真实 RTF

它只发 HTTP 请求，不加载模型，所以测到的就是 Legado 将来会遇到的真实延迟。
（边测试边看服务端日志能看到每块的分块与 RTF。）

用法
  python smoke_test.py                          # 默认打 http://127.0.0.1:8080
  python smoke_test.py --base-url http://127.0.0.1:8080 --out smoke_out
  python smoke_test.py --skip-long              # 跳过长文本，省时间

产物
  smoke_out/*.wav    拉回本地听（这是判断音质的唯一可靠方式）
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

SHORT = "今天的天气很好，适合出门走走。"
MEDIUM = "小米的价值观是真诚，热爱。真诚，就是不欺人也不自欺；热爱，就是全心投入并享受其中。"
LONG = (
    "第一章 夜行。"
    "雨下了一整夜, 城里的灯在积水里碎成一地金子。"
    "他从便利店出来, 手里拎着一袋热腾腾的包子, 站在屋檐下看了一会儿天。"
    "街对面的钟楼敲了十二下, 声音闷闷的, 像是被雨裹住了。"
    "他忽然想起很多年前的一个晚上, 也是这样的雨, 也是这样的钟声, 只是那时候身边还站着一个人。"
    "那个人说过, 等雨停了就走, 可雨一直没有停。"
    "他低头咬了一口包子, 热气糊了眼镜, 世界一下子变得模糊又温和。"
)


# --------------------------------------------------------------------------- HTTP
def http_get(base: str, path: str, params: dict | None, timeout: float) -> tuple[int, dict, bytes]:
    url = base.rstrip("/") + path
    if params:
        url += "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": "smoke-test/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, dict(r.headers), r.read()
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers), e.read()


def tts(base: str, text: str, voice: str | None, nocache: bool, timeout: float):
    params = {"text": text}
    if voice:
        params["voice"] = voice
    if nocache:
        params["nocache"] = "1"
    t0 = time.perf_counter()
    status, headers, body = http_get(base, "/tts", params, timeout)
    client = time.perf_counter() - t0
    if status != 200:
        try:
            err = json.loads(body.decode("utf-8", "replace")).get("error", "")
        except Exception:
            err = body[:200].decode("utf-8", "replace")
        raise RuntimeError(f"HTTP {status}: {err}")
    lower = {k.lower(): v for k, v in headers.items()}
    return {
        "wav": body,
        "client": client,
        "rtf": float(lower.get("x-rtf", "nan")),
        "elapsed": float(lower.get("x-elapsed", "nan")),
        "audio_seconds": float(lower.get("x-audio-seconds", "nan")),
        "chunks": int(lower.get("x-chunks", "0")),
        "cache": lower.get("x-cache", "?"),
    }


# --------------------------------------------------------------------------- 音频体检
def inspect_wav(data: bytes) -> dict:
    import io

    import numpy as np
    import soundfile as sf

    samples, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
    mono = samples[:, 0]
    peak = float(np.abs(mono).max()) if mono.size else 0.0
    rms = float(np.sqrt((mono.astype(np.float64) ** 2).mean())) if mono.size else 0.0
    clip = float((np.abs(mono) > 0.999).mean()) if mono.size else 0.0
    zcr = float((np.diff(np.signbit(mono)) != 0).mean()) if mono.size > 1 else 0.0
    return {
        "sr": sr,
        "duration": len(mono) / sr if sr else 0.0,
        "peak": peak,
        "rms": rms,
        "clip": clip,
        "zcr": zcr,
    }


def audio_verdict(a: dict) -> str:
    if a["peak"] < 0.01:
        return "!! 近乎静音，合成失败"
    if a["zcr"] > 0.4:
        return "!! 过零率异常高，疑似噪音（vocoder 选错？）"
    if a["clip"] > 0.01:
        return "!! 有削波，音量过大"
    return "ok"


def rtf_verdict(rtf: float) -> str:
    if rtf < 1.0:
        return "优秀：比实时快，连听不卡"
    if rtf < 2.0:
        return "可用：段落级预生成/缓存后能听"
    if rtf < 3.0:
        return "偏慢：实时连听会明显停顿"
    return "太慢：只适合离线批量或强依赖缓存"


# --------------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8080")
    ap.add_argument("--out", default="smoke_out")
    ap.add_argument("--voice", action="append", default=None, help="指定音色，可重复；默认测全部")
    ap.add_argument("--timeout", type=float, default=300.0)
    ap.add_argument("--skip-long", action="store_true")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    base = args.base_url
    failures: list[str] = []
    rows: list[dict] = []
    t_start = time.perf_counter()

    def head(title: str) -> None:
        print(f"\n\033[1;36m==> {title}\033[0m")

    # 1) health
    head("1/5 健康检查")
    try:
        status, _, body = http_get(base, "/health", None, 15)
    except Exception as e:
        print(f"  连不上 {base}: {e}")
        print("  先把服务起起来：python engine.py --port 8080")
        return 2
    if status != 200:
        print(f"  /health 返回 {status}: {body[:300]}")
        return 2
    health = json.loads(body)
    m = health["model"]
    print(f"  服务在线，已运行 {health['uptime_seconds']}s")
    print(f"  模型       {Path(m['model_dir']).name}")
    print(f"  vocoder    {Path(m['vocoder']).name}")
    print(f"  采样率     {m['sample_rate']} Hz")
    print(f"  线程/步数  {m['num_threads']} 线程 / num_steps={m['num_steps']}")
    print(f"  词典       {', '.join(m['lexicons']) or '(无)'}")
    print(f"  缓存       {'开' if health['cache']['enabled'] else '关'}（命中 {health['cache']['hits']} 次）")
    print(f"  载入耗时   {m['load_seconds']}s")

    # 2) voices
    head("2/5 音色列表")
    status, _, body = http_get(base, "/voices", None, 15)
    vs = json.loads(body).get("voices", []) if status == 200 else []
    if not vs:
        print("  音色库是空的 —— 先跑 install.sh，或往 voices/ 放 <名>.wav + <名>.txt")
        return 2
    for v in vs:
        print(f"  - {v['name']:<14} {v['duration']:>5.2f}s  {v['chars']:>3} 字  转写: {v['ref_text'][:34]}…")
    names = [v["name"] for v in vs]
    targets = args.voice or names
    for t in targets:
        if t not in names:
            print(f"  指定的音色不存在: {t}")
            return 2

    # 3) 每音色同一句
    head(f"3/5 逐音色测速（{len(MEDIUM)} 字，不受缓存影响）")
    print(f"  文本：{MEDIUM}")
    print(f"  {'音色':<14}{'字数':>4}{'耗时s':>8}{'音频s':>8}{'RTF':>7}{'块':>4}  音频体检")
    for name in targets:
        try:
            r = tts(base, MEDIUM, name, nocache=True, timeout=args.timeout)
        except Exception as e:
            print(f"  {name:<14} 失败: {e}")
            failures.append(f"{name}: {e}")
            continue
        a = inspect_wav(r["wav"])
        verdict = audio_verdict(a)
        if verdict != "ok":
            failures.append(f"{name} 音频体检: {verdict}")
        (out / f"medium-{name}.wav").write_bytes(r["wav"])
        print(
            f"  {name:<14}{len(MEDIUM):>4}{r['client']:>8.2f}{a['duration']:>8.2f}"
            f"{r['rtf']:>7.2f}{r['chunks']:>4}  {verdict}  peak={a['peak']:.2f} rms={a['rms']:.3f} zcr={a['zcr']:.2f}"
        )
        rows.append({"scene": f"medium/{name}", "chars": len(MEDIUM), "rtf": r["rtf"],
                     "client": r["client"], "audio": a["duration"]})

    # 4) 短句
    head("4/5 短句开销（短文本固定开销占比高，别被它误导）")
    try:
        r = tts(base, SHORT, targets[0], nocache=True, timeout=args.timeout)
        a = inspect_wav(r["wav"])
        (out / "short.wav").write_bytes(r["wav"])
        print(f"  {len(SHORT)} 字：耗时 {r['client']:.2f}s  音频 {a['duration']:.2f}s  RTF {r['rtf']:.2f}")
        rows.append({"scene": "short", "chars": len(SHORT), "rtf": r["rtf"],
                     "client": r["client"], "audio": a["duration"]})
    except Exception as e:
        print(f"  失败: {e}")
        failures.append(f"short: {e}")

    # 5) 长文本 + 缓存
    if not args.skip_long:
        head(f"5/5 长文本（{len(LONG)} 字，看分块与整段耗时）")
        try:
            r = tts(base, LONG, targets[0], nocache=True, timeout=args.timeout)
            a = inspect_wav(r["wav"])
            (out / "long.wav").write_bytes(r["wav"])
            print(
                f"  {len(LONG)} 字 -> {r['chunks']} 块，耗时 {r['client']:.1f}s，"
                f"音频 {a['duration']:.1f}s，RTF {r['rtf']:.2f}"
            )
            print(f"  {audio_verdict(a)}  peak={a['peak']:.2f} rms={a['rms']:.3f}")
            rows.append({"scene": "long", "chars": len(LONG), "rtf": r["rtf"],
                         "client": r["client"], "audio": a["duration"]})
        except Exception as e:
            print(f"  失败: {e}")
            failures.append(f"long: {e}")
    else:
        head("5/5 长文本（已跳过）")

    head("缓存效果")
    try:
        tts(base, MEDIUM, targets[0], nocache=False, timeout=args.timeout)   # 预热（写缓存）
        r2 = tts(base, MEDIUM, targets[0], nocache=False, timeout=args.timeout)
        print(f"  第二次请求 {len(MEDIUM)} 字：{r2['client'] * 1000:.0f} ms（X-Cache={r2['cache']}）")
        if r2["cache"] != "hit":
            print("  注意：没有命中缓存，检查 --cache-dir 是否被关掉")
    except Exception as e:
        print(f"  缓存测试失败: {e}")

    # 汇总
    head("汇总")
    if rows:
        print(f"  {'场景':<18}{'字数':>5}{'耗时s':>8}{'音频s':>8}{'RTF':>7}")
        for r in rows:
            print(f"  {r['scene']:<18}{r['chars']:>5}{r['client']:>8.2f}{r['audio']:>8.2f}{r['rtf']:>7.2f}")
        worst = max(rows, key=lambda r: r["rtf"])
        print(f"\n  最差 RTF {worst['rtf']:.2f}（{worst['scene']}）— {rtf_verdict(worst['rtf'])}")

    print(f"\n  总耗时 {time.perf_counter() - t_start:.1f}s")
    print(f"  wav 已存到 {out.resolve()}")
    print("  下一步：scp 回来听音质 —— 数字好看不代表听着舒服（吞字、尾音断、机器味都在 wav 里）")

    if failures:
        print("\n\033[1;31m失败项\033[0m")
        for f in failures:
            print(f"  - {f}")
        return 1

    print("\n\033[1;32m冒烟测试通过\033[0m")
    return 0


if __name__ == "__main__":
    sys.exit(main())
