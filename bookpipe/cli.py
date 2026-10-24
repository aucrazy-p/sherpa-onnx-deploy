"""命令行入口。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__
from .chapter import split_chapters
from .segment import split_segments
from .textio import DecodeError, read_book, tidy


def _fmt(n: int) -> str:
    return f"{n:,}"


def cmd_inspect(args: argparse.Namespace) -> int:
    path = Path(args.book)
    try:
        raw, info = read_book(path, encoding=args.encoding)
    except (DecodeError, FileNotFoundError) as exc:
        print(f"\n读取失败：{exc}", file=sys.stderr)
        return 2

    text = tidy(raw)
    print(f"\n文件      {path.name}")
    print(f"编码      {info.describe()}")
    print(f"字符数    {_fmt(len(text))}")

    split = split_chapters(text, hard_chars=args.hard_chars)
    print(f"\n切章      {split.level}  共 {len(split.chapters)} 章  合计 {_fmt(split.total_chars)} 字")
    for w in split.warnings:
        print(f"          注：{w}")
    if split.chapters:
        avg = split.total_chars / len(split.chapters)
        print(f"          平均 {avg:.0f} 字/章")

    show = min(args.show, len(split.chapters))
    print(f"\n章节明细（前 {show} 章）")
    for ch in split.chapters[:show]:
        flag = "" if 200 <= ch.chars <= 30000 else "  ← 长度异常"
        print(f"  [{ch.idx:>3}] {_fmt(ch.chars):>8} 字  {ch.title[:32]}{flag}")

    # ---- 切段 + 偏移自检
    total_seg = 0
    total_dial = 0
    all_warn: list[str] = []
    first_seg = None
    broken = 0

    for ch in split.chapters:
        seg = split_segments(ch.text)
        total_seg += len(seg.segments)
        total_dial += sum(1 for s in seg.segments if s.type == "dialogue")
        all_warn.extend(seg.warnings)
        for s in seg.segments:
            if ch.text[s.start:s.end] != s.text:
                broken += 1
        if first_seg is None and seg.segments:
            first_seg = (ch, seg)

    chars_all = sum(s.chars for ch in split.chapters for s in split_segments(ch.text).segments)
    print(f"\n切段      共 {_fmt(total_seg)} 段（对白 {_fmt(total_dial)} 段，"
          f"占 {total_dial / total_seg * 100 if total_seg else 0:.0f}%）")
    if all_warn:
        for w in dict.fromkeys(all_warn):
            print(f"          注：{w}")

    if first_seg:
        ch, seg = first_seg
        ratio = seg.dialogue_ratio * 100
        print(f"\n抽样：{ch.title[:30]}  {len(seg.segments)} 段  对白占比 {ratio:.0f}%")
        for s in seg.segments[: args.show_seg]:
            body = s.text.replace("\n", "⏎")
            if len(body) > 56:
                body = body[:56] + "…"
            tag = "对白" if s.type == "dialogue" else "旁白"
            print(f"  {s.seq:>3} {tag}  {body}")

    print(f"\n偏移自检  {'通过（每段都能在原文中精确切回）' if broken == 0 else f'{broken} 段偏移不一致'}")

    if args.jsonl:
        out = Path(args.jsonl)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as fh:
            for ch in split.chapters:
                for s in split_segments(ch.text).segments:
                    fh.write(json.dumps({
                        "chapter": ch.idx,
                        "chapter_title": ch.title,
                        "seq": s.seq,
                        "type": s.type,
                        "speaker": "narrator" if s.type == "narration" else None,
                        "text": s.text,
                        "start": ch.start + s.start,
                        "end": ch.start + s.end,
                    }, ensure_ascii=False) + "\n")
        print(f"          已写出 {out}")

    return 0 if broken == 0 else 3


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="bookpipe", description="小说 → 有声书：前置文本层")
    ap.add_argument("-V", "--version", action="version", version=f"bookpipe {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("inspect", help="看一本书能切得怎么样（只读，不写任何东西）")
    p.add_argument("book", help="txt 文件路径")
    p.add_argument("--encoding", help="强制指定编码（默认自动探测）")
    p.add_argument("--hard-chars", type=int, default=5000, help="L3 硬切字数，默认 5000")
    p.add_argument("--show", type=int, default=10, help="显示前 N 章，默认 10")
    p.add_argument("--show-seg", type=int, default=8, help="抽样显示 N 段，默认 8")
    p.add_argument("--jsonl", help="把分段结果写到这个文件")
    p.set_defaults(func=cmd_inspect)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
