"""bookpipe 自检：对 .fixtures 里的六本书逐项断言。

存在意义：切章/切段的规则一改就容易悄悄回归（比如把 L1 的正则收紧后某类标题再也匹配不到）。
每次改 chapter.py / segment.py / textio.py 都跑一遍。

    python tools/selftest.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bookpipe.chapter import split_chapters          # noqa: E402
from bookpipe.segment import split_segments           # noqa: E402
from bookpipe.textio import read_book, tidy           # noqa: E402

FIX = ROOT / ".fixtures"

# 文件名: (期望编码, 期望切章级, 章节数范围, 至少多少对白段)
EXPECT = {
    "utf8-chapters.txt":  ("utf-8",   "L1", (6, 6), 20),
    "utf16-chapters.txt": ("utf-16",  "L1", (6, 6), 20),
    "gbk-volumes.txt":    ("gb18030", "L1", (6, 6), 20),
    "big5-chapters.txt":  ("big5",    "L1", (4, 4), 10),
    "no-marker.txt":      ("utf-8",   "L2", (2, 40), 50),
    "broken-quotes.txt":  ("utf-8",   "L1", (2, 2), 2),
    "num-chapters.txt":   ("utf-8",   "L1", (6, 6), 20),
    "large-utf8.txt":     ("utf-8",   "L1", (460, 460), 200),
}


def run() -> int:
    if not FIX.is_dir():
        print("先跑 python tools/make_fixtures.py 生成测试书", file=sys.stderr)
        return 2

    fails = 0
    for name, (enc, level, rng, min_dial) in EXPECT.items():
        path = FIX / name
        if not path.is_file():
            print(f"FAIL {name}: 文件不存在")
            fails += 1
            continue

        problems: list[str] = []
        try:
            raw, info = read_book(path)
        except Exception as exc:                       # noqa: BLE001
            print(f"FAIL {name}: 读取失败 {exc}")
            fails += 1
            continue

        if info.encoding != enc:
            problems.append(f"编码 {info.encoding} != {enc}")

        text = tidy(raw)
        split = split_chapters(text)
        if split.level != level:
            problems.append(f"切章级 {split.level} != {level}")
        if not (rng[0] <= len(split.chapters) <= rng[1]):
            problems.append(f"章数 {len(split.chapters)} 不在 {rng}")

        total_seg = dial = broken = unbalanced = 0
        for ch in split.chapters:
            seg = split_segments(ch.text)
            total_seg += len(seg.segments)
            dial += sum(1 for s in seg.segments if s.type == "dialogue")
            unbalanced += len(seg.warnings)
            for s in seg.segments:
                if ch.text[s.start:s.end] != s.text:
                    broken += 1

        if broken:
            problems.append(f"{broken} 段偏移不一致")
        if dial < min_dial:
            problems.append(f"对白段仅 {dial}，期望 ≥{min_dial}")

        status = "PASS" if not problems else "FAIL"
        print(f"{status} {name:22} {info.encoding:8} {split.level} "
              f"{len(split.chapters):>3}章 {total_seg:>4}段 对白{dial:>4}")
        for p in problems:
            print(f"       · {p}")
        fails += len(problems)

    # 编码探测的负例：随机字节应当被拒绝，而不是"猜一个"
    junk = bytes(range(256)) * 40
    try:
        from bookpipe.textio import detect_bytes
        detect_bytes(junk)
        print("FAIL 随机字节没有被拒绝")
        fails += 1
    except Exception:                                  # noqa: BLE001
        print("PASS 随机字节被正确拒绝")

    # 数字编号判定的正负例：靠序列递增性区分"章节号"和"正文里的零散数字"
    from bookpipe.chapter import _num_sequence_ok
    cases = [
        ([1, 2, 3, 4, 5], True, "递增编号"),
        ([1, 2, 3, 5, 6, 8], True, "缺章跳号"),
        ([1, 2, 3, 1, 2, 3], True, "分卷重新编号"),
        ([3, 17, 2, 40, 8, 1], False, "零散数字（正文孤行）"),
        ([1, 2], False, "数量太少"),
    ]
    for nums, want, desc in cases:
        got = _num_sequence_ok(nums)
        if got == want:
            print(f"PASS 数字序列判定 / {desc} -> {got}")
        else:
            print(f"FAIL 数字序列判定 / {desc} 期望 {want} 实得 {got}")
            fails += 1

    print(f"\n{'全部通过' if fails == 0 else f'{fails} 项失败'}")
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    raise SystemExit(run())
