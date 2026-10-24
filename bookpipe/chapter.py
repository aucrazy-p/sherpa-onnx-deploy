"""切章：三级降级。

L1  章节标题行    第X章 / 卷X 第X章 / Chapter N
L2  关键词 + 大段空行   序章、楔子、尾声、番外；或 3 连空行形成的自然分隔
L3  按字数硬切    切点永远落在段落边界，不切断句子

任何一级命中就用那一级，不再往下走。**切不出来不是错误，硬切兜住**——
一本 200 万字的书卡在第一步，比切得粗一点糟得多。
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# 数字：阿拉伯/全角/中文
_NUM = r"[0-9０-９零〇一二三四五六七八九十百千两]{1,10}"
_VOL = rf"第\s*{_NUM}\s*[卷部篇集]"
_CH = rf"第\s*{_NUM}\s*[章节節回话話]"

# 章节标题行的最大长度。超过这个长度几乎肯定是正文里的句子，不是标题。
_MAX_TITLE_LEN = 40

_RE_CH_HEAD = re.compile(
    rf"^(?:正文\s*)?(?:{_VOL}\s*)?{_CH}(?:\s*[:：、.．\-—]?\s*\S.{{0,30}})?$"
)
_RE_VOL_HEAD = re.compile(rf"^(?:正文\s*)?{_VOL}(?:\s*[:：、.．\-—]?\s*\S{0,30})?$")
_RE_EN_HEAD = re.compile(
    r"^(?:chapter|part|book)\s+(?:[0-9]{1,4}|[ivxlcdm]{1,8})\b.{0,40}$", re.IGNORECASE
)
_RE_KEY_HEAD = re.compile(
    r"^(?:序章|序言|自序|序|楔子|引子|引言|前言|后记|後記|尾声|尾聲|结语|結語|终章|終章"
    r"|番外|附录|附錄|外传|外傳|大结局|大結局)(?:\s*[:：、.．\-—]?\s*.{0,20})?$"
)

# 标题行里不该出现的句子结束符
_SENT_END = "。！？!?"


@dataclass
class Chapter:
    idx: int
    title: str
    text: str
    start: int
    end: int
    level: str

    @property
    def chars(self) -> int:
        return len(self.text.strip())


@dataclass
class SplitResult:
    chapters: list[Chapter]
    level: str
    warnings: list[str]

    @property
    def total_chars(self) -> int:
        return sum(c.chars for c in self.chapters)


def _clean(line: str) -> str:
    return line.strip().strip("\u3000").strip()


def _classify(line: str) -> str | None:
    """判断这一行是不是章节标题。返回 'ch' / 'vol' / 'en' / 'key' 或 None。"""
    s = _clean(line)
    if not s or len(s) > _MAX_TITLE_LEN:
        return None
    if any(ch in s for ch in _SENT_END):
        return None
    if _RE_CH_HEAD.match(s):
        return "ch"
    if _RE_VOL_HEAD.match(s):
        return "vol"
    if _RE_EN_HEAD.match(s):
        return "en"
    if _RE_KEY_HEAD.match(s):
        return "key"
    return None


def _headings(text: str) -> list[tuple[int, str, str]]:
    """扫描全部标题行，返回 [(行首偏移, 标题, 类型)]。"""
    found: list[tuple[int, str, str]] = []
    pos = 0
    for line in text.split("\n"):
        kind = _classify(line)
        if kind:
            found.append((pos, _clean(line), kind))
        pos += len(line) + 1
    return found


def _build(text: str, heads: list[tuple[int, str]], level: str) -> list[Chapter]:
    """按给定切点生成章节。第一个切点之前的内容算"前言块"，非空则单独成一章。"""
    chapters: list[Chapter] = []
    first_start = heads[0][0]
    preface = text[:first_start]
    if preface.strip():
        chapters.append(Chapter(idx=0, title="（开头）", text=preface,
                                start=0, end=first_start, level=level))
    for i, (off, title) in enumerate(heads):
        end = heads[i + 1][0] if i + 1 < len(heads) else len(text)
        chapters.append(Chapter(idx=len(chapters), title=title, text=text[off:end],
                                start=off, end=end, level=level))
    return chapters


def _blank_run_splits(text: str) -> list[int]:
    """3 连以上空行形成的分隔点（行首偏移）。用于没有章节标记的书。"""
    splits: list[int] = []
    pos = 0
    blank = 0
    for line in text.split("\n"):
        if not line.strip():
            blank += 1
        else:
            if blank >= 3 and splits and pos - splits[-1] > 800:
                splits.append(pos)
            elif blank >= 3 and not splits and pos > 800:
                splits.append(pos)
            blank = 0
        pos += len(line) + 1
    return splits


def _hard_split(text: str, hard_chars: int) -> list[Chapter]:
    """L3：按段落累积到硬切字数，切点只落在段落边界。"""
    paras = [p for p in re.split(r"\n\s*\n", text)]
    chapters: list[Chapter] = []
    buf: list[str] = []
    buf_len = 0
    cursor = 0
    start = 0

    def flush(end_at: int) -> None:
        nonlocal buf, buf_len, start
        if not buf:
            return
        body = "\n\n".join(buf)
        chapters.append(Chapter(
            idx=len(chapters),
            title=f"第 {len(chapters) + 1} 段（约 {buf_len} 字）",
            text=body, start=start, end=end_at, level="L3",
        ))
        buf, buf_len, start = [], 0, end_at

    for para in paras:
        plen = len(para)
        # 单段就超过硬切上限 → 单独成章，并在句子边界再切
        if plen > hard_chars * 1.6:
            flush(cursor)
            pieces = re.split(r"(?<=[。！？!?…])", para)
            piece_buf: list[str] = []
            piece_len = 0
            seg_start = cursor
            for piece in pieces:
                if piece_len and piece_len + len(piece) > hard_chars:
                    body = "".join(piece_buf)
                    chapters.append(Chapter(
                        idx=len(chapters),
                        title=f"第 {len(chapters) + 1} 段（约 {piece_len} 字）",
                        text=body, start=seg_start, end=seg_start + len(body), level="L3",
                    ))
                    seg_start += len(body)
                    piece_buf, piece_len = [], 0
                piece_buf.append(piece)
                piece_len += len(piece)
            if piece_buf:
                body = "".join(piece_buf)
                chapters.append(Chapter(
                    idx=len(chapters),
                    title=f"第 {len(chapters) + 1} 段（约 {piece_len} 字）",
                    text=body, start=seg_start, end=seg_start + len(body), level="L3",
                ))
            cursor += plen + 2
            start = cursor
            continue

        if buf_len and buf_len + plen > hard_chars:
            flush(cursor)
        buf.append(para)
        buf_len += plen
        cursor += plen + 2

    flush(len(text))
    return chapters


def split_chapters(
    text: str,
    *,
    min_chapters: int = 2,
    hard_chars: int = 5000,
) -> SplitResult:
    """三级降级切章。"""
    warnings: list[str] = []

    # ---- L1：章节标题行
    heads = [(off, title) for off, title, kind in _headings(text) if kind in ("ch", "vol", "en")]
    if len(heads) >= min_chapters:
        chapters = _build(text, heads, "L1")
        return SplitResult(chapters, "L1", warnings)
    if len(heads) == 1:
        warnings.append("L1 只找到 1 处章节标题，不足以切分")

    # ---- L2：关键词标题，或 3 连空行
    key_heads = [(off, title) for off, title, kind in _headings(text) if kind == "key"]
    if len(key_heads) >= min_chapters:
        warnings.append(f"L1 未命中，改用 L2（关键词标题 {len(key_heads)} 处）")
        return SplitResult(_build(text, key_heads, "L2"), "L2", warnings)

    blank_splits = _blank_run_splits(text)
    if len(blank_splits) >= 2:
        heads2 = [(off, f"分段 {i + 1}") for i, off in enumerate(blank_splits)]
        chapters = _build(text, heads2, "L2")
        avg = sum(c.chars for c in chapters) / len(chapters)
        if 3 <= len(chapters) <= 800 and avg >= 800:
            warnings.append(f"L1/L2 关键词均未命中，改用 L2（3 连空行 {len(chapters)} 段，均 {avg:.0f} 字）")
            return SplitResult(chapters, "L2", warnings)
        warnings.append(f"3 连空行切出 {len(chapters)} 段、平均 {avg:.0f} 字，粒度不合适，继续降级")

    # ---- L3：硬切
    warnings.append(f"没有任何章节标记，改用 L3 按 {hard_chars} 字硬切（切点落在段落边界）")
    return SplitResult(_hard_split(text, hard_chars), "L3", warnings)
