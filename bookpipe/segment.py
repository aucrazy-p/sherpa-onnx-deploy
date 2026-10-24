"""切段：把一章拆成 narration / dialogue 两种段。

纯规则，零成本，不调 LLM。这一步唯一的职责是**把对白和旁白分开**——
它们属于不同说话人，混在一起后面就无从归属。

两条铁律：
1. 不改写正文。每个段都是原文的一个连续切片（segment.text == 原文[start:end]），
   引号原样保留。清标点是合成层的事，不是切分层的事。
2. 引号不闭合时整段降级为旁白 + 计一次告警，绝不猜。
   猜错会产出"旁白用一个声音念了两个角色的对话"，这在成品里几乎无法发现。
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# 开引号 → 闭引号
_OPENERS = {"「": "」", "『": "』", "“": "”", "‘": "’", '"': '"'}

# 行尾若是这些字符，说明这一行是完整句子，换行多半是排版换行而非续行
_TERMINAL = "。！？!?…」』”\"'"


@dataclass
class Segment:
    seq: int
    type: str          # narration | dialogue
    text: str
    start: int         # 相对本章正文的偏移
    end: int
    para_idx: int

    @property
    def chars(self) -> int:
        return len(self.text)


@dataclass
class SegmentResult:
    segments: list[Segment]
    warnings: list[str]

    @property
    def dialogue_chars(self) -> int:
        return sum(s.chars for s in self.segments if s.type == "dialogue")

    @property
    def total_chars(self) -> int:
        return sum(s.chars for s in self.segments)

    @property
    def dialogue_ratio(self) -> float:
        t = self.total_chars
        return self.dialogue_chars / t if t else 0.0


def _paragraphs(text: str) -> list[tuple[int, str]]:
    """拆段落，返回 [(行首偏移, 段落文本)]。

    中文小说一段一行，但有些 txt 是硬折行的（一行 40 字掐断）。
    判据：上一行结尾没有句终标点 → 认为是续行，合并回去。
    """
    out: list[tuple[int, str]] = []
    buf: list[str] = []
    buf_start = 0
    pos = 0

    def flush() -> None:
        nonlocal buf
        if buf:
            joined = "\n".join(buf)
            if joined.strip():
                out.append((buf_start, joined))
            buf = []

    for line in text.split("\n"):
        if not line.strip():
            flush()
        else:
            if buf and buf[-1].rstrip() and buf[-1].rstrip()[-1] in _TERMINAL:
                flush()
                buf_start = pos
            elif not buf:
                buf_start = pos
            buf.append(line)
        pos += len(line) + 1
    flush()
    return out


def _quote_spans(para: str) -> tuple[list[tuple[int, int]], bool]:
    """找出所有引号区间 [start, end)，返回 (spans, 是否全部闭合)。"""
    spans: list[tuple[int, int]] = []
    stack: list[tuple[int, str]] = []
    ascii_open = -1

    for i, ch in enumerate(para):
        if ch in _OPENERS:
            closer = _OPENERS[ch]
            if ch == '"':
                # ASCII 引号靠奇偶配对
                if ascii_open < 0:
                    ascii_open = i
                else:
                    spans.append((ascii_open, i + 1))
                    ascii_open = -1
                continue
            stack.append((i, closer))
        elif stack and ch == stack[-1][1]:
            start, _ = stack.pop()
            if not stack:          # 只记最外层，内层引号并入外层文本
                spans.append((start, i + 1))

    balanced = not stack and ascii_open < 0
    if not balanced:
        return [], False
    spans.sort()
    return spans, True


def split_segments(chapter_text: str) -> SegmentResult:
    """把整章拆成段。"""
    segments: list[Segment] = []
    warnings: list[str] = []
    unbalanced = 0

    for para_idx, (p_off, para) in enumerate(_paragraphs(chapter_text)):
        spans, ok = _quote_spans(para)

        if not ok:
            unbalanced += 1
            segments.append(Segment(
                seq=len(segments), type="narration", text=para,
                start=p_off, end=p_off + len(para), para_idx=para_idx,
            ))
            continue

        if not spans:
            segments.append(Segment(
                seq=len(segments), type="narration", text=para,
                start=p_off, end=p_off + len(para), para_idx=para_idx,
            ))
            continue

        cursor = 0
        for s, e in spans:
            if s > cursor:
                head = para[cursor:s]
                if head.strip():
                    segments.append(Segment(
                        seq=len(segments), type="narration", text=head,
                        start=p_off + cursor, end=p_off + s, para_idx=para_idx,
                    ))
            segments.append(Segment(
                seq=len(segments), type="dialogue", text=para[s:e],
                start=p_off + s, end=p_off + e, para_idx=para_idx,
            ))
            cursor = e
        if cursor < len(para):
            tail = para[cursor:]
            if tail.strip():
                segments.append(Segment(
                    seq=len(segments), type="narration", text=tail,
                    start=p_off + cursor, end=p_off + len(para), para_idx=para_idx,
                ))

    if unbalanced:
        warnings.append(
            f"{unbalanced} 个段落的引号不闭合，已整段降级为旁白（多做一次人工抽查）"
        )

    # 把相邻同类段合并，减少合成次数（每段都要重新编码参考音频）
    segments = _merge_adjacent(segments)
    for i, seg in enumerate(segments):
        seg.seq = i
    return SegmentResult(segments, warnings)


def _merge_adjacent(segments: list[Segment]) -> list[Segment]:
    """相邻同类型段合并。切碎成一句一段会让 ZipVoice 白白重编码参考音频。"""
    if not segments:
        return []
    out = [segments[0]]
    for seg in segments[1:]:
        prev = out[-1]
        contiguous = seg.start == prev.end
        if seg.type == prev.type and contiguous:
            prev.text = prev.text + seg.text
            prev.end = seg.end
        else:
            out.append(seg)
    return out


_WS = re.compile(r"[\s\u3000]+")


def spoken_text(text: str) -> str:
    """给合成层用：只做"朗读前清场"，不改词。

    去掉引号、行内多余空白——引号念不出来，但会让某些引擎的韵律判断出错。
    这是**渲染参数**，不是对原文的修改：库里的 text 字段永远保留原样。
    """
    t = text.strip()
    for ch in "「」『』“”‘’":
        t = t.replace(ch, "")
    t = t.replace('"', "")
    t = _WS.sub(" ", t).strip()
    return t
