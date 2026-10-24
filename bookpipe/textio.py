"""编码探测与文本规范化。

铁律：探测置信度不足必须报错退出，绝不 fallback 到 utf-8 硬解。
乱码喂进 TTS 比报错严重得多——会产出一整章毫无意义的音频，而且没人会发现。

不依赖 charset-normalizer：只用标准库，保证「本地跑完直接丢上服务器」不需要装任何东西。

判定思路（第一版写错了，这里记下来）：不能用"绝对分数阈值"。
真实中文正文里只有一半左右是高频字，任何基于常用字表的绝对值都会落在 0.7x，
一刀切 0.80 会把正常文件全部拦死。正确的做法是**比较候选之间的相对优势**——
同一个字节流被正确解码时，分数会明显高于被错误解码时（乱码几乎全是生僻字）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# 候选编码。utf-8 严格解码对中文几乎不会误判，放最前；gb18030 最宽，放 utf-8 之后。
CANDIDATES = ("utf-8", "gb18030", "big5", "utf-16")

# 简繁常用字。用途只有一个：识别"乱码"——乱码解出来全是生僻字，几乎命中不了这张表。
# 它不需要覆盖全文，只需要让正确解码和错误解码拉开差距。
_COMMON = (
    "的一是了我不人在他有这个上们来到时大地为子中你说生国年着就那和要她出也得里后自以会家可下而过天去能对小多然于心学么之都好看起发当没成只如事把还用第样道想作种开美总从无情己面最女但现前些所同日手又行意动方期它头经长儿回位分爱老因很给名法间斯知世什两次使身者被高已亲其进此话常与活正感两问车东马体边变电将从种当经动题头"
    "見許還進說東車馬國點麼樣時來對開會後學現實發電話將從種當經動兩問題寶貝們個這氣聽寫讀買賣頭體變邊讓覺關聲願裡燈積熱騰菸熏黃悶裹聲鏽塵傘豎插兜遠開長尾散掉"
)
_COMMON_SET = set(_COMMON)

# 中文标点 + 常见符号
_PUNCT = set("，。！？；：、“”‘’（）《》〈〉【】〔〕…—～·「」『』％‰℃　・")

_LOW = 0x20
_HIGH = 0x7E


@dataclass
class _Cand:
    enc: str
    text: str
    printable: float
    common: float
    score: float


@dataclass
class EncodingInfo:
    encoding: str
    confidence: float
    scores: dict[str, float] = field(default_factory=dict)
    bom: bool = False
    size: int = 0

    @property
    def size_human(self) -> str:
        if self.size < 1024:
            return f"{self.size}B"
        if self.size < 1024 * 1024:
            return f"{self.size / 1024:.0f}KB"
        return f"{self.size / 1024 / 1024:.1f}MB"

    def describe(self) -> str:
        bom = "，含 BOM" if self.bom else ""
        return f"{self.encoding}（判定优势 {self.confidence:+.2f}{bom}，{self.size_human}）"


class DecodeError(RuntimeError):
    """编码无法可靠判定。宁可停下，也不要产出乱码音频。"""


def _analyse(chunk: str) -> tuple[float, float, float]:
    """返回 (可读率, 常用字率, 综合分)。"""
    if not chunk:
        return 0.0, 0.0, 0.0
    good = bad = 0.0
    cjk = cjk_common = 0
    for ch in chunk:
        o = ord(ch)
        if ch in "\n\r\t":
            good += 1
        elif _LOW <= o <= _HIGH:
            good += 1
        elif 0x4E00 <= o <= 0x9FFF:
            good += 1
            cjk += 1
            if ch in _COMMON_SET:
                cjk_common += 1
        elif ch in _PUNCT:
            good += 1
        elif 0x3000 <= o <= 0x303F or 0xFF01 <= o <= 0xFF65:
            good += 1
        elif 0xE000 <= o <= 0xF8FF or o == 0xFFFD or o < _LOW:
            bad += 2          # 私用区 / 替换符 / 控制符，乱码的典型特征
        else:
            bad += 0.2        # 拉丁扩展、注音、emoji：中性偏罚
    total = good + bad
    printable = good / total if total else 0.0
    common = (cjk_common / cjk) if cjk >= 20 else 0.0
    score = printable if cjk < 20 else 0.5 * printable + 0.5 * common
    return printable, common, score


def _normalize_newlines(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _candidates(sample: bytes) -> list[_Cand]:
    has_bom_utf8 = sample[:3] == b"\xef\xbb\xbf"
    has_bom_utf16 = sample[:2] in (b"\xff\xfe", b"\xfe\xff")
    out: list[_Cand] = []
    for enc in CANDIDATES:
        if enc == "utf-8" and has_bom_utf8:
            enc_use = "utf-8-sig"
        else:
            enc_use = enc
        if enc == "utf-16" and not has_bom_utf16:
            continue          # 无 BOM 的 "utf-16" 是靠猜的，不参与
        try:
            text = sample.decode(enc_use)
        except (UnicodeDecodeError, LookupError):
            continue
        printable, common, score = _analyse(text)
        out.append(_Cand(enc=enc_use, text=text, printable=printable, common=common, score=score))
    return out


def detect_bytes(raw: bytes) -> tuple[str, EncodingInfo]:
    """判定编码并解码。"""
    if not raw:
        raise DecodeError("文件是空的")

    sample = raw[:400_000]
    cands = _candidates(sample)
    if not cands:
        raise DecodeError(
            "没有任何候选编码能解开这个文件（utf-8/gb18030/big5/utf-16 全部失败）。"
            "可能是二进制文件、加密文件，或用了罕见的编码。"
        )

    # 同分时按 CANDIDATES 顺序取前者（gb18030 能解开 GBK，两者文本相同时优先 gb18030 的位次）
    ordered = sorted(cands, key=lambda c: (-c.score, c.enc))
    best = ordered[0]

    # 只有解出**不同文本**的候选才算竞争者
    rivals = [c for c in ordered[1:] if c.text != best.text]
    second = rivals[0].score if rivals else 0.0
    advantage = best.score - second

    scores_map = {c.enc: round(c.score, 2) for c in ordered}
    ok = (
        best.printable >= 0.90          # 没有成片的控制符/替换符
        and best.score >= 0.55          # 中文正文的典型值 0.75±
        and advantage >= 0.12           # 明显优于错误解码
    )

    if not ok:
        detail = "，".join(f"{k}={v:.2f}" for k, v in scores_map.items())
        raise DecodeError(
            f"编码无法可靠判定（最佳 {best.enc} 得分 {best.score:.2f}，"
            f"领先第二名 {advantage:+.2f}，要求 ≥0.12 且可读率 ≥0.90）。\n"
            f"各候选：{detail}。\n"
            f"这种情况继续跑只会产出一章乱码音频，所以在这里停下。"
            f"确认编码后用 --encoding 显式指定。"
        )

    # 全量解码
    try:
        full = raw.decode(best.enc)
    except UnicodeDecodeError as exc:
        raise DecodeError(
            f"前 400KB 判定为 {best.enc}，但全量解码在第 {exc.start} 字节失败——"
            f"文件可能前后混用了两种编码。建议先手工转码。"
        ) from exc

    info = EncodingInfo(
        encoding=best.enc,
        confidence=advantage,
        scores=scores_map,
        bom=raw[:3] == b"\xef\xbb\xbf" or raw[:2] in (b"\xff\xfe", b"\xfe\xff"),
        size=len(raw),
    )
    return full, info


def read_book(path: str | Path, encoding: str | None = None) -> tuple[str, EncodingInfo]:
    """读入整本书，返回（已统一换行的文本, 编码信息）。"""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"找不到文件：{p}")
    raw = p.read_bytes()

    if encoding:
        try:
            text = raw.decode(encoding)
        except (UnicodeDecodeError, LookupError) as exc:
            raise DecodeError(f"按你指定的 {encoding} 解不开：{exc}") from exc
        info = EncodingInfo(encoding=encoding, confidence=1.0, bom=False, size=len(raw))
    else:
        text, info = detect_bytes(raw)

    return _normalize_newlines(text), info


def tidy(text: str) -> str:
    """清掉不可见字符和行尾空白。只动"排版"，不动任何一个正文字。

    刻意不做的事：不合并空行、不动全角空格的位置、不删标点——这些都属于改写原文。
    """
    out = []
    for line in text.split("\n"):
        line = line.replace("\u200b", "").replace("\ufeff", "").replace("\u00a0", " ")
        out.append(line.rstrip())
    return "\n".join(out)
