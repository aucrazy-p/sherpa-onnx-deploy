"""生成测试书：覆盖真实中文小说里最常见的五种排版与编码。

目的不是"造数据"，是把已知会出问题的情况先造出来——切不出来、解不对、引号不闭合。
跑完 self-check 再拿去验 cut。

    python tools/make_fixtures.py            # 写到 .fixtures/
"""

from __future__ import annotations

import random
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / ".fixtures"

NAMES = ["陈默", "老周", "阿丘", "林小满", "赵铁柱"]

BODY = [
    "雨下了一整夜，城里的灯在积水里碎成一片一片。他从便利店出来，手里拎着一袋热腾腾的包子。",
    "街对面的钟楼敲了十二下，声音闷闷的，像是被雨裹住了。",
    "他忽然想起很多年前的一个晚上，也是这样的雨，也是这样的钟声。",
    "只是那时候身边还站着一个人，会一边抱怨衣服湿了，一边把伞往他这边偏。",
    "风从巷口灌进来，带着一股铁锈和灰尘混在一起的味道。",
]

TALK = [
    ("{a}", "你还没走？", "抬头看了他一眼，把烟摁灭在栏杆上。"),
    ("{a}", "等你啊。", "笑了一下，露出被烟熏黄的牙。"),
    ("{a}", "这天儿，走不了。", "把外套的领子竖起来。"),
    ("{a}", "那就等等吧。", "把手插进兜里。"),
]


def chapter(idx: int, title: str, paras: int = 12, seed: int = 0) -> str:
    rnd = random.Random(seed + idx)
    out = [title, ""]
    for i in range(paras):
        r = rnd.random()
        if r < 0.45:
            who, said, act = TALK[i % len(TALK)]
            name = NAMES[(idx + i) % len(NAMES)]
            out.append(f"「{said.format(a=name)}」{name}{act}")
        elif r < 0.65:
            out.append(f"「{rnd.choice(BODY)}」")
        else:
            out.append(rnd.choice(BODY))
        if i % 4 == 3:
            out.append("")
    return "\n".join(out) + "\n"


def book_utf8_chapters() -> str:
    parts = []
    for i in range(1, 7):
        parts.append(chapter(i, f"第{i}章 " + ["夜行", "旧城", "钟楼", "雨停", "来客", "归途"][i - 1]))
    return "\n".join(parts)


def book_gbk_volumes() -> str:
    parts = []
    for v in (1, 2):
        for i in range(1, 4):
            parts.append(chapter(i + v * 10, f"第{v}卷 第{i}章 " + ["风起", "旧事", "长夜"][i - 1]))
    return "\n".join(parts)


def book_no_marker() -> str:
    """完全没有章节标记，只用空行分隔。"""
    parts = []
    for i in range(1, 24):
        parts.append(chapter(i, "", paras=8, seed=i))
    return "\n\n".join(parts)


def book_big5() -> str:
    """繁体 + Big5 编码。检验「不会把 Big5 静默解成 GB18030 乱码」。"""
    trad = [
        "雨下了一整夜，城裡的燈在積水裡碎成一片一片。他從便利店出來，手裡拎著一袋熱騰騰的包子。",
        "「你還沒走？」他抬頭看了他一眼，把菸摁熄在欄杆上。",
        "「等你啊。」他笑了一下，露出被菸熏黃的牙。",
        "街對面的鐘樓敲了十二下，聲音悶悶的，像是被雨裹住了。",
        "他忽然想起很多年前的一個晚上，也是這樣的雨，也是這樣的鐘聲。",
        "只是那時候身邊還站著一個人，會一邊抱怨衣服濕了，一邊把傘往他這邊偏。",
        "風從巷口灌進來，帶著一股鐵鏽和灰塵混在一起的味道。",
        "「這天兒，走不了。」他把外套的領子豎起來。",
        "「那就等等吧。」他把手插進兜裡。",
        "「你說，人這一輩子，到底在等什麼？」他忽然問。",
        "他沒有回答，只是看著雨，看了很久很久。",
        "遠處有車開過，燈光在水裡拉出一條長長的尾巴，又很快散掉了。",
    ]
    parts = []
    for i in range(1, 5):
        parts.append(f"第{i}章 " + ["夜行", "舊城", "鐘樓", "雨停"][i - 1])
        parts.append("")
        for j in range(12):
            parts.append(trad[(i * 3 + j) % len(trad)])
        parts.append("")
    return "\n".join(parts)


def book_broken_quotes() -> str:
    """ASCII 引号 + 故意留一处不闭合。"""
    parts = ["第1章 错位", ""]
    parts += ['"你到底想干什么？"她把手里的杯子放下。', ""]
    parts += ['他沉默了一会儿，说："我在等人。"', ""]
    parts += ['"等谁？"', ""]
    parts += ['"一个不会来的人。"他说完就笑了，"她走了三年了。', ""]     # 引号不闭合
    parts += ['窗外开始下雨，雨点敲在玻璃上，一下一下的。', ""]
    parts += ["第2章 天亮", ""]
    parts += ['"走吧。"她说。', ""]
    parts += ['他点点头，把桌上那张纸折好，放进内袋。', ""]
    return "\n".join(parts)


def book_num_chapters() -> str:
    """纯数字编号标题（`001 梦回七九`）——中文网文的第二种主流格式。

    混入带 ？！ 的标题，因为这类标题会被"标题行不含句末标点"的规则误杀。
    """
    titles = ["梦回七九", "身世", "谁不会捆绑？", "技术难题", "一百年了！", "回家"]
    parts = []
    for i, t in enumerate(titles, start=1):
        parts.append(chapter(i, f"{i:03d} {t}", paras=10, seed=i))
    return "\n".join(parts)


def book_large_utf8(n_chapters: int = 460) -> tuple[str, bytes]:
    """大文件：把字节偏移 400000 精确推到某个多字节汉字的中间。

    复现真实 bug——取前 400KB 做样本时若硬切，切点落在汉字中间会导致严格解码失败，
    正常文件被误判成"解不开"。几千字的测试书永远碰不到，只有真书会。
    """
    parts = [chapter(i, f"第{i}章 长夜", paras=12, seed=i) for i in range(1, n_chapters + 1)]
    text = "\n".join(parts)
    raw = text.encode("utf-8")
    for k in range(4):
        padded = b" " * k + raw
        if len(padded) > 400_000 and 0x80 <= padded[400_000] <= 0xBF:
            return text, padded
    return text, raw


def main() -> None:
    OUT.mkdir(exist_ok=True)
    jobs = [
        ("utf8-chapters.txt", book_utf8_chapters(), "utf-8"),
        ("gbk-volumes.txt", book_gbk_volumes(), "gbk"),
        ("no-marker.txt", book_no_marker(), "utf-8"),
        ("big5-chapters.txt", book_big5(), "big5"),
        ("utf16-chapters.txt", book_utf8_chapters(), "utf-16"),
        ("broken-quotes.txt", book_broken_quotes(), "utf-8"),
        ("num-chapters.txt", book_num_chapters(), "utf-8"),
    ]
    for name, text, enc in jobs:
        p = OUT / name
        p.write_bytes(text.encode(enc))
        print(f"{name:24} {enc:8} {p.stat().st_size:>8,} 字节  {len(text):>7,} 字")

    # 大文件直接写字节（内部已做过偏移校准）
    _t, raw = book_large_utf8()
    p = OUT / "large-utf8.txt"
    p.write_bytes(raw)
    print(f"{'large-utf8.txt':24} {'utf-8':8} {p.stat().st_size:>8,} 字节  "
          f"{len(raw.decode('utf-8')):>7,} 字")


if __name__ == "__main__":
    main()
