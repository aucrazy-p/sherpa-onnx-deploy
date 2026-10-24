"""bookpipe — 小说转有声书的前置层。

只做文本：编码探测 → 切章 → 切段（旁白/对白分离）→ （后续）角色归属。
不认识任何 TTS 引擎，不联网，不依赖第三方库。

用法：
    python -m bookpipe inspect 某本书.txt
    python -m bookpipe inspect 某本书.txt --jsonl out/segments.jsonl
"""

__version__ = "0.1.0"
