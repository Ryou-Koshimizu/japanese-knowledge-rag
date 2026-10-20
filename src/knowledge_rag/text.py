"""日本語テキストの正規化と、辞書を使わない語の切り出し。

形態素解析の辞書を持たずに日本語を検索するため、文字の種類（漢字・カタカナ・英数字・
ひらがな）が変わる位置で区切る。ひらがなの連なりは多くが助詞・語尾なので検索語にしない。
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable

_SPACE = re.compile(r"\s+")
_MONTHS = re.compile(r"[ヶケカヵ箇]月")
_NUM_COMMA = re.compile(r"(?<=\d),(?=\d{3})")


def normalize(text: str) -> str:
    """検索・照合用の正規化。全角英数→半角、互換文字の統一、小文字化、空白の圧縮。"""
    text = unicodedata.normalize("NFKC", text).lower()
    text = _MONTHS.sub("か月", text)
    return _SPACE.sub(" ", text).strip()


def compact(text: str) -> str:
    """引用の一字一句照合用。正規化した上で空白・桁区切りのカンマ・改行を除く。"""
    text = normalize(text)
    text = _NUM_COMMA.sub("", text)
    return re.sub(r"[\s　]", "", text)


def _kind(ch: str) -> str:
    code = ord(ch)
    if ch in "々〆ヶ" or 0x4E00 <= code <= 0x9FFF or 0x3400 <= code <= 0x4DBF:
        return "kanji"
    if 0x30A0 <= code <= 0x30FF or ch == "ー":
        return "kata"
    if 0x3040 <= code <= 0x309F:
        return "hira"
    if ch.isascii() and ch.isalnum():
        return "alnum"
    return "other"


def script_runs(text: str) -> list[tuple[str, str]]:
    """文字の種類ごとの連なりに分ける。(種類, 文字列) のリスト。"""
    runs: list[tuple[str, str]] = []
    for ch in normalize(text):
        kind = _kind(ch)
        if runs and runs[-1][0] == kind:
            runs[-1] = (kind, runs[-1][1] + ch)
        else:
            runs.append((kind, ch))
    return runs


# 質問文にだけ現れ、規程の本文には手がかりにならない語。
QUESTION_WORDS = frozenset(
    {
        "何",
        "何日",
        "何歳",
        "何時",
        "何時間",
        "何円",
        "何回",
        "何年",
        "何人",
        "何割",
        "何営業日",
        "何文字",
        "何桁",
        "何分",
        "何か月",
        "誰",
        "教",
        "場合",
        "必要",
        "可能",
        "方法",
        "具体的",
        "大丈夫",
        "最大",
        "最低",
        "以内",
        "以上",
        "以下",
        "当社",
        "会社",
        "社内",
        "規程",
        "規定",
        "ルール",
    }
)


_SINGLE_STOP = frozenset("何誰教思良知聞言方事物時人的")


def query_terms(text: str) -> list[str]:
    """検索語を取り出す。

    - 漢字の連なりはそのまま1語（1文字でもよい。「祝」「戻」のような1文字が手がかりになる）。
      3文字以上なら2文字ずつの語も加える（「時短勤務」→「勤務」など、複合語の部分一致のため）
    - カタカナ・英数字は2文字以上の連なり
    """
    terms: list[str] = []
    for kind, raw_run in script_runs(text):
        run = raw_run
        if kind == "kanji":
            run = raw_run.removeprefix("何") if len(raw_run) > 1 else raw_run
            if run in QUESTION_WORDS or (len(run) == 1 and run in _SINGLE_STOP):
                continue
            terms.append(run)
            if len(run) >= 3:
                terms.extend(
                    run[i : i + 2]
                    for i in range(len(run) - 1)
                    if run[i : i + 2] not in QUESTION_WORDS
                )
        elif kind in ("kata", "alnum") and len(run) >= 2 and run not in QUESTION_WORDS:
            terms.append(run)
    return dedupe(terms)


def trigrams(term: str) -> list[str]:
    return [term[i : i + 3] for i in range(len(term) - 2)]


def dedupe(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out
