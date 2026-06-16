#!/usr/bin/env python
"""候选新方向的零成本"失分规模扫描"——只读 *_qa.jsonl，不调任何 API、不跑检索。

回答 plan §9 候选动手前必答的问题："这个方向到底有多少道题用得上？"（<15 题就别碰，
见 §9b 冷启动纪律）。仿 doc_anchor 上线前那次本地扫描：报命中题数 + 按题型分布 + 抽样
若干例子供人工核对精度（误命中多就回来收紧正则，别急着写功能）。

当前覆盖两个候选：
  page_loc (§9 候选1)：问"X 在第几页 / on which page"——可确定性定位 content_list 的 page_idx。
  list_ans (§9 候选2)：gold answer 是多项清单，而 query.py 把 response_type 硬编码成
           "One Sentence"（query.py:387/394/400/420），疑似截断 completeness——可按题型放开。

用法：
  python reproduce/scan_candidates.py --qa-dir /root/autodl-tmp/DocBench_subset
  python reproduce/scan_candidates.py --qa-dir ... --show 12   # 每类多看几个例子
"""
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path


# ── 候选1：页码定位题 ──────────────────────────────────────────────────────
# 保守口径：明确问"哪一页/第几页/页码"。不认泛指的 "page" 名词（如 "home page"），
# 也不命中 "how many pages"（数量题归 DSG）。
_PAGE_LOC_RE = re.compile(
    r"\bon\s+(which|what)\s+page\b"
    r"|\b(which|what)\s+page\s+(number\s+)?(does|do|is|are|can|contains?|shows?|has)\b"
    r"|\bwhat\s+is\s+the\s+page\s+number\b",
    re.IGNORECASE,
)


def is_page_location(question: str) -> bool:
    """问题是否在问"某内容在第几页"。"""
    return bool(_PAGE_LOC_RE.search(question or ""))


# ── 候选2：答案为多项清单的题 ─────────────────────────────────────────────
# 问句线索：明确要"列举/有哪些/列出/点名多个"。
_LIST_CUE_RE = re.compile(
    r"\blist\b|\bwhat\s+are\b|\bname\s+(all|the|two|three|four|five)\b"
    r"|\benumerate\b|\bidentify\s+(all|the)\b",
    re.IGNORECASE,
)
# 纯数字/货币/百分比里的逗号不算清单分隔（避免 "$1,234,567" 误判为多项）。
_NUMERICISH = re.compile(r"^[\s\$€£%\d.,:/()+\-]+$")


def _answer_is_multi(answer: str) -> bool:
    """gold answer 结构上是否像"多项清单"。"""
    a = str(answer or "").strip()
    if not a:
        return False
    if len([ln for ln in a.splitlines() if ln.strip()]) >= 3:   # 换行多项
        return True
    if re.search(r"(^|\n)\s*(\d+[.)]|[-*•])\s+\S", a):           # 编号 / 项目符号
        return True
    if a.count(";") >= 2:                                       # 分号分隔 >=3 段
        return True
    if not _NUMERICISH.match(a):                                # 逗号分隔 >=4 段且非纯数字串
        if len([s for s in a.split(",") if s.strip()]) >= 4:
            return True
    return False


def is_list_answer(question: str, answer: str) -> bool:
    """问句要清单，或 gold answer 结构上是多项——其一即算。"""
    return bool(_LIST_CUE_RE.search(question or "")) or _answer_is_multi(answer)


# ── 扫描 + 报告 ───────────────────────────────────────────────────────────
def iter_qa(qa_dir: str):
    """遍历 <id>/<id>_qa.jsonl，逐题 yield (doc, question, answer, type)。"""
    for qa in sorted(Path(qa_dir).glob("*/*_qa.jsonl")):
        with open(qa, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                yield qa.parent.name, d.get("question", ""), d.get("answer", ""), d.get(
                    "type", "unknown"
                )


def report(name, hits, total, by_type, examples, show, threshold):
    print(f"\n══ 候选 [{name}] ══")
    pct = f"（{100 * hits / total:.1f}%）" if total else ""
    print(f"命中 {hits} / 共 {total} 题{pct}")
    print(f"按题型：{dict(by_type)}")
    if hits >= threshold:
        print(f"判定：≥{threshold} → 值得做零成本验证")
    else:
        print(f"判定：<{threshold} → 面太窄，放弃（§9b 纪律：别花钱）")
    print(f"抽样 {min(show, len(examples))} 例（核对精度，误命中多就回来收紧正则）：")
    for doc, q, a in examples[:show]:
        print(f"  [doc {doc}] Q: {q[:90]}")
        print(f"            A: {str(a)[:70]}")


def main():
    ap = argparse.ArgumentParser(description="候选方向零成本失分规模扫描（只读 qa.jsonl）")
    ap.add_argument("--qa-dir", required=True, help="含 <id>/<id>_qa.jsonl 的目录")
    ap.add_argument("--show", type=int, default=8, help="每个候选抽样展示几道题")
    ap.add_argument("--threshold", type=int, default=15, help="低于此命中数则建议放弃")
    args = ap.parse_args()

    total = 0
    page = {"hits": 0, "by_type": defaultdict(int), "ex": []}
    lst = {"hits": 0, "by_type": defaultdict(int), "ex": []}
    for doc, q, a, t in iter_qa(args.qa_dir):
        total += 1
        if is_page_location(q):
            page["hits"] += 1
            page["by_type"][t] += 1
            page["ex"].append((doc, q, a))
        if is_list_answer(q, a):
            lst["hits"] += 1
            lst["by_type"][t] += 1
            lst["ex"].append((doc, q, a))

    if total == 0:
        print(f"未在 {args.qa_dir} 找到 *_qa.jsonl")
        return
    print(f"扫描完成：共 {total} 题")
    report("page_loc 页码定位(§9候选1)", page["hits"], total, page["by_type"],
           page["ex"], args.show, args.threshold)
    report("list_ans 多项清单答案(§9候选2)", lst["hits"], total, lst["by_type"],
           lst["ex"], args.show, args.threshold)


if __name__ == "__main__":
    main()
