# -*- coding: utf-8 -*-
"""用真实错题(dump)验证 dg_core.ground:对有 content_list 的测试文档,
看框架现在产出的 note 是否包含金标答案(数字/页码),量化"修好了多少"。"""
import json
import os
import re
import sys
import glob

REPRO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPRO)
from dg_core import ground, build_doc_model  # noqa

DUMP = os.path.join(REPRO, "diag_meta_errors_dump.json")
DATA = os.path.join(REPRO, "..", "data", "DocBench_download")
CLDIR = os.path.join(REPRO, "_testdata", "content_lists")
TEST_DOCS = {"0", "19", "28", "45", "47", "114", "70", "151", "152", "154", "164", "220"}


def pdf_of(did):
    g = glob.glob(os.path.join(DATA, did, "*.pdf"))
    return g[0] if g else None


def cl_of(did):
    p = pdf_of(did)
    if not p:
        return None
    stem = os.path.splitext(os.path.basename(p))[0]
    c = os.path.join(CLDIR, f"{stem}_content_list.json")
    return c if os.path.exists(c) else None


def gold_numbers(gold):
    return set(re.findall(r"\d+", str(gold)))


def hit(note, gold):
    """启发式:金标里的数字是否出现在 note 里(定位/计数题足够指示)。"""
    gn = gold_numbers(gold)
    if not gn:
        # 非数字金标(标题/文本)——看金标前若干词是否在 note
        g = re.sub(r"[^a-z0-9 ]", " ", str(gold).lower()).split()[:4]
        return bool(g) and all(w in note.lower() for w in g if len(w) > 2)
    nn = set(re.findall(r"\d+", note))
    return bool(gn & nn)


d = json.load(open(DUMP, encoding="utf-8"))
items = d["A_or_C_fired_wrong"] + d["B_notfired_baseline_wrong"]
items = [it for it in items if it["doc_id"] in TEST_DOCS]

print(f"测试文档 {sorted(TEST_DOCS, key=int)};可评错题 {len(items)} 道\n")
fixed = abstain = stillwrong = 0
for it in sorted(items, key=lambda x: int(x["doc_id"])):
    did, q, gold = it["doc_id"], it["question"], it["gold"]
    pdf, cl = pdf_of(did), cl_of(did)
    fact = ground(q, pdf, cl) if pdf else None
    if fact is None:
        abstain += 1
        tag, detail = "ABSTAIN", "(退回基座)"
    else:
        if hit(fact.note, gold):
            fixed += 1
            tag, detail = "FIXED  ", fact.kind
        else:
            stillwrong += 1
            tag, detail = "MISS   ", fact.kind
    print(f"[{tag}] doc{did:>3} {detail:<14} gold={str(gold)[:34]!r}")
    print(f"          Q: {q[:78]}")
    if fact:
        print(f"          note: {fact.note[:110]}")

print(f"\n==== 汇总(可评 {len(items)} 道)====")
print(f"  FIXED(note 含金标答案): {fixed}")
print(f"  ABSTAIN(弃权退回基座) : {abstain}")
print(f"  MISS(产出但未含金标)  : {stillwrong}")
