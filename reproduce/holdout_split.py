#!/usr/bin/env python
"""held-out 双栏：把 dev（做过错题分析）/ test（从没分析过）两组分开报准确率，反过拟合。

test 集 = 方法定型后才加入、从未做过错题分析的文档（默认 = 最后补传的 89 篇，绝对干净）。
若 test 篇还没跑 query（刚建图未查询），test 列会是空——说明要先给这些篇补跑 query。

用法：
  python reproduce/holdout_split.py \
      --eval /root/autodl-tmp/eval_pdsg_uniform/llm_evaluation_results.json \
      --qa-dir /root/autodl-tmp/DocBench_subset \
      --methods paperbase,ours3            # 想比哪几条就写哪几条
  # 自定义 test 集： --test-ids "7,11,21,..."
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

# 最后补传、方法定型后才加入、从未做过错题分析的 89 篇 = 干净 held-out 测试集
DEFAULT_TEST = (
    "7 11 21 52 53 56 57 59 60 64 65 66 67 69 71 73 74 78 81 84 88 89 91 93 100 105 106 "
    "109 110 114 115 120 128 129 130 131 136 137 138 144 151 156 157 162 163 180 181 182 "
    "183 184 185 186 188 189 191 192 193 194 195 196 197 198 200 201 202 203 204 205 206 "
    "207 208 209 210 211 212 213 214 215 217 218 219 220 221 222 223 224 225 226 228"
).split()


def norm(q):
    return " ".join(str(q).split()).strip().lower()


def main():
    ap = argparse.ArgumentParser(description="held-out 双栏（dev/test 分列报准确率）")
    ap.add_argument("--eval", required=True)
    ap.add_argument("--qa-dir", required=True)
    ap.add_argument("--methods", default="paperbase,ours3")
    ap.add_argument("--test-ids", default=" ".join(DEFAULT_TEST))
    args = ap.parse_args()

    test_ids = set(args.test_ids.replace(",", " ").split())
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]

    docmap = {p.name: p.parent.name for p in Path(args.qa_dir).glob("*/*.pdf")}
    q2type = {}
    for qa in Path(args.qa_dir).glob("*/*_qa.jsonl"):
        for ln in open(qa, encoding="utf-8"):
            ln = ln.strip()
            if ln:
                d = json.loads(ln)
                q2type[norm(d.get("question", ""))] = d.get("type", "?")

    ev = json.load(open(args.eval, encoding="utf-8"))
    overall = defaultdict(lambda: defaultdict(lambda: [0, 0]))   # split -> method -> [对, 总]
    meta = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    for r in ev:
        if not r.get("success") or r.get("accuracy") is None:
            continue
        m = r.get("method")
        if m not in methods:
            continue
        folder = docmap.get(str(r.get("doc_id")), str(r.get("doc_id")))
        split = "test" if folder in test_ids else "dev"
        c = int(r["accuracy"])
        overall[split][m][0] += c
        overall[split][m][1] += 1
        if q2type.get(norm(r.get("question"))) == "meta-data":
            meta[split][m][0] += c
            meta[split][m][1] += 1

    def pct(ct):
        return f"{ct[0]}/{ct[1]} ({100 * ct[0] / ct[1]:.0f}%)" if ct[1] else "—（未查询）"

    print(f"\n===== held-out 双栏  (test = {len(test_ids)} 篇从未做过错题分析) =====")
    print(f"{'method':<12}{'DEV 总体':<20}{'TEST 总体':<20}{'DEV meta':<18}{'TEST meta':<18}")
    print("-" * 88)
    for m in methods:
        print(f"{m:<12}{pct(overall['dev'][m]):<20}{pct(overall['test'][m]):<20}"
              f"{pct(meta['dev'][m]):<18}{pct(meta['test'][m]):<18}")
    print("\n解读：若 test 列（尤其 meta）相对 paperbase 的涨幅与 dev 列一致 → 方法在【从没分析过】"
          "的文档上同样有效 = 真泛化、非过拟合。test 列为空 = 这些篇还没跑 query，需先补跑。")


if __name__ == "__main__":
    main()
