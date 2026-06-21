#!/usr/bin/env python
"""逐题对比两个 method 的对错翻转——零成本离线归因，不调任何 API。

回答"B 比 A 多对的 N 题到底是哪些、是不是机制带来的"：
  - 列出 B 赢回的题（A 错→B 对）与 B 搞砸的题（A 对→B 错），按题型分组统计；
  - 自动读取 qa_results_<B>.json 里的 doc_meta_used / outline_used / anchor_used /
    ncg_used / vlm_used / rr_triggered 标记——验证翻转是否真的发生在"被注入统计量/
    抄了前页/接地了被点名元素/看了图/补了检索"的题上；
  - 末尾给出净变化与不一致对数，用于显著性粗判：净变化远小于不一致总数 = 噪声味重，
    净变化接近不一致总数 = 一边倒的真信号。

用法：
  python reproduce/diff_results.py \
      --eval /root/autodl-tmp/eval_out/llm_evaluation_results.json \
      --qa-dir /root/autodl-tmp/DocBench_subset \
      --a base_t0 --b docmeta_v2
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path


def norm(q: str) -> str:
    """问题文本规范化（与 eval_by_type.py 一致），用于跨文件对齐。"""
    return " ".join(str(q).split()).strip().lower()


def load_eval(path):
    """读评测结果 -> {(method, doc_id, norm(question)): 0/1}。"""
    with open(path, encoding="utf-8") as f:
        results = json.load(f)
    acc = {}
    for r in results:
        if not r.get("success") or r.get("accuracy") is None:
            continue
        acc[(r.get("method"), str(r.get("doc_id")), norm(r.get("question")))] = int(
            r["accuracy"]
        )
    return acc


def load_types(qa_dir):
    """{norm(question): type}，来自各文档的 qa.jsonl。"""
    q2type = {}
    for qa in Path(qa_dir).glob("*/*_qa.jsonl"):
        with open(qa, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    d = json.loads(line)
                    q2type[norm(d.get("question", ""))] = d.get("type", "unknown")
    return q2type


def build_doc_map(qa_dir):
    """doc_id -> 文件夹名。评测器的 doc_id 是 PDF 文件名（如 NYSE_ACN_2020.pdf），
    而结果文件存在编号文件夹下（如 79/qa_results_*.json），这里建立映射。"""
    m = {}
    for p in Path(qa_dir).glob("*/*.pdf"):
        m[p.name] = p.parent.name
    return m


def load_flags(qa_dir, doc_map, method, doc_id, q_normed):
    """读 qa_results_<method>.json 里这道题的机制标记（缺文件/缺字段返回空 dict）。"""
    folder = doc_map.get(doc_id, doc_id)  # 映射不到就按原样当文件夹名试
    p = Path(qa_dir) / folder / f"qa_results_{method}.json"
    if not p.exists():
        return {}
    try:
        with open(p, encoding="utf-8") as f:
            for rec in json.load(f):
                if norm(rec.get("question")) == q_normed:
                    return {
                        k: rec[k]
                        for k in ("doc_meta_used", "locate_used", "outline_used",
                                  "anchor_used", "ncg_used", "vlm_used", "vlm_trigger",
                                  "rr_triggered")
                        if k in rec
                    }
    except Exception:
        pass
    return {}


def _triggered(flags):
    """B 的程序（DSG 或 Locate）是否在这题真开火。"""
    return bool(flags.get("doc_meta_used")) or bool(flags.get("locate_used"))


def print_breakdown(common, a_acc, b_acc, qa_dir, doc_map, method_b, q2type):
    """机制贡献分解：把共同题按【是否被程序触发】×【对错/翻转】分类，给数量与占比。

    用途：证明增益集中在"触发∩翻对"（真信号），而非"未触发∩翻转"（温度噪声）；
    并把"触发∩答错"逐题列出 = 方法的边界/瓶颈。
    """
    N = len(common)
    T = Tc = Tw = up = down = notT_flip = 0
    Tw_qs = []
    trig_type = defaultdict(int)
    for key in common:
        va, vb = a_acc[key], b_acc[key]
        trig = _triggered(load_flags(qa_dir, doc_map, method_b, key[0], key[1]))
        flip = va != vb
        if trig:
            T += 1
            trig_type[q2type.get(key[1], "?")] += 1
            if vb == 1:
                Tc += 1
            else:
                Tw += 1
                Tw_qs.append(key)
            if flip and vb > va:
                up += 1
            elif flip and vb < va:
                down += 1
        elif flip:
            notT_flip += 1

    def pct(x, base):
        return f"{100 * x / base:.1f}%" if base else "—"

    print(f"\n===== 机制贡献分解  B={method_b}  (共同题 {N}) =====")
    print(f"触发数 T（程序开火）            = {T:4d}  ({pct(T, N)} of 全部)")
    print(f"  ├ 触发∩答对                   = {Tc:4d}  ({pct(Tc, T)} of 触发)")
    print(f"  └ 触发∩答错（边界/瓶颈）      = {Tw:4d}  ({pct(Tw, T)} of 触发)")
    print(f"触发∩翻对（A错→B对，真赢回）   = {up:4d}  ({pct(up, N)} of 全部)")
    print(f"触发∩翻错（A对→B错，帮倒忙）   = {down:4d}")
    print(f"未触发∩翻转（纯温度噪声）       = {notT_flip:4d}")
    print(f"机制净贡献（翻对 − 翻错）       = {up - down:+d}")
    print(f"触发命中率（触发∩答对 / T）     = {pct(Tc, T)}")
    print("\n触发题按题型（数量 / 占触发）:")
    for t in sorted(trig_type):
        print(f"  {t:<16} {trig_type[t]:3d}  ({pct(trig_type[t], T)})")
    print(f"\n--- 触发∩答错（边界瓶颈，{len(Tw_qs)} 题）---")
    for d, q in Tw_qs:
        print(f"[doc {d}] ({q2type.get(q, '?')}) {q[:100]}")


def main():
    ap = argparse.ArgumentParser(description="逐题对比两个 method 的对错翻转")
    ap.add_argument("--eval", required=True, help="llm_evaluation_results.json 路径")
    ap.add_argument("--qa-dir", required=True, help="含 <id>/<id>_qa.jsonl 的目录")
    ap.add_argument("--a", required=True, help="基准 method（如 base_t0）")
    ap.add_argument("--b", required=True, help="对比 method（如 docmeta_v2）")
    ap.add_argument("--breakdown", action="store_true",
                    help="额外打印机制贡献分解表（触发/翻对/翻错/噪声 的数量与占比）")
    args = ap.parse_args()

    acc = load_eval(args.eval)
    q2type = load_types(args.qa_dir)
    doc_map = build_doc_map(args.qa_dir)
    a_acc = {(d, q): v for (m, d, q), v in acc.items() if m == args.a}
    b_acc = {(d, q): v for (m, d, q), v in acc.items() if m == args.b}
    common = sorted(set(a_acc) & set(b_acc))
    if not common:
        print(f"两个 method 没有共同题目（A={len(a_acc)} 题，B={len(b_acc)} 题）")
        return

    wins, losses = [], []          # B 赢回 / B 搞砸
    by_type = defaultdict(lambda: [0, 0])
    for key in common:
        va, vb = a_acc[key], b_acc[key]
        if va == vb:
            continue
        t = q2type.get(key[1], "?")
        if vb > va:
            wins.append(key)
            by_type[t][0] += 1
        else:
            losses.append(key)
            by_type[t][1] += 1

    sa = sum(a_acc[k] for k in common)
    sb = sum(b_acc[k] for k in common)
    n = len(common)
    print(f"\n对比 {args.a} -> {args.b}    共同题数 {n}"
          f"（A 独有 {len(a_acc) - n}，B 独有 {len(b_acc) - n}）")
    print(f"总分: {args.a} {sa}/{n} ({100 * sa / n:.1f}%)"
          f"  ->  {args.b} {sb}/{n} ({100 * sb / n:.1f}%)")
    print(f"翻转: B 赢回 {len(wins)} 题 / B 搞砸 {len(losses)} 题"
          f"  净 {len(wins) - len(losses):+d}（不一致共 {len(wins) + len(losses)}）")

    print("\n按题型（赢回/搞砸）:")
    for t in sorted(by_type):
        w, l = by_type[t]
        print(f"  {t:<16} +{w} / -{l}")

    for title, items in (("B 赢回的题", wins), ("B 搞砸的题", losses)):
        print(f"\n--- {title} ({len(items)}) ---")
        for d, q in items:
            flags = load_flags(args.qa_dir, doc_map, args.b, d, q)
            ftxt = " ".join(f"{k}={v}" for k, v in flags.items()) or "(无标记)"
            print(f"[doc {d}] ({q2type.get(q, '?')}) {ftxt}")
            print(f"    {q[:110]}")

    if args.breakdown:
        print_breakdown(common, a_acc, b_acc, args.qa_dir, doc_map, args.b, q2type)


if __name__ == "__main__":
    main()
