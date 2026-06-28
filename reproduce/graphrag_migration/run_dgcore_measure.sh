#!/bin/bash
# 一键:50篇子集 重跑 dgcore(改进版) -> 清旧测评 -> eval -> 三方对比 + L3校准。
# 用法(服务器): cd /root/autodl-tmp/rag-L1 && git pull
#   nohup bash reproduce/graphrag_migration/run_dgcore_measure.sh > /root/autodl-tmp/dgcore_measure.log 2>&1 &
# 看进度: tail -f /root/autodl-tmp/dgcore_measure.log
# 结尾会直接打印 三方对比 + 各 resolver 校准;把结尾贴回给 agent 即可。
set -u
PY=/root/miniconda3/envs/rag/bin/python
export PATH=/root/miniconda3/envs/rag/bin:$PATH
cd /root/autodl-tmp/rag-L1 || exit 1

echo "########## [1/4] 重跑 query(改进版 dgcore,50篇)##########"
bash reproduce/graphrag_migration/run_dgcore_subset.sh

echo "########## [2/4] 清掉旧 dgcore 测评(避免缓存命中旧结果)##########"
"$PY" - <<'PYEOF'
import json
p = "/root/autodl-tmp/eval_4cond/llm_evaluation_results.json"
d = json.load(open(p, encoding="utf-8"))
is_dict = isinstance(d, dict)
recs = d.get("results", d) if is_dict else d
recs = [r for r in recs if not str(r.get("method", "")).startswith("dg")]
json.dump({**d, "results": recs} if is_dict else recs,
          open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print("cleared old dg* eval; remaining", len(recs))
PYEOF

echo "########## [3/4] eval(只新判 dgcore,其余缓存跳过)##########"
EVAL_MODEL=qwen-plus "$PY" reproduce/llm_answer_evaluator.py \
  --qa-data-dir /root/autodl-tmp/DocBench_subset --output-dir /root/autodl-tmp/eval_4cond

echo "########## [4/4] 三方对比 + L3 校准 ##########"
echo "===== 三方对比 ====="
"$PY" reproduce/graphrag_migration/compare_subset.py
echo "===== L3 校准 ====="
"$PY" reproduce/graphrag_migration/calibrate_dgcore.py
echo "MEASURE ALL DONE $(date)"
