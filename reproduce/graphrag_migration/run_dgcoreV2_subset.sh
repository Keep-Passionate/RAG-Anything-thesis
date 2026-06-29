#!/bin/bash
# DG-RAG 子集验证 V2 —— 新 dg_core(参数化组合子 + 实例级置信门控)在 50 篇 meta 子集上跑 query+eval。
# 结果写入 qa_results_dgcoreV2.json(method=dgcoreV2),【不动】旧的 qa_results_dgcore.json(64% 缓存,留作非回归对比)。
# eval 不删缓存 -> 只评新 method(dgcoreV2),省钱;paperbase/ours/dgcore 复用已有判定。
# 用法(服务器,建议 nohup 后台): nohup bash reproduce/graphrag_migration/run_dgcoreV2_subset.sh >/dev/null 2>&1 &
set -u
IDS="0 102 119 2 3 11 15 32 52 54 59 69 77 85 86 91 92 100 108 109 110 111 112 113 114 115 116 117 118 120 121 122 123 124 125 145 148 150 154 155 157 158 159 160 162 163 164 165 166 167"
export PATH=/root/miniconda3/envs/rag/bin:$PATH
export ENABLE_RERANK=true ENABLE_VLM=true EMBEDDING_MAX_CONCURRENCY=4
export ENABLE_OPERATOR_LAYER=false ENABLE_OP_ALGEBRA=false ENABLE_ABSTAIN=false ENABLE_NEURAL_ROUTING=false
export ENABLE_DOC_OUTLINE=false ENABLE_DOC_ANCHOR=false ENABLE_NCG=false ENABLE_EMR=false ENABLE_AUTO_VLM=false ENABLE_RETRIEVAL_REFLECT=false
export ENABLE_DOC_META=false ENABLE_DOC_LOCATE=false
export ENABLE_DG_CORE=true
# 新 dg_core 默认行为(V2:实例级置信 + arbitration)。如需精确复现旧 64% 行为做对照:export DG_LEGACY=true
export PARSE_OUTPUT_DIR=/root/autodl-tmp/content_lists
PY=/root/miniconda3/envs/rag/bin/python
DATA=/root/autodl-tmp/DocBench_subset
IDX=/root/autodl-tmp/RAG-Anything-thesis/rag_storage_baseline
EVAL_OUT=/root/autodl-tmp/eval_4cond
LOG=/root/autodl-tmp/dgcoreV2_subset.log
cd /root/autodl-tmp/rag-L1 || exit 1
: > "$LOG"
echo "==== DG-RAG V2 子集跑题开始 $(date) ====" | tee -a "$LOG"
tot=$(echo $IDS | wc -w); n=0
for id in $IDS; do
  n=$((n + 1))
  pdf=$(ls "$DATA/$id"/*.pdf 2>/dev/null | head -1)
  if [ -z "$pdf" ]; then echo "[$n/$tot] doc $id NO PDF" | tee -a "$LOG"; continue; fi
  echo "[$n/$tot] doc $id $(basename "$pdf")" | tee -a "$LOG"
  RESULT_NAME=qa_results_dgcoreV2.json "$PY" reproduce/query.py "$pdf" --working_dir "$IDX/$id" >> "$LOG" 2>&1
  echo "    done doc $id (exit $?)" | tee -a "$LOG"
done
echo "==== QUERY 全部完成,开始 eval(只评新 method=dgcoreV2,复用缓存)$(date) ====" | tee -a "$LOG"
EVAL_MODEL=qwen-plus "$PY" reproduce/llm_answer_evaluator.py --qa-data-dir "$DATA" --output-dir "$EVAL_OUT" >> "$LOG" 2>&1
echo "==== ALL DONE $(date) ====" | tee -a "$LOG"
echo "现在可运行对比: $PY reproduce/graphrag_migration/compare_subset_v2.py" | tee -a "$LOG"
