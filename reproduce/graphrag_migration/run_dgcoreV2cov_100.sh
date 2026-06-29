#!/bin/bash
# DG-RAG 100 篇子集 —— V2 + 覆盖(authors/date)。method=dgcoreV2cov,写 qa_results_dgcoreV2cov.json。
# 100 篇 = meta 题最多的 100 篇(有 index+pdf+content_list),共 188 道 meta。
# paperbase/ours 已在全 229 评过,eval 复用缓存只评新 method,省钱。
# 用法(服务器,后台): nohup bash reproduce/graphrag_migration/run_dgcoreV2cov_100.sh >/dev/null 2>&1 &
set -u
IDS="0 2 3 4 5 6 9 10 11 15 17 18 19 21 22 23 24 25 26 27 32 52 54 59 69 77 85 86 91 92 100 102 108 109 110 111 112 113 114 115 116 117 118 119 120 121 122 123 124 125 145 148 150 154 155 157 158 159 160 161 162 163 164 165 166 167 169 170 171 172 173 174 175 176 177 178 179 191 207 208 209 210 211 212 213 214 215 216 217 218 219 220 221 222 223 224 225 226 227 228"
export PATH=/root/miniconda3/envs/rag/bin:$PATH
export ENABLE_RERANK=true ENABLE_VLM=true EMBEDDING_MAX_CONCURRENCY=4
export ENABLE_OPERATOR_LAYER=false ENABLE_OP_ALGEBRA=false ENABLE_ABSTAIN=false ENABLE_NEURAL_ROUTING=false
export ENABLE_DOC_OUTLINE=false ENABLE_DOC_ANCHOR=false ENABLE_NCG=false ENABLE_EMR=false ENABLE_AUTO_VLM=false ENABLE_RETRIEVAL_REFLECT=false
export ENABLE_DOC_META=false ENABLE_DOC_LOCATE=false
export ENABLE_DG_CORE=true
export PARSE_OUTPUT_DIR=/root/autodl-tmp/content_lists
PY=/root/miniconda3/envs/rag/bin/python
DATA=/root/autodl-tmp/DocBench_subset
IDX=/root/autodl-tmp/RAG-Anything-thesis/rag_storage_baseline
EVAL_OUT=/root/autodl-tmp/eval_4cond
LOG=/root/autodl-tmp/dgcoreV2cov_100.log
cd /root/autodl-tmp/rag-L1 || exit 1
: > "$LOG"
echo "==== DG-RAG V2+cov 100篇跑题开始 $(date) ====" | tee -a "$LOG"
tot=$(echo $IDS | wc -w); n=0
for id in $IDS; do
  n=$((n + 1))
  pdf=$(ls "$DATA/$id"/*.pdf 2>/dev/null | head -1)
  if [ -z "$pdf" ]; then echo "[$n/$tot] doc $id NO PDF" | tee -a "$LOG"; continue; fi
  echo "[$n/$tot] doc $id $(basename "$pdf")" | tee -a "$LOG"
  RESULT_NAME=qa_results_dgcoreV2cov.json "$PY" reproduce/query.py "$pdf" --working_dir "$IDX/$id" >> "$LOG" 2>&1
  echo "    done doc $id (exit $?)" | tee -a "$LOG"
done
echo "==== QUERY 完成,开始 eval(只评新 method=dgcoreV2cov)$(date) ====" | tee -a "$LOG"
EVAL_MODEL=qwen-plus "$PY" reproduce/llm_answer_evaluator.py --qa-data-dir "$DATA" --output-dir "$EVAL_OUT" >> "$LOG" 2>&1
echo "==== ALL DONE $(date) ====" | tee -a "$LOG"
echo "对比: $PY reproduce/graphrag_migration/compare_100.py" | tee -a "$LOG"
