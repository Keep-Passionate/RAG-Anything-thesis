#!/bin/bash
# DG-RAG 子集验证 —— 条件 B(公共基座 + ENABLE_DG_CORE)。
# 条件 A(paperbase=公共基座, dg_core 关)已在全量跑过,复用各 doc 的 qa_results_paperbase.json,不重跑。
# 用法(服务器): bash reproduce/graphrag_migration/run_dgcore_subset.sh
# 关键环境(踩过的坑):
#   - PATH 必须含 rag env bin,否则 raganything 的 `mineru --version` 安装自检失败。
#   - PARSE_OUTPUT_DIR 指向汇总好的 content_lists(dg_core 的 element/title/locate 增强要用)。
set -u
IDS="0 102 119 2 3 11 15 32 52 54 59 69 77 85 86 91 92 100 108 109 110 111 112 113 114 115 116 117 118 120 121 122 123 124 125 145 148 150 154 155 157 158 159 160 162 163 164 165 166 167"
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
LOG=/root/autodl-tmp/dgcore_subset.log
cd /root/autodl-tmp/rag-L1 || exit 1
: > "$LOG"
tot=$(echo $IDS | wc -w); n=0
for id in $IDS; do
  n=$((n + 1))
  pdf=$(ls "$DATA/$id"/*.pdf 2>/dev/null | head -1)
  if [ -z "$pdf" ]; then echo "[$n/$tot] doc $id NO PDF" | tee -a "$LOG"; continue; fi
  echo "[$n/$tot] doc $id $(basename "$pdf")" | tee -a "$LOG"
  RESULT_NAME=qa_results_dgcore.json "$PY" reproduce/query.py "$pdf" --working_dir "$IDX/$id" >> "$LOG" 2>&1
  echo "    done doc $id (exit $?)" | tee -a "$LOG"
done
echo "ALL DONE $(date)" | tee -a "$LOG"
