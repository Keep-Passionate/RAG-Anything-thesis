#!/bin/bash
# DG-RAG 全量(229)·真·全开(all-fire):所有算子都注入(DG_ABSTAIN=false,不读 calib)。
# 这是唯一一次花 LLM 的全量跑;之后门控/消融/各策略对比全部由 gate_ablation.py 零 LLM 后处理完成。
# method=dgAllfire,写 qa_results_dgAllfire.json。自动遍历 DocBench_subset 下所有有 PDF+索引的文档。
# 用法: nohup bash reproduce/graphrag_migration/run_dgcore_allfire_full.sh >/dev/null 2>&1 &
#       tail -f /root/autodl-tmp/dgcore_allfire_full.log
set -u
export PATH=/root/miniconda3/envs/rag/bin:$PATH
export ENABLE_RERANK=true ENABLE_VLM=true EMBEDDING_MAX_CONCURRENCY=4
export ENABLE_OPERATOR_LAYER=false ENABLE_OP_ALGEBRA=false ENABLE_ABSTAIN=false ENABLE_NEURAL_ROUTING=false
export ENABLE_DOC_OUTLINE=false ENABLE_DOC_ANCHOR=false ENABLE_NCG=false ENABLE_EMR=false ENABLE_AUTO_VLM=false ENABLE_RETRIEVAL_REFLECT=false
export ENABLE_DOC_META=false ENABLE_DOC_LOCATE=false
export ENABLE_DG_CORE=true
export DG_ABSTAIN=false                                    # 真·全开:绕过门控,所有算子注入(供校准观测全部 kind)
export PARSE_OUTPUT_DIR=/root/autodl-tmp/content_lists
unset DG_CALIB_FILE                                        # 全开不读 calib
PY=/root/miniconda3/envs/rag/bin/python
DATA=/root/autodl-tmp/DocBench_subset
IDX=/root/autodl-tmp/RAG-Anything-thesis/rag_storage_baseline
EVAL_OUT=/root/autodl-tmp/eval_4cond
LOG=/root/autodl-tmp/dgcore_allfire_full.log
cd /root/autodl-tmp/rag-L1 || exit 1
: > "$LOG"
# 遍历所有有 pdf 且有索引目录的文档(=可跑的全量)
IDS=$(for d in "$DATA"/*/; do id=$(basename "$d"); [ -d "$IDX/$id" ] && ls "$d"/*.pdf >/dev/null 2>&1 && echo "$id"; done)
tot=$(echo $IDS | wc -w); n=0
echo "==== DG-RAG all-fire 全量开始 $(date) | 文档数=$tot | DG_ABSTAIN=false ====" | tee -a "$LOG"
for id in $IDS; do
  n=$((n + 1))
  pdf=$(ls "$DATA/$id"/*.pdf 2>/dev/null | head -1)
  if [ -z "$pdf" ]; then echo "[$n/$tot] doc $id NO PDF" | tee -a "$LOG"; continue; fi
  echo "[$n/$tot] doc $id $(basename "$pdf")" | tee -a "$LOG"
  RESULT_NAME=qa_results_dgAllfire.json "$PY" reproduce/query.py "$pdf" --working_dir "$IDX/$id" >> "$LOG" 2>&1
  echo "    done doc $id (exit $?)" | tee -a "$LOG"
done
echo "==== QUERY 完成,eval(只评 dgAllfire,复用 paperbase/ours 缓存)$(date) ====" | tee -a "$LOG"
EVAL_MODEL=qwen-plus "$PY" reproduce/llm_answer_evaluator.py --qa-data-dir "$DATA" --output-dir "$EVAL_OUT" >> "$LOG" 2>&1
echo "==== ALL DONE $(date) ====" | tee -a "$LOG"
echo "下一步(零 LLM):DG_METHOD=dgAllfire DG_SPLIT=hash $PY reproduce/graphrag_migration/gate_ablation.py" | tee -a "$LOG"
