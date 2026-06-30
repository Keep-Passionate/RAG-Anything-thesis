#!/bin/bash
# DG-RAG 全量(229)·真·全开(all-fire) + 评测 + 零LLM门控消融,一条龙。
# 断点续跑:自动跳过已有 dgAllfire 结果的文档,只对缺的补跑 query(省钱);
# 评测器会自动补判 accuracy=null 的旧记录、跳过已完好的其余 method。
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
# 1) 找出"有 pdf+索引但缺 dgAllfire 结果"的文档(断点续跑,不重复已跑的)
need=""
for d in "$DATA"/*/; do
  id=$(basename "$d")
  [ -d "$IDX/$id" ] || continue
  ls "$d"/*.pdf >/dev/null 2>&1 || continue
  [ -s "$d/qa_results_dgAllfire.json" ] || need="$need $id"
done
nq=$(echo $need | wc -w)
echo "==== all-fire 续跑 $(date) | 缺结果需补跑 query 的文档($nq): $need ====" | tee -a "$LOG"
# 2) 只对缺的补跑 query
n=0
for id in $need; do
  n=$((n + 1))
  pdf=$(ls "$DATA/$id"/*.pdf 2>/dev/null | head -1); [ -z "$pdf" ] && continue
  echo "[query $n/$nq] doc $id $(basename "$pdf")" | tee -a "$LOG"
  RESULT_NAME=qa_results_dgAllfire.json "$PY" reproduce/query.py "$pdf" --working_dir "$IDX/$id" >> "$LOG" 2>&1
  echo "    done doc $id (exit $?)" | tee -a "$LOG"
done
# 3) 评测(自动补判 dgAllfire 的 null 分数 + 新跑文档;其余 method 跳过,不重复花钱)
echo "==== EVAL(只判 dgAllfire 未完成项)$(date) ====" | tee -a "$LOG"
EVAL_MODEL=qwen-plus "$PY" reproduce/llm_answer_evaluator.py --qa-data-dir "$DATA" --output-dir "$EVAL_OUT" >> "$LOG" 2>&1
echo "现有 dgAllfire 结果文档数: $(ls "$DATA"/*/qa_results_dgAllfire.json 2>/dev/null | wc -l)" | tee -a "$LOG"
# 4) 零LLM出主表+消融(A/B/C/all-fire, dev/test, McNemar)
echo "==== ABLATION(零LLM)$(date) ====" | tee -a "$LOG"
DG_METHOD=dgAllfire DG_SPLIT=hash "$PY" reproduce/graphrag_migration/gate_ablation.py 2>&1 | tee -a "$LOG"
echo "==== ALL DONE $(date) ====" | tee -a "$LOG"
