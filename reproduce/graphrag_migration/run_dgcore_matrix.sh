#!/bin/bash
# DG-RAG 分层消融矩阵(07 号 §5):6 条件 × 50 篇 meta 子集,定位每层净贡献。
# 条件 A(paperbase)、ours 复用已有结果;本脚本只跑 dgcore 系列(ENABLE_DG_CORE=true)。
# 用法: nohup bash reproduce/graphrag_migration/run_dgcore_matrix.sh > /root/autodl-tmp/dgcore_matrix.nohup 2>&1 &
set -u
IDS="0 102 119 2 3 11 15 32 52 54 59 69 77 85 86 91 92 100 108 109 110 111 112 113 114 115 116 117 118 120 121 122 123 124 125 145 148 150 154 155 157 158 159 160 162 163 164 165 166 167"
export PATH=/root/miniconda3/envs/rag/bin:$PATH
export ENABLE_RERANK=true ENABLE_VLM=true EMBEDDING_MAX_CONCURRENCY=4
export ENABLE_OPERATOR_LAYER=false ENABLE_OP_ALGEBRA=false ENABLE_ABSTAIN=false ENABLE_NEURAL_ROUTING=false
export ENABLE_DOC_OUTLINE=false ENABLE_DOC_ANCHOR=false ENABLE_NCG=false ENABLE_EMR=false ENABLE_AUTO_VLM=false ENABLE_RETRIEVAL_REFLECT=false
export ENABLE_DOC_META=false ENABLE_DOC_LOCATE=false ENABLE_DG_CORE=true
export PARSE_OUTPUT_DIR=/root/autodl-tmp/content_lists
PY=/root/miniconda3/envs/rag/bin/python
DATA=/root/autodl-tmp/DocBench_subset
IDX=/root/autodl-tmp/RAG-Anything-thesis/rag_storage_baseline
LOG=/root/autodl-tmp/dgcore_matrix.log
cd /root/autodl-tmp/rag-L1 || exit 1
: > "$LOG"

run_cond () {
  local name="$1" result="$2"; shift 2
  echo "##### CONDITION $name | result=$result | env=$* | $(date) #####" | tee -a "$LOG"
  local n=0
  for id in $IDS; do
    n=$((n + 1))
    pdf=$(ls "$DATA/$id"/*.pdf 2>/dev/null | head -1)
    [ -z "$pdf" ] && { echo "  [$name $n/50] doc $id NO PDF" | tee -a "$LOG"; continue; }
    env "$@" RESULT_NAME="$result" "$PY" reproduce/query.py "$pdf" --working_dir "$IDX/$id" >> "$LOG" 2>&1
    echo "  [$name $n/50] doc $id done (exit $?)" | tee -a "$LOG"
  done
}

run_cond default     qa_results_dgcore.json
run_cond noPageMap   qa_results_dg_nopagemap.json   DG_PAGEMAP=false
run_cond noAbstain   qa_results_dg_noabstain.json   DG_ABSTAIN=false
run_cond mentTables  qa_results_dg_menttab.json     DG_MENTION_TABLES=true
run_cond reframe     qa_results_dg_reframe.json     DG_PAGEMAP_REFRAME=true
run_cond noMetaStats qa_results_dg_nometa.json      DG_META_STATS=false
# 逐算子留一法(IP&M tab:ablation)：default 减去每个 offX = 该算子净贡献。
run_cond offCount    qa_results_dg_offcount.json    DG_OFF_COUNT=true
run_cond offLocate   qa_results_dg_offlocate.json   DG_OFF_LOCATE=true
run_cond offExtract  qa_results_dg_offextract.json  DG_OFF_EXTRACT=true
run_cond offLookup   qa_results_dg_offlookup.json   DG_OFF_LOOKUP=true
echo "MATRIX ALL DONE $(date)" | tee -a "$LOG"
