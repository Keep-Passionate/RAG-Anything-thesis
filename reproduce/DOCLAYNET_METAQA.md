# DocLayNet-MetaQA

DocLayNet-MetaQA is a DocBench-style meta-data QA set derived from the
published DocLayNet human layout annotations. DocLayNet itself is a document
layout analysis benchmark, not a QA benchmark. We use its annotations only as
gold labels.

Recommended paper wording:

> We derive page/document-level structural QA instances from the human layout
> annotations of DocLayNet to test deterministic counting beyond DocBench.

Do not describe it as an existing QA benchmark.

## QA Templates

Each selected document receives up to five `type="meta-data"` questions:

- `How many pages does this document have?`
- `How many tables are in this document?`
- `How many figures are in this document?`
- `How many equations are in this document?`
- `How many section headers are in this document?`

Gold labels come from DocLayNet page metadata and human layout annotations.

## Server Usage

```bash
source /etc/network_turbo || true
cd /root/autodl-tmp/rag-L1 || exit 1
git pull --ff-only origin feat/dg-core-coverage

/root/miniconda3/envs/rag/bin/pip show pillow >/dev/null 2>&1 || \
  /root/miniconda3/envs/rag/bin/pip install -q pillow
/root/miniconda3/envs/rag/bin/pip show remotezip >/dev/null 2>&1 || \
  /root/miniconda3/envs/rag/bin/pip install -q remotezip

nohup /root/miniconda3/envs/rag/bin/python reproduce/doclaynet_metaqa.py \
  --selective-download \
  --split val \
  --limit-docs 80 \
  --candidate-docs 140 \
  --remote-timeout 30 \
  --out /root/autodl-tmp/DocLayNet_MetaQA \
  > /root/autodl-tmp/doclaynet_metaqa_build.log 2>&1 &

tail -f /root/autodl-tmp/doclaynet_metaqa_build.log
```

The output follows the DocBench directory convention:

```text
DocLayNet_MetaQA/
  manifest.json
  dataset_card.json
  dl_val_0001_xxxxxxxx/
    dl_val_0001_xxxxxxxx.pdf
    dl_val_0001_xxxxxxxx_qa.jsonl
    meta.json
```

## Notes

- `--selective-download` avoids downloading the full 28GB DocLayNet archive.
  It reads only `COCO/<split>.json` and the selected PNG pages from the remote
  zip via HTTP range requests.
- `--candidate-docs` lets the builder try extra candidate documents and stop
  after `--limit-docs` successful outputs. This is useful because occasional
  remote zip range reads may fail with transient server errors.
- `--remote-timeout` keeps slow remote range reads from hanging indefinitely;
  timed-out reads are retried according to `--remote-retries`.
- `--include-zero` includes zero-count element questions. The default omits
  zero-count element questions but always keeps page count.
- `--require-any Table,Picture,Formula,Section-header` keeps documents with at
  least one useful structural element.
- Generated PDFs are made from DocLayNet page PNGs. The DocLayNet annotations
  are not written into the PDFs and should not be used as inference input.
