import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "reproduce"))

from mm_grounding import build_mm_doc_model, ground, parse


def _write_content_list(tmp_path):
    image = tmp_path / "fig2.png"
    image.write_bytes(b"not-a-real-image-but-path-exists")
    content = [
        {
            "type": "text",
            "text": "The experiment includes a bicycle scene and model scores.",
            "page_idx": 0,
        },
        {
            "type": "image",
            "img_path": str(image),
            "image_caption": ["Figure 2: A bicycle is shown next to a red sign."],
            "_neighbor_text": "The bicycle example illustrates color recognition.",
            "page_idx": 1,
        },
        {
            "type": "table",
            "table_caption": ["Table 1: Accuracy comparison."],
            "table_body": [["Method", "Accuracy"], ["Base", "80"], ["Ours", "92"]],
            "page_idx": 2,
        },
    ]
    cl = tmp_path / "paper_content_list.json"
    cl.write_text(json.dumps(content), encoding="utf-8")
    return cl, image


def test_parse_visual_attribute_question():
    q = parse("What is the color of the bicycle?")
    assert q is not None
    assert q.intent == "visual_evidence"


def test_explicit_figure_grounding_includes_image_path(tmp_path):
    cl, image = _write_content_list(tmp_path)
    model = build_mm_doc_model(str(tmp_path / "paper.pdf"), str(cl))

    fact = ground("What color is the sign in Figure 2?", str(tmp_path / "paper.pdf"), model=model)

    assert fact is not None
    assert fact.kind == "mm_visual"
    assert fact.requires_vlm is True
    assert str(image) in fact.note
    assert "Image Path:" in fact.note


def test_explicit_table_grounding_includes_table_body(tmp_path):
    cl, _ = _write_content_list(tmp_path)
    model = build_mm_doc_model(str(tmp_path / "paper.pdf"), str(cl))

    fact = ground("According to Table 1, what accuracy does Ours achieve?", str(tmp_path / "paper.pdf"), model=model)

    assert fact is not None
    assert fact.kind == "mm_table"
    assert fact.requires_vlm is False
    assert "Ours | 92" in fact.note


def test_element_page_location(tmp_path):
    cl, _ = _write_content_list(tmp_path)
    model = build_mm_doc_model(str(tmp_path / "paper.pdf"), str(cl))

    fact = ground("On which page is Figure 2 located?", str(tmp_path / "paper.pdf"), model=model)

    assert fact is not None
    assert fact.kind == "mm_locate"
    assert "physical page 2" in fact.note


def test_missing_content_list_abstains(tmp_path):
    fact = ground("What color is the bicycle?", str(tmp_path / "missing.pdf"))
    assert fact is None
