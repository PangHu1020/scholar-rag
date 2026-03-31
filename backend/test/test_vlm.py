"""Test VLM integration: figure extraction, visual query detection, VLM analysis."""

import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from rag.integration import PDFParser
from rag.factory import is_visual_query, should_invoke_vlm, VisionService


PDF_PATH = "uploads/6f5cb1d8-31da-4a42-9367-906de466a0d7.pdf"


# --------------- Test 1: is_visual_query ---------------

def test_is_visual_query():
    print("=" * 60)
    print("Test 1: is_visual_query")
    print("=" * 60)

    cases = [
        ("What does Figure 3 show?",              True),
        ("Describe the architecture diagram",      True),
        ("Show me the performance chart",          True),
        ("What is the accuracy of the model?",     False),
        ("How does the method work?",              False),
        ("图中展示了什么结果",                      True),
        ("What are the experimental results?",     False),
    ]

    all_pass = True
    for query, expected in cases:
        result = is_visual_query(query)
        status = "✓" if result == expected else "✗"
        if result != expected:
            all_pass = False
        print(f"  {status} [{str(expected):5s}=={str(result):5s}] {query}")

    print(f"\n  {'PASS' if all_pass else 'FAIL'}\n")
    return all_pass


# --------------- Test 2: should_invoke_vlm ---------------

def test_should_invoke_vlm():
    print("=" * 60)
    print("Test 2: should_invoke_vlm")
    print("=" * 60)

    cases = [
        # (query, has_figure, answer, expected)
        ("What does Figure 3 show?", True,  "",                                        True),
        ("What does Figure 3 show?", False, "",                                        False),
        ("How does the method work?", True, "The model uses attention mechanism.",     False),
        ("How does the method work?", True, "cannot answer based on available info",   True),
        ("What is the accuracy?",    True,  "insufficient information to answer",      True),
        ("What is the accuracy?",    False, "insufficient information to answer",      False),
    ]

    all_pass = True
    for query, has_fig, answer, expected in cases:
        result = should_invoke_vlm(query, has_fig, answer)
        status = "✓" if result == expected else "✗"
        if result != expected:
            all_pass = False
        print(f"  {status} [{str(expected):5s}=={str(result):5s}] has_fig={has_fig} | {query[:45]}")

    print(f"\n  {'PASS' if all_pass else 'FAIL'}\n")
    return all_pass


# --------------- Test 3: Figure image extraction ---------------

def test_figure_extraction(pdf_path: str):
    print("=" * 60)
    print("Test 3: Figure Image Extraction")
    print("=" * 60)

    if not Path(pdf_path).exists():
        print(f"  ✗ PDF not found: {pdf_path}")
        return None

    parser = PDFParser()
    print(f"  Parsing: {pdf_path}")
    nodes = parser.parse(pdf_path, paper_id="vlm_test")

    figure_nodes = [n for n in nodes if n.node_type == "figure"]
    print(f"  Total nodes: {len(nodes)}")
    print(f"  Figure nodes: {len(figure_nodes)}")

    saved = 0
    for node in figure_nodes:
        if node.image_path and Path(node.image_path).exists():
            saved += 1

    print(f"  Images saved: {saved}/{len(figure_nodes)}")

    for i, node in enumerate(figure_nodes[:3], 1):
        exists = node.image_path and Path(node.image_path).exists()
        status = "✓" if exists else "✗"
        print(f"\n  {status} Figure {i}")
        print(f"    Page: {node.page_num} | BBox: {node.bbox}")
        print(f"    image_path: {node.image_path}")
        print(f"    text: {node.text[:80].replace(chr(10), ' ')}...")

    print(f"\n  {'PASS' if saved > 0 else 'WARN (no figures found)'}\n")
    return nodes


# --------------- Test 4: VLM analysis ---------------

def test_vlm_analysis(nodes):
    print("=" * 60)
    print("Test 4: VLM Analysis")
    print("=" * 60)

    from config import Config
    if not Config.VLM_ENABLED:
        print("  VLM_ENABLED=false, skipping.")
        print("  Set VLM_ENABLED=true in .env to test.\n")
        return

    figure_nodes = [
        n for n in (nodes or [])
        if n.node_type == "figure" and n.image_path and Path(n.image_path).exists()
    ]

    if not figure_nodes:
        print("  ✗ No figure images available\n")
        return

    from langchain_openai import ChatOpenAI
    vlm_llm = ChatOpenAI(
        base_url=Config.VLM_BASE_URL,
        model=Config.VLM_MODEL,
        api_key=Config.VLM_API_KEY,
        temperature=0,
    )
    vision_service = VisionService(vlm_llm)

    node = figure_nodes[0]
    print(f"  Analyzing: {node.image_path}")
    print(f"  Caption: {node.text[:100]}")

    description = vision_service.analyze_figure(node.image_path, caption=node.text)

    if description:
        print(f"\n  ✓ VLM description ({len(description)} chars):")
        print(f"  {description[:300]}...")
    else:
        print("  ✗ VLM returned empty description")

    print()


# --------------- Main ---------------

if __name__ == "__main__":
    pdf = sys.argv[1] if len(sys.argv) > 1 else PDF_PATH

    r1 = test_is_visual_query()
    r2 = test_should_invoke_vlm()
    nodes = test_figure_extraction(pdf)
    test_vlm_analysis(nodes)

    log_dir = Path(__file__).parent.parent / "log"
    log_dir.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"Done. Tests 1&2 logic: {'PASS' if r1 and r2 else 'FAIL'}")
