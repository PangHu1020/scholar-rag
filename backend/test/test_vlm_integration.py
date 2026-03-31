"""Test VLM integration end-to-end."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from langchain_openai import ChatOpenAI
from config import Config
from rag.integration import PDFParser
from rag.vision import VisionService

def test_figure_extraction():
    """Test 1: Figure extraction from PDF."""
    print("=" * 60)
    print("Test 1: Figure Extraction")
    print("=" * 60)
    
    parser = PDFParser()
    pdf_path = "./data/sample.pdf"  # Replace with actual PDF
    
    if not Path(pdf_path).exists():
        print(f"PDF not found: {pdf_path}")
        print("Please provide a test PDF with figures.")
        return
    
    nodes = parser.parse(pdf_path, paper_id="test_paper")
    
    figure_nodes = [n for n in nodes if n.node_type == "figure"]
    print(f"\nExtracted {len(figure_nodes)} figure nodes")
    
    for i, node in enumerate(figure_nodes[:3], 1):
        print(f"\n[Figure {i}]")
        print(f"  Page: {node.page_num}")
        print(f"  Image path: {node.image_path}")
        print(f"  Text preview: {node.text[:100]}...")
        
        if node.image_path and Path(node.image_path).exists():
            print(f"  ✓ Image file exists")
        else:
            print(f"  ✗ Image file missing")
    
    return nodes


def test_vlm_analysis():
    """Test 2: VLM analysis of extracted figures."""
    print("\n" + "=" * 60)
    print("Test 2: VLM Analysis")
    print("=" * 60)
    
    if not Config.VLM_ENABLED:
        print("VLM_ENABLED=false, skipping VLM test.")
        print("Set VLM_ENABLED=true in .env to test VLM.")
        return
    
    vlm_llm = ChatOpenAI(
        base_url=Config.VLM_BASE_URL,
        model=Config.VLM_MODEL,
        api_key=Config.VLM_API_KEY,
    )
    
    vision_service = VisionService(vlm_llm)
    
    # Find a test figure
    test_image = Path("./data/figures/test_paper/page1_order5.png")
    if not test_image.exists():
        print(f"Test image not found: {test_image}")
        print("Run test_figure_extraction first.")
        return
    
    print(f"\nAnalyzing: {test_image}")
    description = vision_service.analyze_figure(str(test_image), caption="Test figure")
    
    print(f"\nVLM Description:\n{description}")


if __name__ == "__main__":
    print("VLM Integration Test\n")
    
    nodes = test_figure_extraction()
    
    if nodes:
        test_vlm_analysis()
    
    print("\n" + "=" * 60)
    print("Tests complete")
    print("=" * 60)
