"""Factory for creating reusable RAG components."""

import base64
from pathlib import Path
from langchain_milvus import Milvus, BM25BuiltInFunction
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.messages import HumanMessage
from langchain_core.language_models import BaseChatModel
from sentence_transformers import CrossEncoder


class EmbeddingService:
    """Singleton for embedding model."""
    
    _instances = {}
    
    @classmethod
    def get_embeddings(cls, model_name: str) -> HuggingFaceEmbeddings:
        """Get or create embedding model instance."""
        if model_name not in cls._instances:
            cls._instances[model_name] = HuggingFaceEmbeddings(model_name=model_name)
        return cls._instances[model_name]


class RerankerService:
    """Singleton for reranker model."""
    
    _instances = {}
    
    @classmethod
    def get_reranker(cls, model_name: str) -> CrossEncoder:
        """Get or create reranker model instance."""
        if model_name not in cls._instances:
            cls._instances[model_name] = CrossEncoder(model_name)
        return cls._instances[model_name]


class MilvusStoreFactory:
    """Factory for creating Milvus stores."""
    
    @staticmethod
    def create_store(
        embeddings: HuggingFaceEmbeddings,
        milvus_uri: str,
        collection_name: str,
        is_child: bool = True,
    ) -> Milvus:
        """Create Milvus store with hybrid search."""
        bm25 = BM25BuiltInFunction(input_field_names="text", output_field_names="sparse")
        suffix = "children" if is_child else "parents"
        
        return Milvus(
            embeddings,
            builtin_function=bm25,
            vector_field=["dense", "sparse"],
            collection_name=f"{collection_name}_{suffix}",
            connection_args={"uri": milvus_uri},
        )


class VisionService:
    """Singleton VLM service for analyzing figure images."""
    
    _instance = None
    
    def __init__(self, llm: BaseChatModel):
        self.llm = llm
    
    @classmethod
    def get_instance(cls, llm: BaseChatModel = None):
        """Get or create VisionService instance."""
        if cls._instance is None and llm:
            cls._instance = cls(llm)
        return cls._instance
    
    def analyze_figure(self, image_path: str, caption: str = "") -> str:
        """Analyze figure and return VLM description."""
        if not Path(image_path).exists():
            return ""
        
        try:
            with open(image_path, "rb") as f:
                image_data = base64.b64encode(f.read()).decode("utf-8")
        except Exception:
            return ""
        
        prompt = (
            "Analyze this figure from an academic paper. Describe:\n"
            "1. Chart/diagram type (bar chart, line plot, architecture diagram, etc.)\n"
            "2. Key visual elements (axes, labels, trends, components)\n"
            "3. Main findings or patterns shown\n"
            "4. Numerical values if visible\n\n"
            "Be concise and focus on information useful for answering research questions."
        )
        if caption:
            prompt += f"\n\nCaption context: {caption}"
        
        message = HumanMessage(
            content=[
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_data}"}},
            ]
        )
        
        try:
            response = self.llm.invoke([message])
            return response.content if hasattr(response, "content") else str(response)
        except Exception as e:
            print(f"VLM analysis failed: {e}")
            return ""


def is_visual_query(query: str) -> bool:
    """Check if query is inherently visual."""
    visual_keywords = [
        "show", "display", "visualize", "plot", "chart", "graph", "diagram",
        "figure", "illustration", "image", "picture", "看图", "图中", "图表",
        "what does", "what is shown", "describe the figure", "describe the image",
    ]
    query_lower = query.lower()
    return any(kw in query_lower for kw in visual_keywords)


def should_invoke_vlm(query: str, has_figure: bool, answer: str = "") -> bool:
    """Determine if VLM should be invoked."""
    if not has_figure:
        return False
    
    if is_visual_query(query):
        return True
    
    if answer:
        insufficient_indicators = [
            "not contain sufficient information", "insufficient",
            "cannot answer", "no relevant information",
            "信息不足", "无法回答",
        ]
        if any(ind in answer.lower() for ind in insufficient_indicators):
            return True
    
    return False
