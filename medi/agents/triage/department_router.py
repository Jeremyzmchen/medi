"""
DepartmentRouter — 科室路由器

通过向量检索症状-科室知识库，输出候选科室列表（含置信度）。
知识库由 knowledge/build_index.py 离线构建，存储在 data/chroma/。

使用前需先运行：
    python -m medi.knowledge.build_index
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

CHROMA_PATH = Path(__file__).parents[4] / "data" / "chroma"
COLLECTION_NAME = "symptom_kb"
EMBED_MODEL = "BAAI/bge-large-zh-v1.5"
BGE_QUERY_PREFIX = "为这个句子生成表示以用于检索相关文章："


@dataclass
class DepartmentCandidate:
    department: str
    confidence: float    # 0.0 ~ 1.0
    reason: str          # 推荐理由（面向用户展示）


class DepartmentRouter:
    def __init__(
        self,
        collection_name: str = COLLECTION_NAME,
        chroma_path: Path | None = None,
    ) -> None:
        self._collection_name = collection_name
        self._chroma_path = chroma_path or CHROMA_PATH
        self._model: SentenceTransformer | None = None
        self._collection = None

    def _ensure_loaded(self) -> None:
        """懒加载：第一次调用时初始化模型和 ChromaDB（避免启动时间过长）"""
        if self._collection is not None:
            return

        self._model = SentenceTransformer(EMBED_MODEL)
        client = chromadb.PersistentClient(path=str(self._chroma_path))
        self._collection = client.get_collection(self._collection_name)

    async def route(self, query_text: str, top_k: int = 3) -> list[DepartmentCandidate]:
        """
        检索症状-科室知识库，返回 top_k 个候选科室。

        策略：
        1. 检索 top_k * 5 条原始结果（同一科室可能命中多次）
        2. 按科室聚合，用最高相似度作为该科室得分
        3. 返回 top_k 个科室，相似度转换为置信度
        """
        self._ensure_loaded()

        # bge 检索时加前缀
        embedding = self._model.encode(
            BGE_QUERY_PREFIX + query_text,
            normalize_embeddings=True,
        ).tolist()

        raw_top_k = top_k * 5
        results = self._collection.query(
            query_embeddings=[embedding],
            n_results=raw_top_k,
            include=["documents", "metadatas", "distances"],
        )

        # distances 是余弦距离（0=完全相同，2=完全相反），转为相似度
        distances = results["distances"][0]
        metadatas = results["metadatas"][0]
        documents = results["documents"][0]

        # 按科室聚合：取每个科室的最高相似度
        dept_best: dict[str, tuple[float, str]] = {}  # dept -> (similarity, example_doc)
        for dist, meta, doc in zip(distances, metadatas, documents):
            dept = _clean_department(meta.get("department", "其他"))
            similarity = max(0.0, 1.0 - dist / 2.0)  # 余弦距离 -> 相似度
            if dept not in dept_best or similarity > dept_best[dept][0]:
                dept_best[dept] = (similarity, doc)

        # 按相似度排序，取 top_k
        sorted_depts = sorted(dept_best.items(), key=lambda x: x[1][0], reverse=True)[:top_k]

        candidates = []
        for dept, (similarity, example_doc) in sorted_depts:
            candidates.append(DepartmentCandidate(
                department=dept,
                confidence=round(similarity, 2),
                reason=f"知识库匹配度 {similarity:.0%}，相关症状案例：{example_doc[:30]}…",
            ))

        return candidates


def _clean_department(name: str) -> str:
    """清洗科室名，去掉数据集中可能携带的序号和标点（如 '1. 神经科' → '神经科'）"""
    import re
    name = name.strip()
    # 去掉开头的数字序号，如 "1." "2、" "（1）"
    name = re.sub(r"^[\d（\(]+[\.\、\）\)]\s*", "", name)
    return name.strip()
