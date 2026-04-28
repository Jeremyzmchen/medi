"""
DepartmentRouter — 科室路由器

通过向量检索症状-科室知识库，输出候选科室列表（含置信度）。
Phase 1 使用 ChromaDB，知识库由 knowledge/build_index.py 离线构建。
"""

from __future__ import annotations

from dataclasses import dataclass

# Phase 1: ChromaDB 占位，build_index.py 构建后接入
# import chromadb


@dataclass
class DepartmentCandidate:
    department: str
    confidence: float    # 0.0 ~ 1.0
    reason: str          # 推荐理由（面向用户展示）


class DepartmentRouter:
    def __init__(self, collection_name: str = "symptom_kb") -> None:
        self._collection_name = collection_name
        # TODO: Phase 1 实现时初始化 ChromaDB client
        # self._client = chromadb.PersistentClient(path=chroma_path)
        # self._collection = self._client.get_collection(collection_name)

    async def route(self, query_text: str, top_k: int = 3) -> list[DepartmentCandidate]:
        """
        检索症状-科室知识库，返回 top_k 个候选科室。
        置信度由向量相似度转换而来。
        """
        # TODO: 接入 ChromaDB 后实现
        # results = self._collection.query(query_texts=[query_text], n_results=top_k)
        # return self._parse_results(results)

        # Phase 1 mock，用于测试主流程
        return [
            DepartmentCandidate(
                department="神经内科",
                confidence=0.85,
                reason="症状与偏头痛、神经性头痛高度匹配",
            ),
            DepartmentCandidate(
                department="耳鼻喉科",
                confidence=0.40,
                reason="头痛可能与鼻窦炎相关",
            ),
        ]
