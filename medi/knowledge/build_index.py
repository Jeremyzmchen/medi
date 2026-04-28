"""
build_index.py — 离线构建症状-科室向量知识库

数据来源：FreedomIntelligence/Huatuo26M-Lite
  - 17.7 万条中文医疗问答，16 个科室标签
  - 字段：question（症状描述）+ label（科室）+ score（质量评分）
向量模型：BAAI/bge-large-zh-v1.5
向量库：ChromaDB（本地持久化）

运行：
    python -m medi.knowledge.build_index
    python -m medi.knowledge.build_index --limit 2000  # 快速测试
    python -m medi.knowledge.build_index --min-score 4  # 只用高质量数据
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import chromadb
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

CHROMA_PATH = Path(__file__).parents[3] / "data" / "chroma"
COLLECTION_NAME = "symptom_kb"
EMBED_MODEL = "BAAI/bge-large-zh-v1.5"
BGE_PREFIX = "为这个句子生成表示以用于检索相关文章："

DATASET_NAME = "FreedomIntelligence/Huatuo26M-Lite"


def load_records(limit: int | None = None, min_score: int = 0) -> list[dict]:
    print(f"[1/3] 加载数据集 {DATASET_NAME} ...")
    ds = load_dataset(DATASET_NAME, split="train")

    records = []
    seen = set()

    for row in ds:
        question = (row.get("question") or "").strip()
        label = (row.get("label") or "").strip()
        score = row.get("score") or 0

        if not question or not label:
            continue
        if score < min_score:
            continue

        key = hashlib.md5(question.encode()).hexdigest()
        if key in seen:
            continue
        seen.add(key)

        records.append({"id": key, "text": question, "department": label})

        if limit and len(records) >= limit:
            break

    print(f"    有效记录：{len(records)} 条（min_score>={min_score}）")
    return records


def build_chromadb(records: list[dict], batch_size: int = 512) -> None:
    print(f"[2/3] 加载向量模型 {EMBED_MODEL} ...")
    model = SentenceTransformer(EMBED_MODEL)

    CHROMA_PATH.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))

    try:
        client.delete_collection(COLLECTION_NAME)
        print(f"    已删除旧 collection: {COLLECTION_NAME}")
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    print(f"[3/3] 生成 embedding 并写入 ChromaDB（batch={batch_size}）...")
    total = len(records)

    for start in range(0, total, batch_size):
        batch = records[start : start + batch_size]
        texts = [r["text"] for r in batch]
        ids = [r["id"] for r in batch]
        metadatas = [{"department": r["department"]} for r in batch]

        embeddings = model.encode(
            [BGE_PREFIX + t for t in texts],
            normalize_embeddings=True,
            show_progress_bar=False,
        ).tolist()

        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )

        done = min(start + batch_size, total)
        print(f"    进度：{done}/{total} ({done/total:.0%})")

    print(f"\n完成！共 {collection.count()} 条，索引路径：{CHROMA_PATH}")


def main() -> None:
    parser = argparse.ArgumentParser(description="构建症状-科室向量知识库")
    parser.add_argument("--limit", type=int, default=None, help="限制条数（默认全量）")
    parser.add_argument("--min-score", type=int, default=0, help="最低质量评分过滤（0-5）")
    parser.add_argument("--batch-size", type=int, default=512)
    args = parser.parse_args()

    records = load_records(limit=args.limit, min_score=args.min_score)
    build_chromadb(records, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
