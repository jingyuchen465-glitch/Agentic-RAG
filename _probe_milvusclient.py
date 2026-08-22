"""实测 MilvusClient：flush 后 search，并正确解析 SearchResult 返回结构。"""
from __future__ import annotations
import uuid
from app.core.config import get_settings
from pymilvus import MilvusClient, DataType

s = get_settings()
TMP = f"_probe_{uuid.uuid4().hex[:8]}"
DIM = 8
print("temp:", TMP)

client = MilvusClient(uri=s.milvus_uri, token=s.milvus_token or None, db_name=s.milvus_database)

try:
    schema = client.create_schema(auto_id=False, enable_dynamic_field=False)
    schema.add_field(field_name="id", datatype=DataType.VARCHAR, is_primary=True, max_length=64)
    schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=DIM)
    schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)
    schema.add_field(field_name="document_id", datatype=DataType.VARCHAR, max_length=64)
    schema.add_field(field_name="chunk_id", datatype=DataType.VARCHAR, max_length=64)
    schema.add_field(field_name="document_name", datatype=DataType.VARCHAR, max_length=512)
    schema.add_field(field_name="content", datatype=DataType.VARCHAR, max_length=4096)
    schema.add_field(field_name="page", datatype=DataType.INT64)

    from pymilvus.milvus_client.index import IndexParams
    ip = IndexParams()
    ip.add_index(field_name="vector", index_name="vector_idx", index_type="AUTOINDEX", metric_type="COSINE", params={})
    ip.add_index(field_name="sparse_vector", index_type="SPARSE_INVERTED_INDEX", metric_type="IP", params={})
    client.create_collection(collection_name=TMP, schema=schema, index_params=ip)

    client.insert(collection_name=TMP, data=[
        {"id": "id1", "vector": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], "sparse_vector": {1: 0.5, 5: 0.8}, "document_id": "doc1", "chunk_id": "c1", "document_name": "t.txt", "content": "hello world", "page": 1},
        {"id": "id2", "vector": [0.8, 0.7, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1], "sparse_vector": {2: 0.9}, "document_id": "doc1", "chunk_id": "c2", "document_name": "t.txt", "content": "another chunk", "page": 2},
    ])
    client.flush(TMP)
    print("insert + flush OK")

    res = client.search(
        collection_name=TMP,
        data=[[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]],
        anns_field="vector",
        search_params={"metric_type": "COSINE", "params": {"nprobe": 16}},
        limit=2,
        output_fields=["document_id", "chunk_id", "content", "page", "document_name", "vector"],
    )
    print("search result type:", type(res).__name__)
    hits = res[0]
    print("res[0] type:", type(hits).__name__, "len:", len(hits))
    hit = hits[0]
    print("hit type:", type(hit).__name__)
    if hasattr(hit, "entity"):
        print("hit.entity:", hit.entity)
        print("hit.distance:", hit.distance)
        print("entity.get vector:", hit.entity.get("vector") if hasattr(hit.entity, "get") else "no get")
    elif isinstance(hit, dict):
        print("hit dict keys:", list(hit.keys()))
        print("hit dict:", hit)

    # 尝试 hit["entity"] 方式
    try:
        ent = hit["entity"]
        print("hit['entity']:", ent)
    except Exception as e:
        print("hit['entity'] failed:", type(e).__name__, str(e)[:200])

finally:
    client.drop_collection(TMP)
    print("dropped", TMP)