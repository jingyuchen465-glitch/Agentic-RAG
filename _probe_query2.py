"""确认 MilvusClient.query 返回结构的真实类型与元素。"""
from __future__ import annotations
import uuid
from app.core.config import get_settings
from pymilvus import MilvusClient, DataType

s = get_settings()
TMP = f"_probe_{uuid.uuid4().hex[:8]}"
client = MilvusClient(uri=s.milvus_uri, token=s.milvus_token or None, db_name=s.milvus_database)

try:
    schema = client.create_schema(auto_id=False, enable_dynamic_field=False)
    schema.add_field(field_name="id", datatype=DataType.VARCHAR, is_primary=True, max_length=64)
    schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=2)
    schema.add_field(field_name="filename", datatype=DataType.VARCHAR, max_length=512)
    schema.add_field(field_name="status", datatype=DataType.VARCHAR, max_length=32)
    from pymilvus.milvus_client.index import IndexParams
    ip = IndexParams()
    ip.add_index(field_name="vector", index_type="FLAT", metric_type="IP", params={})
    client.create_collection(TMP, schema=schema, index_params=ip)
    client.insert(TMP, data=[{"id": "a", "vector": [1.0,0.0], "filename": "a.txt", "status": "queued"},
                             {"id": "b", "vector": [2.0,0.0], "filename": "b.txt", "status": "queued"}])
    client.flush(TMP)

    r = client.query(TMP, filter="", output_fields=["id", "filename", "status"], limit=1)
    print("type:", type(r).__name__)
    print("has .data:", hasattr(r, "data"))
    print("is list:", isinstance(r, list))
    print("iterable:", hasattr(r, "__iter__"))
    # 尝试迭代
    try:
        for x in r:
            print("  iter item type:", type(x).__name__, "| val:", repr(x))
            break
    except Exception as e:
        print("iter failed:", e)
    if hasattr(r, "data"):
        d = r.data
        print("data type:", type(d).__name__, "| data[0] type:", type(d[0]).__name__ if d else None)
        print("data[0] repr:", repr(d[0]) if d else None)
        # 如果是字符串，尝试 eval/ast
        if d and isinstance(d[0], str):
            import ast
            print("data[0] ast parse:", ast.literal_eval(d[0]))
    # 是否可以直接下标
    try:
        print("r[0]:", repr(r[0]))
    except Exception as e:
        print("r[0] failed:", type(e).__name__, str(e)[:200])

finally:
    client.drop_collection(TMP)
    print("dropped", TMP)