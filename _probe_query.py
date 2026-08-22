"""验证 query limit/offset、delete filter、get_collection_stats 字段。"""
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
    schema.add_field(field_name="path", datatype=DataType.VARCHAR, max_length=2048)
    schema.add_field(field_name="status", datatype=DataType.VARCHAR, max_length=32)
    schema.add_field(field_name="created_at", datatype=DataType.VARCHAR, max_length=64)
    from pymilvus.milvus_client.index import IndexParams
    ip = IndexParams()
    ip.add_index(field_name="vector", index_type="FLAT", metric_type="IP", params={})
    client.create_collection(TMP, schema=schema, index_params=ip)

    rows = []
    for i in range(5):
        rows.append({"id": f"d{i}", "vector": [float(i), 0.0], "filename": f"f{i}.txt", "path": f"/p/{i}", "status": "queued", "created_at": f"2026-08-2{i}"})
    client.insert(TMP, data=rows)
    client.flush(TMP)
    print("inserted 5 rows")

    # query with limit/offset kwargs?
    try:
        r = client.query(TMP, filter='', output_fields=["id", "filename"], limit=2, offset=1)
        print("query limit/offset OK ->", r)
    except Exception as e:
        print("query limit/offset FAILED:", type(e).__name__, str(e)[:200])

    # empty filter behavior
    try:
        r = client.query(TMP, filter='', output_fields=["id"], limit=1)
        print("empty filter query OK, first:", r)
    except Exception as e:
        print("empty filter FAILED:", type(e).__name__, str(e)[:200])

    # delete by filter
    try:
        res = client.delete(TMP, filter='id in ["d0", "d1"]')
        print("delete filter OK:", res)
    except Exception as e:
        print("delete filter FAILED:", type(e).__name__, str(e)[:200])
    client.flush(TMP)

    # stats
    st = client.get_collection_stats(TMP)
    print("stats dict:", st)
    print("row_count:", st.get("row_count"))

    # upsert single row (整行替换语义验证)
    client.upsert(TMP, data=[{"id": "d2", "vector": [9.0, 9.0], "filename": "OVERWRITTEN", "path": "/x", "status": "running", "created_at": "now"}])
    client.flush(TMP)
    r = client.query(TMP, filter='id == "d2"', output_fields=["id", "filename", "path", "status", "created_at"])
    print("after upsert d2:", r)

    # list collections
    print("list_collections has TMP:", TMP in client.list_collections())

finally:
    client.drop_collection(TMP)
    print("dropped", TMP)