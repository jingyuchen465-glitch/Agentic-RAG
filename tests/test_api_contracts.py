from app.core.api import failed, ok


def test_success_envelope():
    """成功响应信封：success 为 True、data 正确、无 error、时间戳带时区。"""
    response = ok({"value": 1})
    assert response.success is True
    assert response.data == {"value": 1}
    assert response.error is None
    assert response.timestamp.tzinfo is not None


def test_error_envelope():
    """失败响应信封：success 为 False、无 data、错误码与详情正确。"""
    response = failed("BAD_INPUT", "invalid", ["query"])
    assert response.success is False
    assert response.data is None
    assert response.error.code == "BAD_INPUT"
    assert response.error.details == ["query"]
