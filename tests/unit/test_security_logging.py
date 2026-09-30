import logging

from app.core.logging import JsonFormatter, redact
from app.core.security import RateLimiter


def test_redaction_of_secrets():
    s = redact("key=gsk_abcdefghijklmnopqrstuvwxyz123456 url=postgresql+psycopg://u:hunter2@db/x "
               "Authorization: Bearer abc.def X-Admin-API-Key: supersecret")
    assert "gsk_abcdefghijkl" not in s and "hunter2" not in s and "abc.def" not in s and "supersecret" not in s


def test_json_formatter_redacts_extras():
    rec = logging.makeLogRecord({"msg": "calling", "levelname": "INFO", "name": "t",
                                 "api_key": "gsk_abcdefghijklmnopqrstuvwxyz123456"})
    assert "gsk_abcdefghijklmnop" not in JsonFormatter().format(rec)


async def test_local_rate_limiter():
    rl = RateLimiter()
    results = [(await rl.hit("ip", limit=3))[0] for _ in range(5)]
    assert results == [True, True, True, False, False]
    assert (await rl.hit("other-ip", limit=3))[0]
