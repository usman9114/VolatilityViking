"""Alerts (Telegram), heartbeat (healthchecks.io) and secret redaction. Never logs secrets."""
from __future__ import annotations

import logging
import os
import re
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping

from vv.exec.common import SECRET_ENV_VARS, SecretRedactingFilter

SECRET_VARS = (*SECRET_ENV_VARS, "VV_TELEGRAM_TOKEN", "VV_TELEGRAM_CHAT", "VV_HEARTBEAT_URL")
_BOT_RE = re.compile(r"(bot)\d{5,}:[A-Za-z0-9_-]{10,}")
LOG = logging.getLogger("vv.runner")


def redact(text: str, env: Mapping[str, str] | None = None) -> str:
    env = os.environ if env is None else env
    for n in SECRET_VARS:
        v = env.get(n, "")
        if v and len(v) >= 4:
            text = text.replace(v, "***")
    return _BOT_RE.sub(r"\1***", text)


def install_log_redaction(env: Mapping[str, str] | None = None) -> None:
    env = os.environ if env is None else env
    f = SecretRedactingFilter([env.get(n, "") for n in SECRET_VARS])
    for lg in (LOG, logging.getLogger("vv.exec")):
        lg.filters = [x for x in lg.filters if not isinstance(x, SecretRedactingFilter)] + [f]
    # handlers see records from child loggers too, so attach there as well
    for h in logging.getLogger().handlers:
        h.addFilter(f)


def _http_get(url: str) -> None:
    urllib.request.urlopen(url, timeout=10).read()


class Alerter:
    """Collects alerts for the report; sends to Telegram when env is set, otherwise log only."""

    def __init__(self, env: Mapping[str, str] | None = None,
                 http_get: Callable[[str], None] = _http_get) -> None:
        self.env = os.environ if env is None else env
        self._get = http_get
        self.messages: list[str] = []

    def alert(self, msg: str) -> None:
        msg = redact(msg, self.env)
        self.messages.append(msg)
        LOG.warning("ALERT %s", msg)
        token, chat = self.env.get("VV_TELEGRAM_TOKEN"), self.env.get("VV_TELEGRAM_CHAT")
        if not (token and chat):
            return
        q = urllib.parse.urlencode({"chat_id": chat, "text": f"[VV] {msg}"})
        try:
            self._get(f"https://api.telegram.org/bot{token}/sendMessage?{q}")
        except Exception as e:  # exception text can embed the URL (token): log the type only
            LOG.error("telegram send failed (%s)", type(e).__name__)

    def heartbeat(self, ok: bool = True) -> None:
        url = self.env.get("VV_HEARTBEAT_URL")
        if not url:
            return
        try:
            self._get(url if ok else url.rstrip("/") + "/fail")
        except Exception as e:
            LOG.error("heartbeat failed (%s)", type(e).__name__)
