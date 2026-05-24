"""Unit tests for services/live/f1_signalr.py."""

from __future__ import annotations

from unittest.mock import MagicMock

from services.live.f1_signalr import _acquire_f1_cookies


class TestAcquireF1Cookies:
    def test_returns_cookie_from_post_response(self) -> None:
        session = MagicMock()
        resp = MagicMock()
        resp.cookies.get.return_value = "abc123"
        session.post.return_value = resp

        headers = _acquire_f1_cookies(session)

        assert headers["Cookie"] == "AWSALBCORS=abc123"
        assert headers["Origin"] == "https://www.formula1.com"
        session.post.assert_called_once()

    def test_falls_back_to_session_cookie(self) -> None:
        session = MagicMock()
        resp = MagicMock()
        resp.cookies.get.return_value = None
        session.post.return_value = resp
        session.options.return_value = resp
        session.cookies.get.return_value = "stored"

        headers = _acquire_f1_cookies(session)

        assert headers["Cookie"] == "AWSALBCORS=stored"

    def test_returns_browser_headers_without_cookie(self) -> None:
        session = MagicMock()
        resp = MagicMock()
        resp.cookies.get.return_value = None
        session.post.return_value = resp
        session.options.return_value = resp
        session.cookies.get.return_value = None

        headers = _acquire_f1_cookies(session)

        assert "Cookie" not in headers
        assert "User-Agent" in headers
