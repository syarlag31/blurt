"""Tests for network egress guards and data leakage prevention."""

from __future__ import annotations

import socket
import time

import httpx
import pytest

from blurt.config.settings import BlurtConfig, DeploymentMode
from blurt.core.app import create_app
from blurt.middleware.egress_guard import (
    EgressBlockedError,
    EgressGuard,
    EgressViolation,
    GuardedSyncTransport,
    GuardedTransport,
    SocketEgressGuard,
    ViolationSeverity,
    _is_loopback,
)


# ---------------------------------------------------------------------------
# _is_loopback tests
# ---------------------------------------------------------------------------


class TestIsLoopback:
    def test_localhost(self) -> None:
        assert _is_loopback("localhost") is True

    def test_ipv4_loopback(self) -> None:
        assert _is_loopback("127.0.0.1") is True

    def test_ipv6_loopback(self) -> None:
        assert _is_loopback("::1") is True

    def test_ipv6_bracketed(self) -> None:
        assert _is_loopback("[::1]") is True

    def test_zero_address(self) -> None:
        assert _is_loopback("0.0.0.0") is True

    def test_external_ip(self) -> None:
        assert _is_loopback("8.8.8.8") is False

    def test_external_hostname(self) -> None:
        assert _is_loopback("api.google.com") is False

    def test_private_ip_not_loopback(self) -> None:
        assert _is_loopback("192.168.1.1") is False

    def test_ipv4_loopback_range(self) -> None:
        assert _is_loopback("127.0.0.2") is True


# ---------------------------------------------------------------------------
# GuardedTransport tests
# ---------------------------------------------------------------------------


class TestGuardedTransport:
    @pytest.mark.asyncio
    async def test_blocks_external_request(self) -> None:
        violations: list[EgressViolation] = []
        transport = GuardedTransport(on_violation=violations.append)

        request = httpx.Request("GET", "https://api.google.com/v1/data")

        with pytest.raises(EgressBlockedError) as exc_info:
            await transport.handle_async_request(request)

        assert exc_info.value.layer == "httpx"
        assert "api.google.com" in exc_info.value.destination
        assert len(violations) == 1
        assert violations[0].severity == ViolationSeverity.BLOCKED

    @pytest.mark.asyncio
    async def test_loopback_raises_passthrough(self) -> None:
        """Loopback requests raise a special error (not a real block)."""
        transport = GuardedTransport()
        request = httpx.Request("GET", "http://localhost:8000/health")

        with pytest.raises(EgressBlockedError) as exc_info:
            await transport.handle_async_request(request)
        assert "loopback-passthrough" in exc_info.value.layer

    @pytest.mark.asyncio
    async def test_violation_callback(self) -> None:
        violations: list[EgressViolation] = []
        transport = GuardedTransport(on_violation=violations.append)
        request = httpx.Request("POST", "https://external.api.com/data")

        with pytest.raises(EgressBlockedError):
            await transport.handle_async_request(request)

        assert len(violations) == 1
        v = violations[0]
        assert v.layer == "httpx"
        assert "external.api.com" in v.destination
        assert v.severity == ViolationSeverity.BLOCKED


class TestGuardedSyncTransport:
    def test_blocks_external_request(self) -> None:
        violations: list[EgressViolation] = []
        transport = GuardedSyncTransport(on_violation=violations.append)

        request = httpx.Request("GET", "https://api.notion.com/v1/pages")

        with pytest.raises(EgressBlockedError) as exc_info:
            transport.handle_request(request)

        assert exc_info.value.layer == "httpx-sync"
        assert len(violations) == 1


# ---------------------------------------------------------------------------
# SocketEgressGuard tests
# ---------------------------------------------------------------------------


class TestSocketEgressGuard:
    def test_activate_deactivate(self) -> None:
        guard = SocketEgressGuard()
        assert guard.active is False

        guard.activate()
        assert guard.active is True

        guard.deactivate()
        assert guard.active is False

    def test_blocks_external_connection(self) -> None:
        violations: list[EgressViolation] = []
        guard = SocketEgressGuard(on_violation=violations.append)

        guard.activate()
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            with pytest.raises(EgressBlockedError) as exc_info:
                sock.connect(("8.8.8.8", 443))
            assert exc_info.value.layer == "socket"
            assert len(violations) == 1
            sock.close()
        finally:
            guard.deactivate()

    def test_allows_loopback(self) -> None:
        guard = SocketEgressGuard()

        guard.activate()
        try:
            # Create a listening socket to connect to
            server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server.bind(("127.0.0.1", 0))
            port = server.getsockname()[1]
            server.listen(1)

            # This should NOT raise — loopback is allowed
            client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            client.connect(("127.0.0.1", port))
            client.close()
            server.close()
        finally:
            guard.deactivate()

    def test_context_manager(self) -> None:
        violations: list[EgressViolation] = []
        with SocketEgressGuard(on_violation=violations.append) as guard:
            assert guard.active is True
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            with pytest.raises(EgressBlockedError):
                sock.connect(("1.2.3.4", 80))
            sock.close()
        # Guard should be deactivated after context
        assert guard.active is False

    def test_idempotent_activate(self) -> None:
        guard = SocketEgressGuard()
        guard.activate()
        guard.activate()  # Should not raise or double-patch
        guard.deactivate()

    def test_restores_original_connect(self) -> None:
        original = socket.socket.connect
        guard = SocketEgressGuard()
        guard.activate()
        guard.deactivate()
        assert socket.socket.connect is original


# ---------------------------------------------------------------------------
# EgressGuard (unified) tests
# ---------------------------------------------------------------------------


class TestEgressGuard:
    def test_default_inactive(self) -> None:
        guard = EgressGuard()
        assert guard.enabled is False
        assert guard.violation_count == 0

    def test_activate_deactivate(self) -> None:
        guard = EgressGuard()
        guard.activate()
        assert guard.enabled is True
        guard.deactivate()
        assert guard.enabled is False

    def test_records_violations(self) -> None:
        guard = EgressGuard()
        guard._record_violation(
            EgressViolation(
                timestamp=time.time(),
                destination="api.google.com:443",
                layer="test",
                severity=ViolationSeverity.BLOCKED,
            )
        )
        assert guard.violation_count == 1
        assert guard.violations[0].destination == "api.google.com:443"

    def test_violation_cap(self) -> None:
        guard = EgressGuard()
        guard._max_violations = 10
        for i in range(15):
            guard._record_violation(
                EgressViolation(
                    timestamp=time.time(),
                    destination=f"host{i}.com:443",
                    layer="test",
                    severity=ViolationSeverity.BLOCKED,
                )
            )
        # Should have been trimmed
        assert guard.violation_count <= 10

    def test_check_destination_when_disabled(self) -> None:
        guard = EgressGuard()
        assert guard.check_destination("api.google.com") is True

    def test_check_destination_when_enabled(self) -> None:
        guard = EgressGuard()
        guard.enabled = True  # Enable without socket guard
        assert guard.check_destination("api.google.com") is False
        assert guard.check_destination("localhost") is True
        assert guard.check_destination("127.0.0.1") is True

    def test_create_guarded_transport(self) -> None:
        guard = EgressGuard()
        transport = guard.create_guarded_transport()
        assert isinstance(transport, GuardedTransport)

    def test_status(self) -> None:
        guard = EgressGuard()
        status = guard.status()
        assert status["enabled"] is False
        assert status["violation_count"] == 0
        assert "recent_violations" in status

    def test_clear_violations(self) -> None:
        guard = EgressGuard()
        guard._record_violation(
            EgressViolation(
                timestamp=time.time(),
                destination="test:443",
                layer="test",
                severity=ViolationSeverity.BLOCKED,
            )
        )
        assert guard.violation_count == 1
        guard.clear_violations()
        assert guard.violation_count == 0

    def test_activate_blocks_socket(self) -> None:
        guard = EgressGuard()
        guard.activate()
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            with pytest.raises(EgressBlockedError):
                sock.connect(("8.8.8.8", 443))
            sock.close()
            assert guard.violation_count == 1
        finally:
            guard.deactivate()

    def test_idempotent_activate(self) -> None:
        guard = EgressGuard()
        guard.activate()
        guard.activate()  # No double-activate
        assert guard.enabled is True
        guard.deactivate()
        guard.deactivate()  # No double-deactivate
        assert guard.enabled is False


# ---------------------------------------------------------------------------
# EgressViolation tests
# ---------------------------------------------------------------------------


class TestEgressViolation:
    def test_to_dict(self) -> None:
        v = EgressViolation(
            timestamp=1234567890.0,
            destination="api.example.com:443",
            layer="httpx",
            severity=ViolationSeverity.BLOCKED,
            caller="test_func",
        )
        d = v.to_dict()
        assert d["timestamp"] == 1234567890.0
        assert d["destination"] == "api.example.com:443"
        assert d["layer"] == "httpx"
        assert d["severity"] == "blocked"
        assert d["caller"] == "test_func"


# ---------------------------------------------------------------------------
# App integration tests
# ---------------------------------------------------------------------------


class TestAppIntegration:
    def test_install_egress_guards_local_mode(self) -> None:
        """EgressGuard is activated when app is in local mode."""
        config = BlurtConfig(mode=DeploymentMode.LOCAL)
        app = create_app(config)

        guard = app.state.egress_guard
        assert guard is not None
        assert guard.enabled is True
        # Clean up socket guard
        guard.deactivate()

    def test_install_egress_guards_cloud_mode(self) -> None:
        """EgressGuard exists but is NOT activated in cloud mode."""
        config = BlurtConfig(mode=DeploymentMode.CLOUD)
        app = create_app(config)

        guard = app.state.egress_guard
        assert guard is not None
        assert guard.enabled is False

    @pytest.mark.asyncio
    async def test_egress_status_endpoint(self) -> None:
        """Local mode app exposes /egress-status diagnostic endpoint."""
        config = BlurtConfig(mode=DeploymentMode.LOCAL)
        app = create_app(config)
        guard: EgressGuard = app.state.egress_guard

        try:
            from httpx import ASGITransport, AsyncClient

            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as client:
                resp = await client.get("/egress-status")
                assert resp.status_code == 200
                data = resp.json()
                assert data["enabled"] is True
                assert "violation_count" in data
        finally:
            guard.deactivate()


# ---------------------------------------------------------------------------
# EgressBlockedError tests
# ---------------------------------------------------------------------------


class TestEgressBlockedError:
    def test_error_message(self) -> None:
        err = EgressBlockedError("api.google.com:443", layer="socket")
        assert "api.google.com:443" in str(err)
        assert "local-only mode" in str(err)
        assert err.destination == "api.google.com:443"
        assert err.layer == "socket"

    def test_default_layer(self) -> None:
        err = EgressBlockedError("test:443")
        assert err.layer == "unknown"
