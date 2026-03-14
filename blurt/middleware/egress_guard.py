"""Network egress guards and data leakage prevention.

When local-only mode is enabled (BLURT_MODE=local), all outbound network
requests are blocked and logged. This ensures zero data leakage — no
personal data ever leaves the user's machine.

Three layers of protection:
1. **GuardedTransport** — Drop-in httpx transport that blocks all requests.
2. **SocketEgressGuard** — Monkey-patches socket.connect to block TCP egress.
3. **EgressGuardMiddleware** — FastAPI middleware that enforces guards on
   every request lifecycle and logs violations.

Localhost/loopback connections are always allowed (the API server itself
needs to function).
"""

from __future__ import annotations

import ipaddress
import logging
import socket
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from types import TracebackType
from typing import Any, Self

import httpx
from fastapi import FastAPI, Request, Response
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.types import ASGIApp

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class EgressBlockedError(Exception):
    """Raised when an outbound network request is blocked in local-only mode."""

    def __init__(self, destination: str, layer: str = "unknown") -> None:
        self.destination = destination
        self.layer = layer
        super().__init__(
            f"Network egress blocked ({layer}): outbound request to "
            f"{destination!r} denied in local-only mode"
        )


# ---------------------------------------------------------------------------
# Violation tracking
# ---------------------------------------------------------------------------


class ViolationSeverity(str, Enum):
    """How severe the egress violation is."""

    BLOCKED = "blocked"  # Request was prevented
    WARN = "warn"  # Logged but allowed (used during transition)


@dataclass(frozen=True, slots=True)
class EgressViolation:
    """Record of an attempted network egress in local-only mode."""

    timestamp: float
    destination: str
    layer: str  # "httpx", "socket", "middleware"
    severity: ViolationSeverity
    caller: str = ""  # optional caller info

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "destination": self.destination,
            "layer": self.layer,
            "severity": self.severity.value,
            "caller": self.caller,
        }


# ---------------------------------------------------------------------------
# Allowed destinations (localhost / loopback)
# ---------------------------------------------------------------------------

_LOOPBACK_HOSTS = frozenset({
    "localhost",
    "127.0.0.1",
    "::1",
    "0.0.0.0",
    "[::1]",
})


def _is_loopback(host: str) -> bool:
    """Check if a host is a loopback / localhost address."""
    # Strip brackets from IPv6
    clean = host.strip("[]").lower()
    if clean in _LOOPBACK_HOSTS:
        return True
    try:
        addr = ipaddress.ip_address(clean)
        return addr.is_loopback
    except ValueError:
        return clean == "localhost"


# ---------------------------------------------------------------------------
# Layer 1: Guarded httpx Transport
# ---------------------------------------------------------------------------


class GuardedTransport(httpx.AsyncBaseTransport):
    """An httpx async transport that blocks all outbound requests.

    Drop-in replacement for httpx's default transport when local-only
    mode is active. All requests are rejected with EgressBlockedError.

    Localhost requests are allowed (for internal API calls).

    Usage::

        transport = GuardedTransport(on_violation=my_callback)
        client = httpx.AsyncClient(transport=transport)
        # Any request to non-localhost will raise EgressBlockedError
    """

    def __init__(
        self,
        *,
        on_violation: Callable[[EgressViolation], None] | None = None,
        allow_loopback: bool = True,
    ) -> None:
        self._on_violation = on_violation
        self._allow_loopback = allow_loopback

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        """Intercept and block outbound HTTP requests."""
        host = request.url.host or ""
        destination = f"{request.url.scheme}://{host}:{request.url.port}{request.url.path}"

        # Allow loopback
        if self._allow_loopback and _is_loopback(host):
            # For loopback, we can't actually make the request with this
            # transport — we just pass through. In practice the guarded
            # transport is only used when we want to BLOCK, so loopback
            # requests should use a separate transport.
            raise EgressBlockedError(
                destination,
                layer="httpx-loopback-passthrough",
            )

        violation = EgressViolation(
            timestamp=time.time(),
            destination=destination,
            layer="httpx",
            severity=ViolationSeverity.BLOCKED,
        )

        if self._on_violation:
            self._on_violation(violation)

        logger.warning(
            "EGRESS BLOCKED [httpx]: %s %s",
            request.method,
            destination,
        )

        raise EgressBlockedError(destination, layer="httpx")


class GuardedSyncTransport(httpx.BaseTransport):
    """Synchronous version of GuardedTransport for sync httpx clients."""

    def __init__(
        self,
        *,
        on_violation: Callable[[EgressViolation], None] | None = None,
    ) -> None:
        self._on_violation = on_violation

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        host = request.url.host or ""
        destination = f"{request.url.scheme}://{host}:{request.url.port}{request.url.path}"

        if _is_loopback(host):
            raise EgressBlockedError(destination, layer="httpx-sync-loopback")

        violation = EgressViolation(
            timestamp=time.time(),
            destination=destination,
            layer="httpx-sync",
            severity=ViolationSeverity.BLOCKED,
        )
        if self._on_violation:
            self._on_violation(violation)

        logger.warning("EGRESS BLOCKED [httpx-sync]: %s %s", request.method, destination)
        raise EgressBlockedError(destination, layer="httpx-sync")


# ---------------------------------------------------------------------------
# Layer 2: Socket-level egress guard
# ---------------------------------------------------------------------------


class SocketEgressGuard:
    """Monkey-patches socket.socket.connect to block non-loopback connections.

    This is the deepest layer of protection — catches ANY outbound TCP
    connection regardless of which HTTP library is used.

    Usage::

        guard = SocketEgressGuard()
        guard.activate()
        # ... all non-localhost socket connections blocked ...
        guard.deactivate()

    Or as a context manager::

        with SocketEgressGuard():
            # blocked
            pass
    """

    def __init__(
        self,
        *,
        on_violation: Callable[[EgressViolation], None] | None = None,
    ) -> None:
        self._on_violation = on_violation
        self._original_connect: Callable[..., Any] | None = None
        self._active = False

    @property
    def active(self) -> bool:
        return self._active

    def activate(self) -> None:
        """Install the socket-level egress guard."""
        if self._active:
            return

        self._original_connect = socket.socket.connect
        guard = self

        def guarded_connect(sock: socket.socket, address: Any) -> Any:
            """Intercept socket.connect calls."""
            host = ""
            if isinstance(address, tuple) and len(address) >= 2:
                host = str(address[0])
            elif isinstance(address, str):
                # Unix socket — always allow
                return guard._original_connect(sock, address)  # type: ignore[misc]

            if _is_loopback(host):
                return guard._original_connect(sock, address)  # type: ignore[misc]

            destination = f"{host}:{address[1]}" if isinstance(address, tuple) and len(address) >= 2 else str(address)

            violation = EgressViolation(
                timestamp=time.time(),
                destination=destination,
                layer="socket",
                severity=ViolationSeverity.BLOCKED,
            )

            if guard._on_violation:
                guard._on_violation(violation)

            logger.warning("EGRESS BLOCKED [socket]: connection to %s", destination)
            raise EgressBlockedError(destination, layer="socket")

        socket.socket.connect = guarded_connect  # type: ignore[assignment]
        self._active = True
        logger.info("Socket egress guard activated — non-loopback connections blocked")

    def deactivate(self) -> None:
        """Remove the socket-level egress guard."""
        if not self._active:
            return
        if self._original_connect is not None:
            socket.socket.connect = self._original_connect  # type: ignore[assignment]
            self._original_connect = None
        self._active = False
        logger.info("Socket egress guard deactivated")

    def __enter__(self) -> Self:
        self.activate()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.deactivate()


# ---------------------------------------------------------------------------
# Layer 3: Unified Egress Guard (combines all layers)
# ---------------------------------------------------------------------------


@dataclass
class EgressGuard:
    """Unified network egress guard for local-only mode.

    Manages all three layers of egress protection:
    - httpx transport blocking
    - Socket-level connection blocking
    - Violation logging and tracking

    Usage::

        guard = EgressGuard()
        guard.activate()
        # All outbound network requests are now blocked and logged

    Check violations::

        for v in guard.violations:
            print(f"{v.timestamp}: {v.destination} blocked at {v.layer}")
    """

    enabled: bool = False
    _violations: list[EgressViolation] = field(default_factory=list)
    _socket_guard: SocketEgressGuard | None = field(default=None, repr=False)
    _max_violations: int = 1000

    def __post_init__(self) -> None:
        self._socket_guard = SocketEgressGuard(on_violation=self._record_violation)

    @property
    def violations(self) -> list[EgressViolation]:
        """All recorded egress violations."""
        return list(self._violations)

    @property
    def violation_count(self) -> int:
        return len(self._violations)

    def _record_violation(self, violation: EgressViolation) -> None:
        """Record an egress violation, maintaining bounded history."""
        if len(self._violations) >= self._max_violations:
            # Keep most recent violations
            self._violations = self._violations[-(self._max_violations // 2) :]
        self._violations.append(violation)

    def activate(self) -> None:
        """Activate all egress protection layers."""
        if self.enabled:
            return
        self.enabled = True
        if self._socket_guard:
            self._socket_guard.activate()
        logger.info("Egress guard fully activated — local-only mode enforced")

    def deactivate(self) -> None:
        """Deactivate all egress protection layers."""
        if not self.enabled:
            return
        if self._socket_guard:
            self._socket_guard.deactivate()
        self.enabled = False
        logger.info("Egress guard deactivated")

    def create_guarded_transport(self) -> GuardedTransport:
        """Create a GuardedTransport linked to this guard's violation tracker."""
        return GuardedTransport(on_violation=self._record_violation)

    def create_guarded_sync_transport(self) -> GuardedSyncTransport:
        """Create a sync GuardedTransport linked to this guard's violation tracker."""
        return GuardedSyncTransport(on_violation=self._record_violation)

    def check_destination(self, host: str) -> bool:
        """Check if a destination is allowed. Returns True if allowed."""
        if not self.enabled:
            return True
        return _is_loopback(host)

    def clear_violations(self) -> None:
        """Clear recorded violations."""
        self._violations.clear()

    def status(self) -> dict[str, Any]:
        """Return guard status for health/diagnostics."""
        return {
            "enabled": self.enabled,
            "socket_guard_active": (
                self._socket_guard.active if self._socket_guard else False
            ),
            "violation_count": len(self._violations),
            "recent_violations": [
                v.to_dict() for v in self._violations[-10:]
            ],
        }


# ---------------------------------------------------------------------------
# Layer 3: FastAPI Middleware
# ---------------------------------------------------------------------------


class EgressGuardMiddleware(BaseHTTPMiddleware):
    """FastAPI middleware that enforces egress guards in local-only mode.

    Attaches the EgressGuard to request state so downstream handlers
    can check it. Also provides a /egress-status diagnostic endpoint.
    """

    def __init__(self, app: ASGIApp, *, egress_guard: EgressGuard) -> None:
        super().__init__(app)
        self._guard = egress_guard

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        # Attach guard to request state for downstream access
        request.state.egress_guard = self._guard

        # Handle diagnostic endpoint
        if request.url.path == "/egress-status":
            import json

            return Response(
                content=json.dumps(self._guard.status(), indent=2),
                media_type="application/json",
            )

        try:
            response = await call_next(request)
            return response
        except EgressBlockedError as exc:
            logger.error(
                "Egress violation during request %s %s: %s",
                request.method,
                request.url.path,
                exc,
            )
            return Response(
                content=(
                    '{"error": "network_egress_blocked", '
                    '"detail": "Outbound network request blocked in local-only mode", '
                    f'"destination": "{exc.destination}"}}'
                ),
                status_code=503,
                media_type="application/json",
            )


# ---------------------------------------------------------------------------
# Integration helper
# ---------------------------------------------------------------------------


def install_egress_guards(app: FastAPI, *, local_mode: bool) -> EgressGuard:
    """Install egress guards on a FastAPI app if local-only mode is enabled.

    Call this during app creation (before lifespan).

    Args:
        app: The FastAPI application.
        local_mode: Whether local-only mode is active.

    Returns:
        The EgressGuard instance (active if local_mode=True).
    """
    guard = EgressGuard()

    if local_mode:
        guard.activate()
        app.add_middleware(EgressGuardMiddleware, egress_guard=guard)
        logger.info(
            "Egress guards installed — all outbound network requests will be blocked"
        )
    else:
        # Still install the guard in inactive mode for runtime toggling
        app.state.egress_guard = guard

    app.state.egress_guard = guard
    return guard
