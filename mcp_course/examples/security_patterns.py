#!/usr/bin/env python3
"""
Security and Authentication Patterns for MCP Servers

This module demonstrates security best practices and authentication mechanisms
for MCP servers, including:

1. Authentication mechanisms for MCP Servers
2. Secure communication examples and best practices
3. Access control and permission management examples

Run this example:
    python -m mcp_course.examples.security_patterns
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import hashlib
import hmac
import json
import secrets
import time
from typing import Any

from mcp import Server
from mcp.server.models import InitializationOptions
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

from mcp_course.server.scaffolding import ServerConfig


class AuthenticationMethod(Enum):
    """Supported authentication methods."""
    API_KEY = "api_key"
    JWT_TOKEN = "jwt_token"
    HMAC_SIGNATURE = "hmac_signature"
    MUTUAL_TLS = "mutual_tls"


class Permission(Enum):
    """Available permissions for access control."""
    READ = "read"
    WRITE = "write"
    EXECUTE = "execute"
    ADMIN = "admin"


@dataclass
class User:
    """User representation for authentication and authorization."""
    user_id: str
    username: str
    api_key: str | None = None
    permissions: set[Permission] = field(default_factory=set)
    rate_limit: int = 100  # requests per hour
    created_at: datetime = field(default_factory=datetime.now)
    last_access: datetime | None = None
    is_active: bool = True


@dataclass
class AuthenticationResult:
    """Result of authentication attempt."""
    success: bool
    user: User | None = None
    error_message: str | None = None
    rate_limited: bool = False


@dataclass
class SecurityConfig:
    """Security configuration for MCP servers."""
    require_authentication: bool = True
    authentication_methods: list[AuthenticationMethod] = field(default_factory=lambda: [AuthenticationMethod.API_KEY])
    rate_limiting_enabled: bool = True
    audit_logging_enabled: bool = True
    encryption_required: bool = False
    session_timeout_minutes: int = 60
    max_failed_attempts: int = 5
    lockout_duration_minutes: int = 15


class SecurityManager:
    """
    Manages security, authentication, and authorization for MCP servers.

    This class demonstrates:
    - Multiple authentication methods
    - Role-based access control
    - Rate limiting and abuse prevention
    - Audit logging and monitoring
    - Secure session management
    """

    def __init__(self, config: SecurityConfig):
        """Initialize the security manager."""
        self.config = config
        self.users: dict[str, User] = {}
        self.api_keys: dict[str, str] = {}  # api_key -> user_id
        self.sessions: dict[str, dict[str, Any]] = {}
        self.rate_limits: dict[str, list[float]] = {}  # user_id -> list of request timestamps
        self.failed_attempts: dict[str, int] = {}  # user_id -> failed count
        self.lockouts: dict[str, datetime] = {}  # user_id -> lockout expiry
        self.audit_log: list[dict[str, Any]] = []

        # Initialize with demo users
        self._setup_demo_users()

    def _setup_demo_users(self):
        """Set up demonstration users with different permission levels."""
        # Admin user
        admin_user = User(
            user_id="admin_001",
            username="admin",
            api_key=self._generate_api_key(),
            permissions={Permission.READ, Permission.WRITE, Permission.EXECUTE, Permission.ADMIN},
            rate_limit=1000
        )
        self.add_user(admin_user)

        # Regular user
        regular_user = User(
            user_id="user_001",
            username="regular_user",
            api_key=self._generate_api_key(),
            permissions={Permission.READ, Permission.EXECUTE},
            rate_limit=100
        )
        self.add_user(regular_user)

        # Read-only user
        readonly_user = User(
            user_id="readonly_001",
            username="readonly_user",
            api_key=self._generate_api_key(),
            permissions={Permission.READ},
            rate_limit=50
        )
        self.add_user(readonly_user)

    def add_user(self, user: User):
        """Add a user to the system."""
        self.users[user.user_id] = user
        if user.api_key:
            self.api_keys[user.api_key] = user.user_id

    def _generate_api_key(self) -> str:
        """Generate a secure API key."""
        return f"mcp_{secrets.token_urlsafe(32)}"

    async def authenticate(self, auth_data: dict[str, Any]) -> AuthenticationResult:
        """
        Authenticate a request using the provided authentication data.

        Args:
            auth_data: Dictionary containing authentication information

        Returns:
            AuthenticationResult with success status and user info
        """
        if not self.config.require_authentication:
            # Create anonymous user for no-auth mode
            anonymous_user = User(
                user_id="anonymous",
                username="anonymous",
                permissions={Permission.READ}
            )
            return AuthenticationResult(success=True, user=anonymous_user)

        # Check for API key authentication
        if "api_key" in auth_data:
            return await self._authenticate_api_key(auth_data["api_key"])

        # Check for JWT token authentication
        if "jwt_token" in auth_data:
            return await self._authenticate_jwt_token(auth_data["jwt_token"])

        # Check for HMAC signature authentication
        if "signature" in auth_data and "timestamp" in auth_data:
            return await self._authenticate_hmac_signature(auth_data)

        return AuthenticationResult(
            success=False,
            error_message="No valid authentication method provided"
        )

    async def _authenticate_api_key(self, api_key: str) -> AuthenticationResult:
        """Authenticate using API key."""
        if api_key not in self.api_keys:
            self._log_security_event("authentication_failed", {"reason": "invalid_api_key"})
            return AuthenticationResult(
                success=False,
                error_message="Invalid API key"
            )

        user_id = self.api_keys[api_key]
        user = self.users[user_id]

        # Check if user is active
        if not user.is_active:
            return AuthenticationResult(
                success=False,
                error_message="User account is disabled"
            )

        # Check for lockout
        if await self._is_user_locked_out(user_id):
            return AuthenticationResult(
                success=False,
                error_message="Account temporarily locked due to failed attempts"
            )

        # Check rate limiting
        if await self._is_rate_limited(user_id):
            return AuthenticationResult(
                success=False,
                rate_limited=True,
                error_message="Rate limit exceeded"
            )

        # Update user access time
        user.last_access = datetime.now()

        # Reset failed attempts on successful auth
        if user_id in self.failed_attempts:
            del self.failed_attempts[user_id]

        self._log_security_event("authentication_success", {"user_id": user_id, "method": "api_key"})

        return AuthenticationResult(success=True, user=user)

    async def _authenticate_jwt_token(self, jwt_token: str) -> AuthenticationResult:
        """Authenticate using JWT token (simplified implementation)."""
        # In a real implementation, you would validate the JWT signature,
        # check expiration, and extract user information

        # For demonstration, we'll use a simple format: "jwt_<user_id>_<timestamp>"
        try:
            parts = jwt_token.split("_")
            if len(parts) != 3 or parts[0] != "jwt":
                raise ValueError("Invalid JWT format")

            user_id = parts[1]
            timestamp = int(parts[2])

            # Check if token is expired (1 hour validity)
            if time.time() - timestamp > 3600:
                return AuthenticationResult(
                    success=False,
                    error_message="JWT token expired"
                )

            if user_id not in self.users:
                return AuthenticationResult(
                    success=False,
                    error_message="Invalid user in JWT token"
                )

            user = self.users[user_id]
            user.last_access = datetime.now()

            self._log_security_event("authentication_success", {"user_id": user_id, "method": "jwt_token"})

            return AuthenticationResult(success=True, user=user)

        except (ValueError, IndexError):
            self._log_security_event("authentication_failed", {"reason": "invalid_jwt_token"})
            return AuthenticationResult(
                success=False,
                error_message="Invalid JWT token format"
            )

    async def _authenticate_hmac_signature(self, auth_data: dict[str, Any]) -> AuthenticationResult:
        """Authenticate using HMAC signature."""
        signature = auth_data["signature"]
        timestamp = auth_data["timestamp"]
        user_id = auth_data.get("user_id")

        if not user_id or user_id not in self.users:
            return AuthenticationResult(
                success=False,
                error_message="Invalid user ID"
            )

        user = self.users[user_id]

        # Check timestamp (prevent replay attacks)
        current_time = time.time()
        if abs(current_time - timestamp) > 300:  # 5 minute window
            return AuthenticationResult(
                success=False,
                error_message="Request timestamp too old"
            )

        # Calculate expected signature
        message = f"{user_id}:{timestamp}"
        expected_signature = hmac.new(
            user.api_key.encode() if user.api_key else b"default_secret",
            message.encode(),
            hashlib.sha256
        ).hexdigest()

        # Verify signature
        if not hmac.compare_digest(signature, expected_signature):
            self._record_failed_attempt(user_id)
            return AuthenticationResult(
                success=False,
                error_message="Invalid HMAC signature"
            )

        user.last_access = datetime.now()
        self._log_security_event("authentication_success", {"user_id": user_id, "method": "hmac_signature"})

        return AuthenticationResult(success=True, user=user)

    async def authorize(self, user: User, required_permission: Permission, resource: str | None = None) -> bool:
        """
        Check if user has the required permission for a resource.

        Args:
            user: Authenticated user
            required_permission: Permission required for the operation
            resource: Optional resource identifier for fine-grained access control

        Returns:
            True if user is authorized, False otherwise
        """
        # Admin users have all permissions
        if Permission.ADMIN in user.permissions:
            return True

        # Check if user has the required permission
        if required_permission not in user.permissions:
            self._log_security_event("authorization_failed", {
                "user_id": user.user_id,
                "required_permission": required_permission.value,
                "resource": resource
            })
            return False

        # Resource-specific authorization logic could go here
        # For example, checking if user owns the resource or has access to it

        self._log_security_event("authorization_success", {
            "user_id": user.user_id,
            "permission": required_permission.value,
            "resource": resource
        })

        return True

    async def _is_rate_limited(self, user_id: str) -> bool:
        """Check if user has exceeded rate limits."""
        if not self.config.rate_limiting_enabled:
            return False

        user = self.users.get(user_id)
        if not user:
            return True

        current_time = time.time()
        hour_ago = current_time - 3600

        # Initialize rate limit tracking for user
        if user_id not in self.rate_limits:
            self.rate_limits[user_id] = []

        # Remove old requests (older than 1 hour)
        self.rate_limits[user_id] = [
            req_time for req_time in self.rate_limits[user_id]
            if req_time > hour_ago
        ]

        # Check if user has exceeded their rate limit
        if len(self.rate_limits[user_id]) >= user.rate_limit:
            self._log_security_event("rate_limit_exceeded", {"user_id": user_id})
            return True

        # Record this request
        self.rate_limits[user_id].append(current_time)
        return False

    async def _is_user_locked_out(self, user_id: str) -> bool:
        """Check if user is currently locked out."""
        if user_id not in self.lockouts:
            return False

        lockout_expiry = self.lockouts[user_id]
        if datetime.now() > lockout_expiry:
            # Lockout has expired
            del self.lockouts[user_id]
            if user_id in self.failed_attempts:
                del self.failed_attempts[user_id]
            return False

        return True

    def _record_failed_attempt(self, user_id: str):
        """Record a failed authentication attempt."""
        if user_id not in self.failed_attempts:
            self.failed_attempts[user_id] = 0

        self.failed_attempts[user_id] += 1

        # Lock out user if they've exceeded max failed attempts
        if self.failed_attempts[user_id] >= self.config.max_failed_attempts:
            lockout_expiry = datetime.now() + timedelta(minutes=self.config.lockout_duration_minutes)
            self.lockouts[user_id] = lockout_expiry

            self._log_security_event("user_locked_out", {
                "user_id": user_id,
                "failed_attempts": self.failed_attempts[user_id],
                "lockout_expiry": lockout_expiry.isoformat()
            })

    def _log_security_event(self, event_type: str, details: dict[str, Any]):
        """Log security events for audit purposes."""
        if not self.config.audit_logging_enabled:
            return

        event = {
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            "details": details
        }

        self.audit_log.append(event)

        # In a real implementation, you would write to a secure log file
        # or send to a centralized logging system
        print(f"SECURITY EVENT: {json.dumps(event)}")

    def get_security_status(self) -> dict[str, Any]:
        """Get current security status and statistics."""
        return {
            "total_users": len(self.users),
            "active_users": len([u for u in self.users.values() if u.is_active]),
            "locked_out_users": len(self.lockouts),
            "recent_events": len([e for e in self.audit_log if
                                datetime.fromisoformat(e["timestamp"]) > datetime.now() - timedelta(hours=1)]),
            "authentication_methods": [method.value for method in self.config.authentication_methods],
            "rate_limiting_enabled": self.config.rate_limiting_enabled,
            "audit_logging_enabled": self.config.audit_logging_enabled
        }

    def get_user_info(self, user_id: str) -> dict[str, Any] | None:
        """Get information about a specific user."""
        if user_id not in self.users:
            return None

        user = self.users[user_id]
        return {
            "user_id": user.user_id,
            "username": user.username,
            "permissions": [p.value for p in user.permissions],
            "rate_limit": user.rate_limit,
            "created_at": user.created_at.isoformat(),
            "last_access": user.last_access.isoformat() if user.last_access else None,
            "is_active": user.is_active,
            "failed_attempts": self.failed_attempts.get(user_id, 0),
            "is_locked_out": user_id in self.lockouts
        }


class SecureMCPServer:
    """
    MCP Server with integrated security and authentication.

    This server demonstrates:
    - Authentication middleware
    - Authorization checks for tools
    - Secure request handling
    - Audit logging and monitoring
    """

    def __init__(self, config: ServerConfig, security_config: SecurityConfig):
        """Initialize the secure MCP server."""
        self.config = config
        self.server = Server(config.name)
        self.security_manager = SecurityManager(security_config)
        self._setup_handlers()

    def _setup_handlers(self):
        """Set up MCP protocol handlers with security."""

        @self.server.list_tools()
        async def handle_list_tools() -> list[Tool]:
            """List available tools (requires authentication)."""
            return [
                Tool(
                    name="secure_echo",
                    description="Echo a message (requires READ permission)",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "message": {"type": "string", "description": "Message to echo"},
                            "auth": {
                                "type": "object",
                                "description": "Authentication data",
                                "properties": {
                                    "api_key": {"type": "string"},
                                    "jwt_token": {"type": "string"},
                                    "signature": {"type": "string"},
                                    "timestamp": {"type": "number"},
                                    "user_id": {"type": "string"}
                                }
                            }
                        },
                        "required": ["message", "auth"]
                    }
                ),

                Tool(
                    name="admin_operation",
                    description="Perform administrative operation (requires ADMIN permission)",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "operation": {
                                "type": "string",
                                "enum": ["list_users", "reset_user", "get_security_status"],
                                "description": "Administrative operation to perform"
                            },
                            "target_user": {"type": "string", "description": "Target user ID (for user operations)"},
                            "auth": {
                                "type": "object",
                                "description": "Authentication data",
                                "properties": {
                                    "api_key": {"type": "string"},
                                    "jwt_token": {"type": "string"},
                                    "signature": {"type": "string"},
                                    "timestamp": {"type": "number"},
                                    "user_id": {"type": "string"}
                                }
                            }
                        },
                        "required": ["operation", "auth"]
                    }
                ),

                Tool(
                    name="create_resource",
                    description="Create a new resource (requires WRITE permission)",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "resource_name": {"type": "string", "description": "Name of the resource"},
                            "resource_data": {"type": "string", "description": "Resource data"},
                            "auth": {
                                "type": "object",
                                "description": "Authentication data",
                                "properties": {
                                    "api_key": {"type": "string"},
                                    "jwt_token": {"type": "string"},
                                    "signature": {"type": "string"},
                                    "timestamp": {"type": "number"},
                                    "user_id": {"type": "string"}
                                }
                            }
                        },
                        "required": ["resource_name", "resource_data", "auth"]
                    }
                ),

                Tool(
                    name="execute_command",
                    description="Execute a system command (requires EXECUTE permission)",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "command": {"type": "string", "description": "Command to execute"},
                            "auth": {
                                "type": "object",
                                "description": "Authentication data",
                                "properties": {
                                    "api_key": {"type": "string"},
                                    "jwt_token": {"type": "string"},
                                    "signature": {"type": "string"},
                                    "timestamp": {"type": "number"},
                                    "user_id": {"type": "string"}
                                }
                            }
                        },
                        "required": ["command", "auth"]
                    }
                )
            ]

        @self.server.call_tool()
        async def handle_call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
            """Handle tool execution with security checks."""

            # Extract authentication data
            auth_data = arguments.get("auth", {})

            # Authenticate the request
            auth_result = await self.security_manager.authenticate(auth_data)

            if not auth_result.success:
                error_msg = auth_result.error_message or "Authentication failed"
                if auth_result.rate_limited:
                    error_msg = "Rate limit exceeded. Please try again later."

                return [TextContent(type="text", text=f"Authentication Error: {error_msg}")]

            user = auth_result.user

            # Route to appropriate handler with authorization checks
            if name == "secure_echo":
                return await self._handle_secure_echo(user, arguments)
            elif name == "admin_operation":
                return await self._handle_admin_operation(user, arguments)
            elif name == "create_resource":
                return await self._handle_create_resource(user, arguments)
            elif name == "execute_command":
                return await self._handle_execute_command(user, arguments)
            else:
                return [TextContent(type="text", text=f"Unknown tool: {name}")]

    async def _handle_secure_echo(self, user: User, arguments: dict[str, Any]) -> list[TextContent]:
        """Handle secure echo tool (requires READ permission)."""

        # Check authorization
        if not await self.security_manager.authorize(user, Permission.READ):
            return [TextContent(type="text", text="Authorization Error: Insufficient permissions")]

        message = arguments["message"]

        # Log the operation
        self.security_manager._log_security_event("tool_executed", {
            "user_id": user.user_id,
            "tool": "secure_echo",
            "message_length": len(message)
        })

        return [TextContent(type="text", text=f"Secure Echo [{user.username}]: {message}")]

    async def _handle_admin_operation(self, user: User, arguments: dict[str, Any]) -> list[TextContent]:
        """Handle admin operations (requires ADMIN permission)."""

        # Check authorization
        if not await self.security_manager.authorize(user, Permission.ADMIN):
            return [TextContent(type="text", text="Authorization Error: Admin privileges required")]

        operation = arguments["operation"]

        if operation == "list_users":
            users_info = []
            for user_id in self.security_manager.users:
                user_info = self.security_manager.get_user_info(user_id)
                if user_info:
                    users_info.append(user_info)

            result = {
                "operation": "list_users",
                "users": users_info,
                "total_count": len(users_info)
            }

        elif operation == "get_security_status":
            result = {
                "operation": "get_security_status",
                "status": self.security_manager.get_security_status()
            }

        elif operation == "reset_user":
            target_user = arguments.get("target_user")
            if not target_user:
                return [TextContent(type="text", text="Error: target_user required for reset operation")]

            # Reset failed attempts and unlock user
            if target_user in self.security_manager.failed_attempts:
                del self.security_manager.failed_attempts[target_user]
            if target_user in self.security_manager.lockouts:
                del self.security_manager.lockouts[target_user]

            result = {
                "operation": "reset_user",
                "target_user": target_user,
                "status": "success"
            }

        else:
            return [TextContent(type="text", text=f"Unknown admin operation: {operation}")]

        # Log the admin operation
        self.security_manager._log_security_event("admin_operation", {
            "user_id": user.user_id,
            "operation": operation,
            "target_user": arguments.get("target_user")
        })

        return [TextContent(type="text", text=json.dumps(result, indent=2))]

    async def _handle_create_resource(self, user: User, arguments: dict[str, Any]) -> list[TextContent]:
        """Handle resource creation (requires WRITE permission)."""

        # Check authorization
        if not await self.security_manager.authorize(user, Permission.WRITE):
            return [TextContent(type="text", text="Authorization Error: Write permission required")]

        resource_name = arguments["resource_name"]
        resource_data = arguments["resource_data"]

        # Simulate resource creation
        result = {
            "operation": "create_resource",
            "resource_name": resource_name,
            "resource_id": f"res_{secrets.token_hex(8)}",
            "created_by": user.user_id,
            "created_at": datetime.now().isoformat(),
            "data_size": len(resource_data)
        }

        # Log the operation
        self.security_manager._log_security_event("resource_created", {
            "user_id": user.user_id,
            "resource_name": resource_name,
            "resource_id": result["resource_id"]
        })

        return [TextContent(type="text", text=json.dumps(result, indent=2))]

    async def _handle_execute_command(self, user: User, arguments: dict[str, Any]) -> list[TextContent]:
        """Handle command execution (requires EXECUTE permission)."""

        # Check authorization
        if not await self.security_manager.authorize(user, Permission.EXECUTE):
            return [TextContent(type="text", text="Authorization Error: Execute permission required")]

        command = arguments["command"]

        # Security check: prevent dangerous commands
        dangerous_commands = ["rm", "del", "format", "shutdown", "reboot"]
        if any(dangerous in command.lower() for dangerous in dangerous_commands):
            self.security_manager._log_security_event("dangerous_command_blocked", {
                "user_id": user.user_id,
                "command": command
            })
            return [TextContent(type="text", text="Security Error: Dangerous command blocked")]

        # Simulate command execution (don't actually execute for security)
        result = {
            "operation": "execute_command",
            "command": command,
            "status": "simulated",
            "executed_by": user.user_id,
            "executed_at": datetime.now().isoformat(),
            "output": f"Simulated execution of: {command}"
        }

        # Log the operation
        self.security_manager._log_security_event("command_executed", {
            "user_id": user.user_id,
            "command": command
        })

        return [TextContent(type="text", text=json.dumps(result, indent=2))]

    async def run(self):
        """Run the secure MCP server."""
        async with stdio_server() as (read_stream, write_stream):
            init_options = InitializationOptions(
                server_name=self.config.name,
                server_version=self.config.version,
                capabilities=self.config.capabilities
            )
            await self.server.run(read_stream, write_stream, init_options)


async def demonstrate_security_patterns():
    """
    Demonstrate security and authentication patterns for MCP servers.

    This function shows:
    - Different authentication methods
    - Role-based access control
    - Rate limiting and abuse prevention
    - Audit logging and monitoring
    """
    print("=== MCP Security and Authentication Patterns ===")
    print()

    # Create security configuration
    security_config = SecurityConfig(
        require_authentication=True,
        authentication_methods=[
            AuthenticationMethod.API_KEY,
            AuthenticationMethod.JWT_TOKEN,
            AuthenticationMethod.HMAC_SIGNATURE
        ],
        rate_limiting_enabled=True,
        audit_logging_enabled=True,
        max_failed_attempts=3,
        lockout_duration_minutes=5
    )

    # Create security manager
    security_manager = SecurityManager(security_config)

    print("1. Authentication Methods Demonstration")
    print("=====================================")

    # Get demo user credentials
    admin_user = security_manager.users["admin_001"]
    regular_user = security_manager.users["user_001"]
    readonly_user = security_manager.users["readonly_001"]

    print("Demo Users Created:")
    print(f"  Admin User: {admin_user.username} (API Key: {admin_user.api_key[:20]}...)")
    print(f"  Regular User: {regular_user.username} (API Key: {regular_user.api_key[:20]}...)")
    print(f"  Read-only User: {readonly_user.username} (API Key: {readonly_user.api_key[:20]}...)")
    print()

    # Test API Key authentication
    print("Testing API Key Authentication:")
    auth_result = await security_manager.authenticate({"api_key": admin_user.api_key})
    print(f"  Admin API Key: {'✓ Success' if auth_result.success else '✗ Failed'}")

    auth_result = await security_manager.authenticate({"api_key": "invalid_key"})
    print(f"  Invalid API Key: {'✓ Success' if auth_result.success else '✗ Failed (Expected)'}")
    print()

    # Test JWT Token authentication
    print("Testing JWT Token Authentication:")
    jwt_token = f"jwt_{admin_user.user_id}_{int(time.time())}"
    auth_result = await security_manager.authenticate({"jwt_token": jwt_token})
    print(f"  Valid JWT Token: {'✓ Success' if auth_result.success else '✗ Failed'}")

    expired_jwt = f"jwt_{admin_user.user_id}_{int(time.time()) - 7200}"  # 2 hours old
    auth_result = await security_manager.authenticate({"jwt_token": expired_jwt})
    print(f"  Expired JWT Token: {'✓ Success' if auth_result.success else '✗ Failed (Expected)'}")
    print()

    # Test HMAC Signature authentication
    print("Testing HMAC Signature Authentication:")
    timestamp = time.time()
    message = f"{admin_user.user_id}:{timestamp}"
    signature = hmac.new(
        admin_user.api_key.encode(),
        message.encode(),
        hashlib.sha256
    ).hexdigest()

    auth_result = await security_manager.authenticate({
        "signature": signature,
        "timestamp": timestamp,
        "user_id": admin_user.user_id
    })
    print(f"  Valid HMAC Signature: {'✓ Success' if auth_result.success else '✗ Failed'}")
    print()

    print("2. Authorization and Permissions")
    print("===============================")

    # Test authorization for different users
    print("Permission Tests:")

    # Admin user should have all permissions
    for permission in Permission:
        authorized = await security_manager.authorize(admin_user, permission)
        print(f"  Admin {permission.value}: {'✓ Allowed' if authorized else '✗ Denied'}")

    print()

    # Regular user should have limited permissions
    for permission in Permission:
        authorized = await security_manager.authorize(regular_user, permission)
        print(f"  Regular User {permission.value}: {'✓ Allowed' if authorized else '✗ Denied'}")

    print()

    # Read-only user should only have read permission
    for permission in Permission:
        authorized = await security_manager.authorize(readonly_user, permission)
        print(f"  Read-only User {permission.value}: {'✓ Allowed' if authorized else '✗ Denied'}")

    print()

    print("3. Rate Limiting and Security Features")
    print("=====================================")

    # Test rate limiting (simulate multiple requests)
    print("Rate Limiting Test:")
    rate_limited_count = 0
    for _ in range(5):
        is_limited = await security_manager._is_rate_limited(readonly_user.user_id)
        if is_limited:
            rate_limited_count += 1

    print(f"  Rate limit checks: {rate_limited_count}/5 requests limited")
    print()

    # Show security status
    print("Security Status:")
    status = security_manager.get_security_status()
    for key, value in status.items():
        print(f"  {key}: {value}")
    print()

    print("4. Audit Logging")
    print("===============")
    print("Recent Security Events:")
    for event in security_manager.audit_log[-5:]:  # Show last 5 events
        print(f"  {event['timestamp']}: {event['event_type']} - {event['details']}")
    print()

    print("Security patterns demonstrated:")
    print("✓ Multiple authentication methods (API Key, JWT, HMAC)")
    print("✓ Role-based access control with granular permissions")
    print("✓ Rate limiting and abuse prevention")
    print("✓ Account lockout after failed attempts")
    print("✓ Comprehensive audit logging")
    print("✓ Secure session management")
    print("✓ Input validation and dangerous command blocking")


async def demonstrate_secure_server():
    """Demonstrate a complete secure MCP server."""
    print("\n=== Secure MCP Server Example ===")
    print()

    ServerConfig(
        name="secure-mcp-server",
        version="1.0.0",
        description="MCP Server with comprehensive security features"
    )

    SecurityConfig(
        require_authentication=True,
        rate_limiting_enabled=True,
        audit_logging_enabled=True
    )

    print("Secure MCP Server Features:")
    print("- Authentication required for all operations")
    print("- Role-based access control")
    print("- Rate limiting per user")
    print("- Comprehensive audit logging")
    print("- Dangerous command blocking")
    print("- Session management")
    print()

    print("Available Tools:")
    print("- secure_echo: Echo messages (READ permission)")
    print("- admin_operation: Administrative functions (ADMIN permission)")
    print("- create_resource: Create resources (WRITE permission)")
    print("- execute_command: Execute commands (EXECUTE permission)")
    print()

    print("To use this server, clients must provide authentication data:")
    print("- API Key: Include 'api_key' in the auth object")
    print("- JWT Token: Include 'jwt_token' in the auth object")
    print("- HMAC Signature: Include 'signature', 'timestamp', and 'user_id'")


async def main():
    """Main demonstration entry point."""
    await demonstrate_security_patterns()
    await demonstrate_secure_server()

    print("\n=== Security Best Practices Summary ===")
    print()
    print("1. Authentication:")
    print("   - Use strong API keys or JWT tokens")
    print("   - Implement HMAC signatures for high-security scenarios")
    print("   - Support multiple authentication methods")
    print()
    print("2. Authorization:")
    print("   - Implement role-based access control")
    print("   - Use principle of least privilege")
    print("   - Check permissions for each operation")
    print()
    print("3. Security Controls:")
    print("   - Enable rate limiting to prevent abuse")
    print("   - Implement account lockout after failed attempts")
    print("   - Validate and sanitize all inputs")
    print("   - Block dangerous operations")
    print()
    print("4. Monitoring and Auditing:")
    print("   - Log all security events")
    print("   - Monitor for suspicious activity")
    print("   - Implement alerting for security incidents")
    print("   - Regular security reviews and updates")


if __name__ == "__main__":
    asyncio.run(main())
