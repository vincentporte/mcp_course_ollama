#!/usr/bin/env python3
"""
Secure Communication Patterns for MCP

This module demonstrates secure communication patterns and best practices
for MCP client-server interactions, including:

1. Encrypted communication channels
2. Certificate validation and mutual TLS
3. Secure message signing and verification
4. Data encryption and decryption utilities

Run this example:
    python -m mcp_course.examples.secure_communication
"""

import asyncio
import base64
from dataclasses import dataclass
from datetime import datetime, timedelta
import json
from pathlib import Path
import secrets
import ssl
from typing import Any

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.x509.oid import NameOID

from mcp_course.client.basic import BasicMCPClient, ClientConfig


@dataclass
class EncryptionConfig:
    """Configuration for encryption settings."""
    algorithm: str = "AES-256-GCM"
    key_size: int = 32  # 256 bits
    iv_size: int = 12   # 96 bits for GCM
    tag_size: int = 16  # 128 bits for GCM
    pbkdf2_iterations: int = 100000


@dataclass
class CertificateInfo:
    """Information about a certificate."""
    subject: str
    issuer: str
    serial_number: str
    not_valid_before: datetime
    not_valid_after: datetime
    fingerprint: str
    is_self_signed: bool


class CryptographyManager:
    """
    Manages cryptographic operations for secure MCP communication.

    This class provides:
    - Symmetric encryption/decryption
    - Asymmetric key generation and operations
    - Digital signatures
    - Certificate management
    - Secure key derivation
    """

    def __init__(self, config: EncryptionConfig = None):
        """Initialize the cryptography manager."""
        self.config = config or EncryptionConfig()
        self.private_key: rsa.RSAPrivateKey | None = None
        self.public_key: rsa.RSAPublicKey | None = None
        self.certificate: x509.Certificate | None = None

    def generate_key_pair(self, key_size: int = 2048) -> tuple[rsa.RSAPrivateKey, rsa.RSAPublicKey]:
        """Generate an RSA key pair."""
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=key_size
        )
        public_key = private_key.public_key()

        self.private_key = private_key
        self.public_key = public_key

        return private_key, public_key

    def create_self_signed_certificate(
        self,
        private_key: rsa.RSAPrivateKey,
        subject_name: str = "MCP Server",
        validity_days: int = 365
    ) -> x509.Certificate:
        """Create a self-signed certificate."""

        # Create subject and issuer (same for self-signed)
        subject = issuer = x509.Name([
            x509.NameAttribute(NameOID.COUNTRY_NAME, "US"),
            x509.NameAttribute(NameOID.STATE_OR_PROVINCE_NAME, "CA"),
            x509.NameAttribute(NameOID.LOCALITY_NAME, "San Francisco"),
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, "MCP Course"),
            x509.NameAttribute(NameOID.COMMON_NAME, subject_name),
        ])

        # Create certificate
        cert = x509.CertificateBuilder().subject_name(
            subject
        ).issuer_name(
            issuer
        ).public_key(
            private_key.public_key()
        ).serial_number(
            x509.random_serial_number()
        ).not_valid_before(
            datetime.utcnow()
        ).not_valid_after(
            datetime.utcnow() + timedelta(days=validity_days)
        ).add_extension(
            x509.SubjectAlternativeName([
                x509.DNSName("localhost"),
                x509.IPAddress("127.0.0.1"),
            ]),
            critical=False,
        ).sign(private_key, hashes.SHA256())

        self.certificate = cert
        return cert

    def get_certificate_info(self, certificate: x509.Certificate) -> CertificateInfo:
        """Extract information from a certificate."""

        # Get subject and issuer names
        subject = certificate.subject.rfc4514_string()
        issuer = certificate.issuer.rfc4514_string()

        # Calculate fingerprint
        fingerprint = certificate.fingerprint(hashes.SHA256()).hex()

        # Check if self-signed
        is_self_signed = subject == issuer

        return CertificateInfo(
            subject=subject,
            issuer=issuer,
            serial_number=str(certificate.serial_number),
            not_valid_before=certificate.not_valid_before,
            not_valid_after=certificate.not_valid_after,
            fingerprint=fingerprint,
            is_self_signed=is_self_signed
        )

    def encrypt_symmetric(self, data: bytes, password: str) -> dict[str, str]:
        """Encrypt data using symmetric encryption (AES-GCM)."""

        # Generate salt and derive key
        salt = secrets.token_bytes(16)
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=self.config.key_size,
            salt=salt,
            iterations=self.config.pbkdf2_iterations,
        )
        key = kdf.derive(password.encode())

        # Generate IV
        iv = secrets.token_bytes(self.config.iv_size)

        # Encrypt data
        cipher = Cipher(algorithms.AES(key), modes.GCM(iv))
        encryptor = cipher.encryptor()
        ciphertext = encryptor.update(data) + encryptor.finalize()

        # Return encrypted data with metadata
        return {
            "ciphertext": base64.b64encode(ciphertext).decode(),
            "iv": base64.b64encode(iv).decode(),
            "salt": base64.b64encode(salt).decode(),
            "tag": base64.b64encode(encryptor.tag).decode(),
            "algorithm": self.config.algorithm,
            "iterations": self.config.pbkdf2_iterations
        }

    def decrypt_symmetric(self, encrypted_data: dict[str, str], password: str) -> bytes:
        """Decrypt data using symmetric encryption (AES-GCM)."""

        # Extract components
        ciphertext = base64.b64decode(encrypted_data["ciphertext"])
        iv = base64.b64decode(encrypted_data["iv"])
        salt = base64.b64decode(encrypted_data["salt"])
        tag = base64.b64decode(encrypted_data["tag"])
        iterations = encrypted_data["iterations"]

        # Derive key
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=self.config.key_size,
            salt=salt,
            iterations=iterations,
        )
        key = kdf.derive(password.encode())

        # Decrypt data
        cipher = Cipher(algorithms.AES(key), modes.GCM(iv, tag))
        decryptor = cipher.decryptor()
        plaintext = decryptor.update(ciphertext) + decryptor.finalize()

        return plaintext

    def encrypt_asymmetric(self, data: bytes, public_key: rsa.RSAPublicKey) -> bytes:
        """Encrypt data using asymmetric encryption (RSA)."""

        # RSA can only encrypt small amounts of data
        # For larger data, use hybrid encryption (RSA + AES)
        max_size = (public_key.key_size // 8) - 2 * (hashes.SHA256().digest_size) - 2

        if len(data) <= max_size:
            # Direct RSA encryption for small data
            ciphertext = public_key.encrypt(
                data,
                padding.OAEP(
                    mgf=padding.MGF1(algorithm=hashes.SHA256()),
                    algorithm=hashes.SHA256(),
                    label=None
                )
            )
            return ciphertext
        else:
            # Hybrid encryption for larger data
            # Generate symmetric key
            symmetric_key = secrets.token_bytes(32)

            # Encrypt data with symmetric key
            iv = secrets.token_bytes(12)
            cipher = Cipher(algorithms.AES(symmetric_key), modes.GCM(iv))
            encryptor = cipher.encryptor()
            ciphertext = encryptor.update(data) + encryptor.finalize()

            # Encrypt symmetric key with RSA
            encrypted_key = public_key.encrypt(
                symmetric_key,
                padding.OAEP(
                    mgf=padding.MGF1(algorithm=hashes.SHA256()),
                    algorithm=hashes.SHA256(),
                    label=None
                )
            )

            # Combine encrypted key, IV, tag, and ciphertext
            result = (
                len(encrypted_key).to_bytes(4, 'big') +
                encrypted_key +
                iv +
                encryptor.tag +
                ciphertext
            )

            return result

    def decrypt_asymmetric(self, encrypted_data: bytes, private_key: rsa.RSAPrivateKey) -> bytes:
        """Decrypt data using asymmetric encryption (RSA)."""

        try:
            # Try direct RSA decryption first
            plaintext = private_key.decrypt(
                encrypted_data,
                padding.OAEP(
                    mgf=padding.MGF1(algorithm=hashes.SHA256()),
                    algorithm=hashes.SHA256(),
                    label=None
                )
            )
            return plaintext
        except ValueError:
            # Hybrid decryption
            # Extract encrypted key length
            key_length = int.from_bytes(encrypted_data[:4], 'big')

            # Extract components
            encrypted_key = encrypted_data[4:4+key_length]
            iv = encrypted_data[4+key_length:4+key_length+12]
            tag = encrypted_data[4+key_length+12:4+key_length+12+16]
            ciphertext = encrypted_data[4+key_length+12+16:]

            # Decrypt symmetric key
            symmetric_key = private_key.decrypt(
                encrypted_key,
                padding.OAEP(
                    mgf=padding.MGF1(algorithm=hashes.SHA256()),
                    algorithm=hashes.SHA256(),
                    label=None
                )
            )

            # Decrypt data
            cipher = Cipher(algorithms.AES(symmetric_key), modes.GCM(iv, tag))
            decryptor = cipher.decryptor()
            plaintext = decryptor.update(ciphertext) + decryptor.finalize()

            return plaintext

    def sign_data(self, data: bytes, private_key: rsa.RSAPrivateKey) -> bytes:
        """Create a digital signature for data."""
        signature = private_key.sign(
            data,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256()
        )
        return signature

    def verify_signature(self, data: bytes, signature: bytes, public_key: rsa.RSAPublicKey) -> bool:
        """Verify a digital signature."""
        try:
            public_key.verify(
                signature,
                data,
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
            return True
        except Exception:
            return False

    def save_private_key(self, private_key: rsa.RSAPrivateKey, filename: str, password: str | None = None):
        """Save private key to file."""
        encryption_algorithm = serialization.NoEncryption()
        if password:
            encryption_algorithm = serialization.BestAvailableEncryption(password.encode())

        pem = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=encryption_algorithm
        )

        with Path.open(filename, 'wb') as f:
            f.write(pem)

    def load_private_key(self, filename: str, password: str | None = None) -> rsa.RSAPrivateKey:
        """Load private key from file."""
        with Path.open(filename, 'rb') as f:
            pem_data = f.read()

        password_bytes = password.encode() if password else None
        private_key = serialization.load_pem_private_key(pem_data, password=password_bytes)

        return private_key

    def save_certificate(self, certificate: x509.Certificate, filename: str):
        """Save certificate to file."""
        pem = certificate.public_bytes(serialization.Encoding.PEM)

        with Path.open(filename, 'wb') as f:
            f.write(pem)

    def load_certificate(self, filename: str) -> x509.Certificate:
        """Load certificate from file."""
        with Path.open(filename, 'rb') as f:
            pem_data = f.read()

        certificate = x509.load_pem_x509_certificate(pem_data)
        return certificate


class SecureMCPClient:
    """
    MCP Client with enhanced security features.

    This client demonstrates:
    - Encrypted communication
    - Certificate validation
    - Message signing and verification
    - Secure credential management
    """

    def __init__(self, client_config: ClientConfig = None):
        """Initialize the secure MCP client."""
        self.client = BasicMCPClient(client_config or ClientConfig(name="secure-mcp-client"))
        self.crypto_manager = CryptographyManager()
        self.server_certificates: dict[str, x509.Certificate] = {}
        self.trusted_fingerprints: set[str] = set()

    async def setup_encryption(self, generate_keys: bool = True):
        """Set up encryption keys and certificates."""
        if generate_keys:
            private_key, _public_key = self.crypto_manager.generate_key_pair()
            certificate = self.crypto_manager.create_self_signed_certificate(
                private_key,
                "MCP Secure Client"
            )

            print("Generated client certificate:")
            cert_info = self.crypto_manager.get_certificate_info(certificate)
            print(f"  Subject: {cert_info.subject}")
            print(f"  Fingerprint: {cert_info.fingerprint}")
            print(f"  Valid until: {cert_info.not_valid_after}")

    async def add_trusted_server(self, server_name: str, certificate_path: str):
        """Add a trusted server certificate."""
        try:
            certificate = self.crypto_manager.load_certificate(certificate_path)
            cert_info = self.crypto_manager.get_certificate_info(certificate)

            self.server_certificates[server_name] = certificate
            self.trusted_fingerprints.add(cert_info.fingerprint)

            print(f"Added trusted server: {server_name}")
            print(f"  Fingerprint: {cert_info.fingerprint}")

        except Exception as e:
            print(f"Error adding trusted server: {e}")

    async def encrypt_message(self, message: dict[str, Any], server_name: str) -> dict[str, Any]:
        """Encrypt a message for a specific server."""
        if server_name not in self.server_certificates:
            raise ValueError(f"No certificate found for server: {server_name}")

        # Serialize message
        message_bytes = json.dumps(message).encode()

        # Get server's public key
        server_cert = self.server_certificates[server_name]
        server_public_key = server_cert.public_key()

        # Encrypt message
        encrypted_data = self.crypto_manager.encrypt_asymmetric(message_bytes, server_public_key)

        # Sign the encrypted data
        signature = self.crypto_manager.sign_data(encrypted_data, self.crypto_manager.private_key)

        return {
            "encrypted_data": base64.b64encode(encrypted_data).decode(),
            "signature": base64.b64encode(signature).decode(),
            "client_fingerprint": self.crypto_manager.get_certificate_info(
                self.crypto_manager.certificate
            ).fingerprint
        }

    async def decrypt_message(self, encrypted_message: dict[str, Any], server_name: str) -> dict[str, Any]:
        """Decrypt a message from a specific server."""
        if server_name not in self.server_certificates:
            raise ValueError(f"No certificate found for server: {server_name}")

        # Extract components
        encrypted_data = base64.b64decode(encrypted_message["encrypted_data"])
        signature = base64.b64decode(encrypted_message["signature"])

        # Verify signature
        server_cert = self.server_certificates[server_name]
        server_public_key = server_cert.public_key()

        if not self.crypto_manager.verify_signature(encrypted_data, signature, server_public_key):
            raise ValueError("Invalid signature from server")

        # Decrypt message
        decrypted_data = self.crypto_manager.decrypt_asymmetric(
            encrypted_data,
            self.crypto_manager.private_key
        )

        # Parse message
        message = json.loads(decrypted_data.decode())
        return message

    def create_ssl_context(self, verify_mode: ssl.VerifyMode = ssl.CERT_REQUIRED) -> ssl.SSLContext:
        """Create SSL context for secure connections."""
        context = ssl.create_default_context(ssl.Purpose.SERVER_AUTH)
        context.verify_mode = verify_mode

        # Load client certificate if available
        if self.crypto_manager.certificate and self.crypto_manager.private_key:
            # Save temporary files for SSL context
            cert_file = "temp_client_cert.pem"
            key_file = "temp_client_key.pem"

            self.crypto_manager.save_certificate(self.crypto_manager.certificate, cert_file)
            self.crypto_manager.save_private_key(self.crypto_manager.private_key, key_file)

            context.load_cert_chain(cert_file, key_file)

            # Clean up temporary files
            Path(cert_file).unlink()
            Path(key_file).unlink()

        return context

    async def validate_server_certificate(self, server_name: str, presented_cert: x509.Certificate) -> bool:
        """Validate a server's certificate."""
        cert_info = self.crypto_manager.get_certificate_info(presented_cert)

        # Check if certificate is in trusted list
        if cert_info.fingerprint in self.trusted_fingerprints:
            return True

        # Check certificate validity
        now = datetime.utcnow()
        if now < cert_info.not_valid_before or now > cert_info.not_valid_after:
            print(f"Certificate for {server_name} is not valid at current time")
            return False

        # Additional validation logic could go here
        # - Check certificate chain
        # - Verify against CA
        # - Check revocation status

        return False


async def demonstrate_cryptographic_operations():
    """Demonstrate cryptographic operations for secure MCP communication."""
    print("=== Cryptographic Operations for MCP Security ===")
    print()

    crypto_manager = CryptographyManager()

    print("1. Key Generation and Certificate Creation")
    print("=========================================")

    # Generate key pair
    private_key, public_key = crypto_manager.generate_key_pair()
    print("✓ Generated RSA key pair (2048 bits)")

    # Create self-signed certificate
    certificate = crypto_manager.create_self_signed_certificate(private_key, "MCP Demo Server")
    cert_info = crypto_manager.get_certificate_info(certificate)

    print("✓ Created self-signed certificate:")
    print(f"  Subject: {cert_info.subject}")
    print(f"  Serial: {cert_info.serial_number}")
    print(f"  Fingerprint: {cert_info.fingerprint[:32]}...")
    print(f"  Valid from: {cert_info.not_valid_before}")
    print(f"  Valid until: {cert_info.not_valid_after}")
    print()

    print("2. Symmetric Encryption")
    print("======================")

    # Test symmetric encryption
    test_data = "This is sensitive MCP communication data that needs to be encrypted."
    password = "secure_mcp_password_123"

    print(f"Original data: {test_data}")

    # Encrypt
    encrypted = crypto_manager.encrypt_symmetric(test_data.encode(), password)
    print("✓ Data encrypted with AES-256-GCM")
    print(f"  Algorithm: {encrypted['algorithm']}")
    print(f"  Ciphertext length: {len(encrypted['ciphertext'])} characters")

    # Decrypt
    decrypted = crypto_manager.decrypt_symmetric(encrypted, password)
    print(f"✓ Data decrypted: {decrypted.decode()}")
    print(f"  Encryption/Decryption: {'✓ Success' if decrypted.decode() == test_data else '✗ Failed'}")
    print()

    print("3. Asymmetric Encryption")
    print("=======================")

    # Test asymmetric encryption
    test_message = "Confidential MCP server configuration data."
    print(f"Original message: {test_message}")

    # Encrypt with public key
    encrypted_data = crypto_manager.encrypt_asymmetric(test_message.encode(), public_key)
    print("✓ Message encrypted with RSA public key")
    print(f"  Encrypted data length: {len(encrypted_data)} bytes")

    # Decrypt with private key
    decrypted_data = crypto_manager.decrypt_asymmetric(encrypted_data, private_key)
    print(f"✓ Message decrypted: {decrypted_data.decode()}")
    print(f"  Encryption/Decryption: {'✓ Success' if decrypted_data.decode() == test_message else '✗ Failed'}")
    print()

    print("4. Digital Signatures")
    print("====================")

    # Test digital signatures
    document = "MCP tool execution request: calculate_statistics with parameters {...}"
    print(f"Document to sign: {document}")

    # Sign document
    signature = crypto_manager.sign_data(document.encode(), private_key)
    print("✓ Document signed with RSA private key")
    print(f"  Signature length: {len(signature)} bytes")

    # Verify signature
    is_valid = crypto_manager.verify_signature(document.encode(), signature, public_key)
    print(f"✓ Signature verification: {'✓ Valid' if is_valid else '✗ Invalid'}")

    # Test with tampered document
    tampered_document = document + " [TAMPERED]"
    is_valid_tampered = crypto_manager.verify_signature(tampered_document.encode(), signature, public_key)
    print(f"✓ Tampered document verification: {'✓ Valid' if is_valid_tampered else '✗ Invalid (Expected)'}")
    print()

    print("5. Hybrid Encryption (Large Data)")
    print("================================")

    # Test with larger data that requires hybrid encryption
    large_data = "Large MCP response data: " + "x" * 1000  # 1KB+ data
    print(f"Large data size: {len(large_data)} characters")

    # Encrypt large data
    encrypted_large = crypto_manager.encrypt_asymmetric(large_data.encode(), public_key)
    print("✓ Large data encrypted using hybrid encryption (RSA + AES)")
    print(f"  Encrypted size: {len(encrypted_large)} bytes")

    # Decrypt large data
    decrypted_large = crypto_manager.decrypt_asymmetric(encrypted_large, private_key)
    print("✓ Large data decrypted successfully")
    print(f"  Hybrid encryption: {'✓ Success' if decrypted_large.decode() == large_data else '✗ Failed'}")


async def demonstrate_secure_client():
    """Demonstrate secure MCP client capabilities."""
    print("\n=== Secure MCP Client Demonstration ===")
    print()

    # Create secure client
    secure_client = SecureMCPClient()

    print("1. Client Setup and Key Generation")
    print("=================================")

    # Setup encryption
    await secure_client.setup_encryption()
    print()

    print("2. Secure Message Exchange")
    print("=========================")

    # Simulate server certificate (in real scenario, this would be received from server)
    server_crypto = CryptographyManager()
    server_private_key, _server_public_key = server_crypto.generate_key_pair()
    server_certificate = server_crypto.create_self_signed_certificate(
        server_private_key,
        "MCP Demo Server"
    )

    # Save and add server certificate to client's trusted list
    server_cert_file = "demo_server_cert.pem"
    server_crypto.save_certificate(server_certificate, server_cert_file)
    await secure_client.add_trusted_server("demo-server", server_cert_file)

    # Test message encryption/decryption
    test_message = {
        "method": "call_tool",
        "params": {
            "name": "secure_echo",
            "arguments": {"message": "Hello from secure client!"}
        }
    }

    print("Original message:")
    print(json.dumps(test_message, indent=2))
    print()

    try:
        # Encrypt message
        encrypted_msg = await secure_client.encrypt_message(test_message, "demo-server")
        print("✓ Message encrypted for server")
        print(f"  Encrypted data length: {len(encrypted_msg['encrypted_data'])} characters")
        print(f"  Signature length: {len(encrypted_msg['signature'])} characters")
        print()

        # Simulate server decrypting and processing message
        # (In real scenario, server would decrypt, process, and send encrypted response)

        # Create response message
        response_message = {
            "result": "Secure Echo: Hello from secure client!",
            "status": "success",
            "timestamp": datetime.now().isoformat()
        }

        # Server encrypts response
        response_bytes = json.dumps(response_message).encode()
        encrypted_response_data = server_crypto.encrypt_asymmetric(
            response_bytes,
            secure_client.crypto_manager.public_key
        )
        response_signature = server_crypto.sign_data(encrypted_response_data, server_private_key)

        encrypted_response = {
            "encrypted_data": base64.b64encode(encrypted_response_data).decode(),
            "signature": base64.b64encode(response_signature).decode(),
            "server_fingerprint": server_crypto.get_certificate_info(server_certificate).fingerprint
        }

        # Client decrypts response
        decrypted_response = await secure_client.decrypt_message(encrypted_response, "demo-server")
        print("✓ Response decrypted from server")
        print("Decrypted response:")
        print(json.dumps(decrypted_response, indent=2))

    except Exception as e:
        print(f"✗ Error in secure communication: {e}")

    finally:
        # Clean up
        if Path(server_cert_file).exists():
            Path(server_cert_file).unlink()

    print()

    print("3. SSL/TLS Configuration")
    print("=======================")

    # Demonstrate SSL context creation
    ssl_context = secure_client.create_ssl_context()
    print("✓ SSL context created for secure connections")
    print(f"  Protocol: {ssl_context.protocol}")
    print(f"  Verify mode: {ssl_context.verify_mode}")
    print(f"  Check hostname: {ssl_context.check_hostname}")


async def demonstrate_security_best_practices():
    """Demonstrate security best practices for MCP."""
    print("\n=== MCP Security Best Practices ===")
    print()

    print("1. Communication Security:")
    print("   ✓ Use TLS 1.3 for transport encryption")
    print("   ✓ Implement certificate pinning for known servers")
    print("   ✓ Validate server certificates and check revocation")
    print("   ✓ Use mutual TLS for high-security environments")
    print()

    print("2. Message Security:")
    print("   ✓ Sign all messages to ensure integrity")
    print("   ✓ Encrypt sensitive data in message payloads")
    print("   ✓ Use nonces to prevent replay attacks")
    print("   ✓ Implement message expiration timestamps")
    print()

    print("3. Key Management:")
    print("   ✓ Generate strong cryptographic keys (2048+ bit RSA)")
    print("   ✓ Rotate keys regularly")
    print("   ✓ Store private keys securely (encrypted at rest)")
    print("   ✓ Use hardware security modules (HSMs) when available")
    print()

    print("4. Authentication and Authorization:")
    print("   ✓ Implement strong authentication mechanisms")
    print("   ✓ Use role-based access control")
    print("   ✓ Implement rate limiting and abuse prevention")
    print("   ✓ Log all security events for auditing")
    print()

    print("5. Network Security:")
    print("   ✓ Use secure network protocols (HTTPS, WSS)")
    print("   ✓ Implement proper firewall rules")
    print("   ✓ Monitor network traffic for anomalies")
    print("   ✓ Use VPNs for remote access")


async def main():
    """Main demonstration entry point."""
    await demonstrate_cryptographic_operations()
    await demonstrate_secure_client()
    await demonstrate_security_best_practices()

    print("\n=== Secure Communication Summary ===")
    print()
    print("This module demonstrated:")
    print("✓ Symmetric and asymmetric encryption")
    print("✓ Digital signatures and verification")
    print("✓ Certificate management and validation")
    print("✓ Secure message exchange patterns")
    print("✓ SSL/TLS configuration")
    print("✓ Security best practices for MCP")
    print()
    print("These patterns ensure that MCP communications are:")
    print("- Confidential (encrypted)")
    print("- Authentic (signed)")
    print("- Tamper-proof (integrity protected)")
    print("- Non-repudiable (digitally signed)")


if __name__ == "__main__":
    asyncio.run(main())
