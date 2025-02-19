from dataclasses import dataclass

@dataclass
class NetworkConfig:
    USE_UDP: bool = True  # True for UDP, False for TCP
    CLIENT_HOST: str = "127.0.0.1"
    CLIENT_PORT: int = 5000
    RELAY_HOST: str = "127.0.0.1"
    RELAY_CLIENT_PORT: int = 5001
    RELAY_SERVER_PORT: int = 5002
    SERVER_HOST: str = "127.0.0.1"
    SERVER_PORT: int = 5003