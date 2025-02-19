import asyncio
import logging
from config import NetworkConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Client:
    def __init__(self):
        self.config = NetworkConfig()
        self.transport = None
        self.protocol = None

    async def send_data(self, message: str):
        try:
            if self.config.USE_UDP:
                self.transport.sendto(message.encode(), 
                                    (self.config.RELAY_HOST, self.config.RELAY_CLIENT_PORT))
            else:
                self.transport.write(message.encode())
            logger.info(f"Sent message: {message}")
        except Exception as e:
            logger.error(f"Error sending data: {e}")

    async def start(self):
        loop = asyncio.get_event_loop()
        try:
            if self.config.USE_UDP:
                # 移除 remote_addr，让 UDP 能接收任何地址的数据
                self.transport, self.protocol = await loop.create_datagram_endpoint(
                    lambda: ClientProtocol(self),
                    local_addr=(self.config.CLIENT_HOST, self.config.CLIENT_PORT)
                )
            else:
                self.transport, self.protocol = await loop.create_connection(
                    lambda: ClientTCPProtocol(self),
                    self.config.RELAY_HOST,
                    self.config.RELAY_CLIENT_PORT
                )
            logger.info("Client started successfully")
        except Exception as e:
            logger.error(f"Failed to start client: {e}")
            raise

class ClientProtocol(asyncio.DatagramProtocol):
    def __init__(self, client):
        self.client = client
        self.transport = None

    def connection_made(self, transport):
        self.transport = transport
        logger.info("UDP connection established")

    def datagram_received(self, data, addr):
        message = data.decode()
        logger.info(f"Received from relay {addr}: {message}")

    def error_received(self, exc):
        logger.error(f"UDP error: {exc}")

class ClientTCPProtocol(asyncio.Protocol):
    def __init__(self, client):
        self.client = client
        self.transport = None

    def connection_made(self, transport):
        self.transport = transport
        logger.info("TCP connection established")

    def data_received(self, data):
        message = data.decode()
        logger.info(f"Received from relay: {message}")

    def connection_lost(self, exc):
        logger.info("Connection closed")
        if exc:
            logger.error(f"Connection lost due to error: {exc}")

async def get_input():
    while True:
        message = await asyncio.get_event_loop().run_in_executor(
            None, input, "Enter message to send (or 'quit' to exit): "
        )
        if message.lower() == 'quit':
            return
        yield message

async def main():
    client = Client()
    await client.start()
    try:
        async for message in get_input():
            await client.send_data(message)
            # 添加短暂延迟，让接收消息有机会处理
            await asyncio.sleep(0.1)
    except KeyboardInterrupt:
        logger.info("Client shutting down...")
    finally:
        if client.transport:
            client.transport.close()

if __name__ == "__main__":
    asyncio.run(main())