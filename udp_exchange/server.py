import asyncio
from config import NetworkConfig

class Server:
    def __init__(self):
        self.config = NetworkConfig()
        self.transport = None

    async def start(self):
        loop = asyncio.get_event_loop()
        if self.config.USE_UDP:
            self.transport, _ = await loop.create_datagram_endpoint(
                lambda: ServerProtocol(self),
                local_addr=(self.config.SERVER_HOST, self.config.SERVER_PORT)
            )
        else:
            server = await loop.create_server(
                lambda: ServerProtocol(self),
                self.config.SERVER_HOST,
                self.config.SERVER_PORT
            )
            await server.start_serving()

class ServerProtocol(asyncio.Protocol if not NetworkConfig.USE_UDP else asyncio.DatagramProtocol):
    def __init__(self, server):
        self.server = server

    def process_data(self, data):
        # Process the received data here
        return f"Processed: {data.decode()}"

    def datagram_received(self, data, addr):
        response = self.process_data(data)
        self.server.transport.sendto(response.encode(), addr)

    def data_received(self, data):
        response = self.process_data(data)
        self.transport.write(response.encode())

async def main():
    server = Server()
    await server.start()
    try:
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    asyncio.run(main())