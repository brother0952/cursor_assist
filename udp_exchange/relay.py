import asyncio
import logging
from config import NetworkConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Relay:
    def __init__(self):
        self.config = NetworkConfig()
        self.client_transport = None
        self.server_transport = None
        self.client_protocol = None
        self.server_protocol = None
        self.clients = {}

    async def start(self):
        loop = asyncio.get_event_loop()
        
        if self.config.USE_UDP:
            # Client side of relay
            self.client_transport, self.client_protocol = await loop.create_datagram_endpoint(
                lambda: RelayClientProtocol(self),
                local_addr=(self.config.RELAY_HOST, self.config.RELAY_CLIENT_PORT)
            )
            
            # Server side of relay
            self.server_transport, self.server_protocol = await loop.create_datagram_endpoint(
                lambda: RelayServerProtocol(self),
                remote_addr=(self.config.SERVER_HOST, self.config.SERVER_PORT)
            )
        else:
            server = await loop.create_server(
                lambda: RelayClientProtocol(self),
                self.config.RELAY_HOST,
                self.config.RELAY_CLIENT_PORT
            )
            await server.start_serving()

    def forward_to_server(self, data, addr=None):
        logger.info(f"Forwarding to server: {data}")
        if self.config.USE_UDP:
            self.server_transport.sendto(data, 
                                       (self.config.SERVER_HOST, self.config.SERVER_PORT))
        else:
            if self.server_transport:
                self.server_transport.write(data)

    def forward_to_client(self, data, addr):
        logger.info(f"Forwarding to client: {data}")
        if self.config.USE_UDP:
            self.client_transport.sendto(data, addr)
        else:
            if addr in self.clients:
                self.clients[addr].write(data)

class RelayClientProtocol(asyncio.Protocol if not NetworkConfig.USE_UDP else asyncio.DatagramProtocol):
    def __init__(self, relay):
        self.relay = relay

    def datagram_received(self, data, addr):
        self.relay.forward_to_server(data, addr)
        self.relay.clients[addr] = self

    def data_received(self, data):
        self.relay.forward_to_server(data)

class RelayServerProtocol(asyncio.Protocol if not NetworkConfig.USE_UDP else asyncio.DatagramProtocol):
    def __init__(self, relay):
        self.relay = relay

    def datagram_received(self, data, addr):
        for client_addr in self.relay.clients:
            self.relay.forward_to_client(data, client_addr)

    def data_received(self, data):
        for client_addr in self.relay.clients:
            self.relay.forward_to_client(data, client_addr)

async def main():
    relay = Relay()
    await relay.start()
    try:
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    asyncio.run(main())