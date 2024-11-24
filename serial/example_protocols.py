from protocol_handler import ProtocolHandler, ProtocolMessage
import json
import struct
from typing import Any, Optional

class SimpleTextProtocol(ProtocolHandler):
    """简单文本协议处理器"""
    
    def parse_message(self, data: bytes) -> Optional[ProtocolMessage]:
        try:
            decoded = data.decode().strip()
            return ProtocolMessage(
                raw_data=data,
                decoded_data=decoded,
                message_type="text"
            )
        except UnicodeDecodeError:
            return None
    
    def build_message(self, message: str) -> bytes:
        return f"{message}\n".encode()

class JsonProtocol(ProtocolHandler):
    """JSON协议处理器"""
    
    def parse_message(self, data: bytes) -> Optional[ProtocolMessage]:
        try:
            decoded = data.decode().strip()
            json_data = json.loads(decoded)
            return ProtocolMessage(
                raw_data=data,
                decoded_data=json_data,
                message_type="json"
            )
        except (UnicodeDecodeError, json.JSONDecodeError):
            return None
    
    def build_message(self, message: dict) -> bytes:
        return (json.dumps(message) + "\n").encode()

class ModbusProtocol(ProtocolHandler):
    """Modbus协议处理器示例"""
    
    def parse_message(self, data: bytes) -> Optional[ProtocolMessage]:
        if len(data) < 4:  # 最小长度检查
            return None
            
        try:
            # 示例：假设前两个字节是地址，后两个字节是数据
            address, value = struct.unpack(">HH", data[:4])
            decoded_data = {
                "address": address,
                "value": value
            }
            return ProtocolMessage(
                raw_data=data,
                decoded_data=decoded_data,
                message_type="modbus"
            )
        except struct.error:
            return None
    
    def build_message(self, message: dict) -> bytes:
        # 示例：打包地址和数据
        return struct.pack(">HH", 
                         message.get("address", 0),
                         message.get("value", 0)) 