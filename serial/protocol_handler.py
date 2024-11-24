from abc import ABC, abstractmethod
from typing import Any, Optional
from dataclasses import dataclass

@dataclass
class ProtocolMessage:
    """协议消息基类，可以根据需要扩展"""
    raw_data: bytes
    decoded_data: Any = None
    message_type: str = ""
    
class ProtocolHandler(ABC):
    """协议处理器基类"""
    
    @abstractmethod
    def parse_message(self, data: bytes) -> Optional[ProtocolMessage]:
        """
        解析接收到的数据
        :param data: 原始字节数据
        :return: 解析后的消息对象，解析失败返回None
        """
        pass
    
    @abstractmethod
    def build_message(self, message: Any) -> bytes:
        """
        构建要发送的消息
        :param message: 要发送的消息内容
        :return: 编码后的字节数据
        """
        pass
    
    def validate_checksum(self, data: bytes) -> bool:
        """
        校验数据
        :param data: 要校验的数据
        :return: 校验是否通过
        """
        return True 