from ucryptolib import aes
import ubinascii

class Crypto:
    def __init__(self, key):
        # 使用固定的16字节密钥
        self.key = key.ljust(16)[:16]
        self.block_size = 16
        
    def encrypt(self, data):
        # 确保数据长度是16的倍数
        pad_len = self.block_size - (len(data) % self.block_size)
        data = data + bytes([pad_len] * pad_len)
        
        cipher = aes(self.key, 1)  # MODE_CBC = 1
        return cipher.encrypt(data)
    
    def decrypt(self, data):
        cipher = aes(self.key, 1)  # MODE_CBC = 1
        decrypted = cipher.decrypt(data)
        
        # 移除填充
        pad_len = decrypted[-1]
        return decrypted[:-pad_len] 