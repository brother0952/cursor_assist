from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad

class Crypto:
    def __init__(self, key=None):
        self.enabled = key is not None
        if self.enabled:
            self.key = key.ljust(16)[:16]
            self.block_size = 16
        
    def encrypt(self, data):
        if not self.enabled:
            return data
        cipher = AES.new(self.key, AES.MODE_CBC)
        encrypted_data = cipher.encrypt(pad(data, self.block_size))
        return encrypted_data
    
    def decrypt(self, data):
        if not self.enabled:
            return data
        cipher = AES.new(self.key, AES.MODE_CBC)
        decrypted_data = unpad(cipher.decrypt(data), self.block_size)
        return decrypted_data 