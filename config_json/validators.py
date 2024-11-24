import re
import socket

class ValidationError(Exception):
    pass

def validate_required(value):
    """验证必填字段"""
    if value is None or str(value).strip() == "":
        raise ValidationError("此字段不能为空")
    return True

def validate_port_range(value):
    """验证端口范围"""
    try:
        port = int(value)
        if not (1 <= port <= 65535):
            raise ValidationError("端口必须在1-65535之间")
    except ValueError:
        raise ValidationError("端口必须是数字")
    return True

def validate_ip_or_hostname(value):
    """验证IP地址或主机名"""
    # 验证IP地址
    try:
        socket.inet_aton(value)
        return True
    except socket.error:
        pass
    
    # 验证主机名
    if not re.match(r'^[a-zA-Z0-9-]+(\.[a-zA-Z0-9-]+)*$', value):
        raise ValidationError("无效的主机名或IP地址")
    return True

# 验证器映射
VALIDATORS = {
    "required": validate_required,
    "port_range": validate_port_range,
    "ip_or_hostname": validate_ip_or_hostname
} 