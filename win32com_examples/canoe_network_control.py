import win32com.client
import time

class CANoeNetworkControl:
    def __init__(self):
        self.application = None
        self.measurement = None
        self.network = None
        self.simulation = None
    
    def connect(self):
        """连接到CANoe并初始化网络控制"""
        try:
            self.application = win32com.client.Dispatch("CANoe.Application")
            self.measurement = self.application.Measurement
            self.network = self.application.Network
            self.simulation = self.application.Simulation
            print("成功连接到CANoe")
            return True
        except Exception as e:
            print(f"连接CANoe失败: {str(e)}")
            return False
    
    def get_node_status(self, node_name):
        """获取节点状态"""
        try:
            node = self.network.Nodes.Item(node_name)
            return {
                'name': node.Name,
                'active': node.Active,
                'simulated': node.Simulated,
                'network': node.Network
            }
        except Exception as e:
            print(f"获取节点状态失败: {str(e)}")
            return None
    
    def set_node_simulation(self, node_name, enable):
        """设置节点仿真状态"""
        try:
            node = self.network.Nodes.Item(node_name)
            node.Simulated = enable
            print(f"节点 {node_name} 仿真状态已设置为: {'启用' if enable else '禁用'}")
            return True
        except Exception as e:
            print(f"设置节点仿真状态失败: {str(e)}")
            return False
    
    def get_network_info(self):
        """获取网络信息"""
        try:
            networks = []
            for i in range(self.network.Count):
                network = self.network.Item(i)
                networks.append({
                    'name': network.Name,
                    'type': network.NetworkType,
                    'channel_count': network.Channels.Count
                })
            return networks
        except Exception as e:
            print(f"获取网络信息失败: {str(e)}")
            return None
    
    def get_channel_info(self, network_name):
        """获取通道信息"""
        try:
            network = self.network.Item(network_name)
            channels = []
            for i in range(network.Channels.Count):
                channel = network.Channels.Item(i)
                channels.append({
                    'number': channel.Number,
                    'name': channel.Name,
                    'hardware': channel.Hardware
                })
            return channels
        except Exception as e:
            print(f"获取通道信息失败: {str(e)}")
            return None
    
    def start_simulation_setup(self, setup_name):
        """启动指定的仿真设置"""
        try:
            setup = self.simulation.SimulationSetup.Item(setup_name)
            setup.Start()
            print(f"仿真设置 {setup_name} 已启动")
            return True
        except Exception as e:
            print(f"启动仿真设置失败: {str(e)}")
            return False
    
    def stop_simulation_setup(self, setup_name):
        """停止指定的仿真设置"""
        try:
            setup = self.simulation.SimulationSetup.Item(setup_name)
            setup.Stop()
            print(f"仿真设置 {setup_name} 已停止")
            return True
        except Exception as e:
            print(f"停止仿真设置失败: {str(e)}")
            return False
    
    def set_system_variable(self, namespace, variable, value):
        """设置系统变量"""
        try:
            var = self.application.System.Namespaces.Item(namespace).Variables.Item(variable)
            var.Value = value
            print(f"系统变量 {namespace}.{variable} 已设置为: {value}")
            return True
        except Exception as e:
            print(f"设置系统变量失败: {str(e)}")
            return False
    
    def control_network_node(self, node_name, command):
        """控制网络节点（启动/停止/重启）"""
        try:
            node = self.network.Nodes.Item(node_name)
            if command.lower() == 'start':
                node.Start()
                print(f"节点 {node_name} 已启动")
            elif command.lower() == 'stop':
                node.Stop()
                print(f"节点 {node_name} 已停止")
            elif command.lower() == 'restart':
                node.Stop()
                time.sleep(1)
                node.Start()
                print(f"节点 {node_name} 已重启")
            return True
        except Exception as e:
            print(f"控制网络节点失败: {str(e)}")
            return False

def main():
    # 创建示例
    canoe = CANoeNetworkControl()
    
    # 连接到CANoe
    if not canoe.connect():
        return
    
    try:
        # 示例1：获取网络信息
        print("\n示例1：获取网络信息")
        networks = canoe.get_network_info()
        if networks:
            for network in networks:
                print(f"网络名称: {network['name']}")
                print(f"网络类型: {network['type']}")
                print(f"通道数量: {network['channel_count']}")
                print("---")
        
        # 示例2：获取节点状态
        print("\n示例2：获取节点状态")
        node_name = "ECU1"
        status = canoe.get_node_status(node_name)
        if status:
            print(f"节点名称: {status['name']}")
            print(f"是否激活: {status['active']}")
            print(f"是否仿真: {status['simulated']}")
        
        # 示例3：控制节点仿真
        print("\n示例3：控制节点仿真")
        canoe.set_node_simulation(node_name, True)
        
        # 示例4：启动仿真设置
        print("\n示例4：启动仿真设置")
        setup_name = "DefaultSetup"
        canoe.start_simulation_setup(setup_name)
        
        # 等待一段时间
        time.sleep(5)
        
        # 示例5：控制网络节点
        print("\n示例5：控制网络节点")
        canoe.control_network_node(node_name, "restart")
        
        # 停止仿真
        canoe.stop_simulation_setup(setup_name)
        
    except Exception as e:
        print(f"执行过程中出错: {str(e)}")

if __name__ == "__main__":
    main() 