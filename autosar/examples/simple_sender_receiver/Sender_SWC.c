#include "Rte_Sender.h"

// 发送者软件组件实现
void Sender_Run(void) {
    // 创建要发送的数据
    VehicleData_Type vehicleData;
    vehicleData.speed = 60;
    vehicleData.rpm = 2500;
    
    // 通过RTE发送数据
    Rte_Write_VehicleData(&vehicleData);
} 