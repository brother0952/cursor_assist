#ifndef RTE_SENDER_H
#define RTE_SENDER_H

#include "Rte_Type.h"

// RTE 状态定义
#define RTE_E_OK 0
#define RTE_E_NOK 1

// RTE API 声明
Std_ReturnType Rte_Write_VehicleData(const VehicleData_Type* data);

#endif 