#ifndef RTE_TYPE_H
#define RTE_TYPE_H

// 定义数据类型
typedef unsigned char uint8;
typedef unsigned short uint16;

// 定义发送/接收的数据结构
typedef struct {
    uint8 speed;
    uint16 rpm;
} VehicleData_Type;

#endif 