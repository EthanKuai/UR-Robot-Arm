# Universal Robots (UR) Control

## Hardware

Since the UR arm is mounted on a [MiR](MiR.md), need to know how to control that first.

## Protocols
> https://www.universal-robots.com/developer/

|           Protocol           |  Language   |
| :--------------------------: | :---------: |
|    [Modbus](ModbusTCP.md)    |   Python    |
| [UR Official ROS](UR-ROS.md) |     C++     |
|       [RTDE](RTDE.md)        | Python, C++ |

### General TCP

A good AI generated ASCII diagram to show the open ports (`ping 192.168.1.20`):

```
┌──────────────────┐         Ethernet           ┌──────────────────────────┐
│   Your PC        │◄──────────────────────────►│  UR Control Box          │
│  192.168.x.x     │    1 Gb/s (e-Series)       │  192.168.1.20            │
│                  │                            │ (Change via MiR Control) │
│                  │                            │                          │
│  Client apps:    │                            │  Server ports:           │
│  • urx library   │                            │  29999  Dashboard (text) │
│  • RTDE client   │                            │  30001  Primary (binary) │
│  • pymodbus      │                            │  30002  Secondary        │
│  • ROS 2 driver  │                            │  30003  Real-time 500Hz  │
│  • raw sockets   │                            │  30004  RTDE     500Hz   │
└──────────────────┘                            │  30011-13  Read-only     │
                                                │  502    Modbus TCP       │
                                                │  44818  EtherNet/IP      │
                                                │  34964  PROFINET         │
                                                │  4840   OPC-UA (URCap)   │
                                                └────────────┬─────────────┘
                                                             │ internal bus
                                                             ▼
                                                     ┌────────────────┐
                                                     │  UR Robot Arm  │
                                                     │ (6-axis cobot) │
                                                     └────────────────┘
```

### I have demonstrated

- [x] read/write over RTDE
- [x] connect over ROS
