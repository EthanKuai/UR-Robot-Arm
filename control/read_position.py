import os
import rtde_control
import rtde_receive


def main():
    ## Read position once
    UR_IP = os.environ.get("UR_IP", "192.168.1.20")

    rtde_r = rtde_receive.RTDEReceiveInterface(UR_IP)

    tcp = rtde_r.getActualTCPPose()          # [x, y, z, rx, ry, rz] meters/radians
    q   = rtde_r.getActualQ()                # joint angles [q1..q6] radians

    print("TCP pose:", tcp)
    print("Joints:  ", q)

    ## Continuous streaming (500 Hz)
    rtde_r = rtde_receive.RTDEReceiveInterface(UR_IP)

    for i in range(10):
        tcp = rtde_r.getActualTCPPose()
        print("TCP:", tcp)

    ## Motion (optional, via `rtde_control`)
    # Note: `rtde_control` and `rtde_receive` must be **separate connections** (two objects, two sockets).
    import time

    rtde_rec = rtde_receive.RTDEReceiveInterface(UR_IP)
    rtde_c = rtde_control.RTDEControlInterface(UR_IP)

    current = rtde_rec.getActualTCPPose()
    current[0] += 0.01            # +10 mm along X (meters)
    rtde_c.moveL(current, speed=0.1, acceleration=0.1)

    """
    rtde_c = rtde_control.RTDEControlInterface("192.168.1.20")
    rtde_r = rtde_receive.RTDEReceiveInterface("192.168.1.20")

    rtde_c.moveJ([0, -1.57, 0, -1.57, 0, 0], 1.0, 0.5)   # in joint space
    print("New pose:", rtde_r.getActualTCPPose())
    """


if __name__ == "__main__":
    main()
