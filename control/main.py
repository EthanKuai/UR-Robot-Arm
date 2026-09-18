import importlib
import os


def main():
    ur_ip = os.environ.get("UR_IP", "192.168.1.20")
    print("Hello from control!")
    print("UR_IP:", ur_ip)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    other_scripts = sorted(
        f for f in os.listdir(script_dir)
        if f.endswith(".py") and f != os.path.basename(__file__)
    )

    for i, f in enumerate(other_scripts):
        print(f"{i}: {f}")

    choice = int(input("Pick a number to run: "))
    module_name = other_scripts[choice][:-3]
    module = importlib.import_module(module_name)
    module.main()


if __name__ == "__main__":
    main()
