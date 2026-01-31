from lerobot.motors.dynamixel.dynamixel import DynamixelMotorsBus
import logging

# Configure logging to see debug output
logging.basicConfig(level=logging.INFO)

def scan():
    port = "/dev/ttyACM0"
    print(f"Scanning port {port} for Dynamixel motors...")
    
    try:
        # DynamixelMotorsBus.scan_port is a class method
        results = DynamixelMotorsBus.scan_port(port)
        print("Scan Results:")
        print(results)
    except Exception as e:
        print(f"Scan failed: {e}")

if __name__ == "__main__":
    scan()
