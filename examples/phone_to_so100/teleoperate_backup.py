#!/usr/bin/env python

import time
import numpy as np
import torch
import pygame
from dataclasses import dataclass, field
from pathlib import Path

# --- IMPORTS ---
from lerobot.model.kinematics import RobotKinematics
from lerobot.processor import RobotAction, RobotObservation, ProcessorStep
from lerobot.robots.so_follower import SO100Follower, SO100FollowerConfig
# Note: We import the steps but NOT the Pipeline class
from lerobot.robots.so_follower.robot_kinematic_processor import (
    EEBoundsAndSafety,
    EEReferenceAndDelta,
    GripperVelocityToJoint,
    InverseKinematicsEEToJoints,
)
from lerobot.teleoperators.gamepad.teleop_gamepad import GamepadTeleop
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data

FPS = 30

@dataclass
class GamepadTeleopConfig:
    type: str = "Xbox Wireless Controller"
    id: str = "xbox_controller"  
    streaming_port: int = 9999
    max_speed: float = 1.0
    fps: int = 30
    calibration_dir: Path = None
    robot_type: str = "so100"
    features: dict = field(default_factory=dict)
    use_gripper: bool = True
    deadzone: float = 0.05
    axis_mapping: dict = field(default_factory=dict)
    
    def __post_init__(self):
        if self.calibration_dir is None:
            self.calibration_dir = Path(".cache/calibration")

# --- CUSTOM MAPPER ---
class CustomGamepadMapper(ProcessorStep):
    def __init__(self):
        super().__init__()
        self.last_gripper_val = 0.0

    def process(self, obs_tuple):
        return self.__call__(obs_tuple)

    def __call__(self, obs_tuple):
        # 1. Unpack
        if isinstance(obs_tuple, (tuple, list)):
            gamepad_obs = obs_tuple[0]
            robot_obs = obs_tuple[1]
        else:
            gamepad_obs = obs_tuple
            robot_obs = {}
            
        raw_inputs = gamepad_obs.get("gamepad.raw_inputs", {})
        
        # 2. Map Inputs to 7-DoF Array [X, Y, Z, Rx, Ry, Rz, Gripper]
        action_array = np.zeros(7, dtype=np.float32)
        
        # Sticks (Adjust signs/axes here if robot moves backwards/sideways)
        action_array[0] = -raw_inputs.get("axis1", 0.0) # X (Fwd/Back)
        action_array[1] = -raw_inputs.get("axis0", 0.0) # Y (Left/Right)
        action_array[2] = -raw_inputs.get("axis4", 0.0) # Z (Up/Down)
        
        # Gripper Buttons
        if raw_inputs.get("btn0", 0): 
            self.last_gripper_val = -1.0 # Close
        elif raw_inputs.get("btn1", 0): 
            self.last_gripper_val = 1.0  # Open
        action_array[6] = self.last_gripper_val
        
        # 3. Pack into RobotAction
        action_tensor = torch.from_numpy(action_array)
        out = RobotAction(action=action_tensor)
        
        # 4. Merge Robot State (Crucial for next steps)
        if isinstance(robot_obs, (dict, RobotObservation)):
            out.update(robot_obs)
            
        return out

def main():
    # 1. Initialize Robot
    robot_config = SO100FollowerConfig(
        port="/dev/ttyACM0", id="my_awesome_follower_arm", use_degrees=True
    )
    robot = SO100Follower(robot_config)

    # 2. Initialize Xbox
    teleop_config = GamepadTeleopConfig(type="Xbox Wireless Controller") 
    teleop_device = GamepadTeleop(teleop_config)

    # 3. Setup Kinematics
    kinematics_solver = RobotKinematics(
        urdf_path="SO-ARM100/Simulation/SO100/so100.urdf",
        target_frame_name="jaw",
        joint_names=list(robot.bus.motors.keys()),
    )

    # 4. INSTANTIATE STEPS MANUALLY (Bypassing Pipeline Wrapper)
    step_mapper = CustomGamepadMapper()
    
    step_ee_ref = EEReferenceAndDelta(
        kinematics=kinematics_solver,
        end_effector_step_sizes={"x": 0.2, "y": 0.2, "z": 0.2},
        motor_names=list(robot.bus.motors.keys()),
        use_latched_reference=True,
    )
    
    step_safety = EEBoundsAndSafety(
        end_effector_bounds={"min": [-1.0, -1.0, -1.0], "max": [1.0, 1.0, 1.0]},
        max_ee_step_m=0.10,
    )
    
    step_gripper = GripperVelocityToJoint(
        speed_factor=20.0,
    )
    
    step_ik = InverseKinematicsEEToJoints(
        kinematics=kinematics_solver,
        motor_names=list(robot.bus.motors.keys()),
        initial_guess_current_joints=True,
    )

    # 5. Connect
    print("🔌 Connecting to Robot...")
    robot.connect()
    print("✅ Robot Connected!")
    
    print("🎮 Connecting to Xbox Controller...")
    teleop_device.connect()
    print("✅ Controller Connected!")

    # 6. Pygame Window (Click this!)
    pygame.display.init()
    window = pygame.display.set_mode((400, 400))
    pygame.display.set_caption("CLICK HERE TO CONTROL ROBOT")
    window.fill((0, 100, 200)) 
    pygame.display.flip()

    init_rerun(session_name="xbox_so100_teleop")

    if not robot.is_connected or not teleop_device.is_connected:
        raise ValueError("Robot or Controller is not connected!")

    print("\n🚀 TELEOP STARTED!")
    print("------------------")
    print("👉 STEP 1: Click the BLUE window.")
    print("👉 STEP 2: Move sticks!")
    print("------------------\n")

    # 7. MAIN LOOP (Manual Execution)
    while True:
        t0 = time.perf_counter()
        pygame.event.pump()
        
        # A. Get Raw Data
        robot_obs = robot.get_observation()
        gamepad_obs = teleop_device.get_action()
        
        # B. Run Steps Sequentially
        #    Pass the result of one into the next.
        try:
            state = step_mapper((gamepad_obs, robot_obs))
            state = step_ee_ref(state)
            state = step_safety(state)
            state = step_gripper(state)
            state = step_ik(state)
            
            # C. Extract Final Action
            # If IK returns a dict, grab 'action'. If it returns a Tensor, use it directly.
            if isinstance(state, (dict, RobotAction)):
                joint_action = state['action']
            else:
                joint_action = state
            
            # D. Send to Robot
            _ = robot.send_action(joint_action)
            
            # E. Log
            log_rerun_data(observation=gamepad_obs, action=joint_action)
            
        except Exception as e:
            print(f"Loop Error: {e}")
            # Optional: break or continue depending on severity
            
        precise_sleep(max(1.0 / FPS - (time.perf_counter() - t0), 0.0))

if __name__ == "__main__":
    main()
