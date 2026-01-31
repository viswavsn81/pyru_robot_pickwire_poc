#!/usr/bin/env python

import pygame
import time

def main():
    pygame.init()
    pygame.joystick.init()
    
    if pygame.joystick.get_count() == 0:
        print("❌ NO CONTROLLER FOUND!")
        return

    # Connect to the first controller
    joystick = pygame.joystick.Joystick(0)
    joystick.init()
    
    name = joystick.get_name()
    axes = joystick.get_numaxes()
    
    print(f"\n🎮 FOUND: {name}")
    print(f"ℹ️  It has {axes} axes (sticks/triggers).")
    print("\n---------------------------------------------------")
    print("👉 MOVE YOUR STICKS NOW!")
    print("   Identify which Axis Number changes when you move")
    print("   the stick you want to use for Joint 2.")
    print("---------------------------------------------------\n")
    
    # Dummy window to catch focus
    window = pygame.display.set_mode((300, 100))
    pygame.display.set_caption("INPUT SCANNER")

    try:
        while True:
            pygame.event.pump()
            
            # Build a string of all axis values
            status = "\r"
            for i in range(axes):
                val = joystick.get_axis(i)
                # Highlight active axes with stars
                if abs(val) > 0.1:
                    status += f"Ax{i}:[**{val:.2f}**]  "
                else:
                    status += f"Ax{i}: {val:.2f}   "
            
            print(status, end="")
            time.sleep(0.05)
            
    except KeyboardInterrupt:
        print("\n\n✅ Done.")

if __name__ == "__main__":
    main()
