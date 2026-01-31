import pygame
import sys

pygame.init()
pygame.joystick.init()

if pygame.joystick.get_count() == 0:
    print("❌ No controller found.")
    sys.exit()

joystick = pygame.joystick.Joystick(0)
joystick.init()

print(f"🎮 Controller: {joystick.get_name()}")
print("Press any button to see its ID. Press Ctrl+C to exit.")

try:
    while True:
        for event in pygame.event.get():
            if event.type == pygame.JOYBUTTONDOWN:
                print(f"🔘 Button Pressed: ID {event.button}")
            elif event.type == pygame.JOYAXISMOTION:
                if abs(event.value) > 0.5:
                   print(f"🕹️ Axis {event.axis} moved: {event.value:.2f}")
except KeyboardInterrupt:
    print("\nExiting.")
