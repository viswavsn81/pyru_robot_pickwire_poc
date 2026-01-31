import pygame
pygame.init()
pygame.joystick.init()
j = pygame.joystick.Joystick(0)
j.init()

print(f"Controller: {j.get_name()}")
print("Press buttons to see their IDs. Press Ctrl+C to stop.")

try:
    while True:
        pygame.event.pump()
        for i in range(j.get_numbuttons()):
            if j.get_button(i):
                print(f"Button {i} is pressed!")
except KeyboardInterrupt:
    pass
