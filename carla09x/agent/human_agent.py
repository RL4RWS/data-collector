# from carla09x.agent.agent import Agent
# from carla09x.client import VehicleControl

try:
    import sys
    import glob
    import os
    sys.path.append(glob.glob('/home/jhshin/Sources/carla/PythonAPI/carla/dist/carla-0.9.9-py3.7-linux-x86_64.egg')[0])
    import carla
except IndexError:
    print('unable to import carla library')
    pass

try:
    import pygame
    from pygame.locals import K_DOWN
    from pygame.locals import K_LEFT
    from pygame.locals import K_RIGHT
    from pygame.locals import K_SPACE
    from pygame.locals import K_UP
    from pygame.locals import K_a
    from pygame.locals import K_d
    from pygame.locals import K_q
    from pygame.locals import K_s
    from pygame.locals import K_w

except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')


class HumanAgent:
    """
    Derivation of Agent Class for human control,

    """

    def __init__(self):
        """
         TODO: add the parameter for a joystick to be used, default keyboard.
        """
        # super(HumanAgent).__init__()
        # pygame.init()
        self._is_on_reverse = False

    def _get_keyboard_control(self, keys):
        """
        Return a VehicleControl message based on the pressed keys.

        Return None
        if a new episode was requested.
        """

        control = carla.VehicleControl()
        if keys[K_LEFT] or keys[K_a]:
            control.steer = -1.0
        if keys[K_RIGHT] or keys[K_d]:
            control.steer = 1.0
        if keys[K_UP] or keys[K_w]:
            control.throttle = 1.0
        if keys[K_DOWN] or keys[K_s]:
            control.brake = 1.0
        if keys[K_SPACE]:
            control.hand_brake = True
        if keys[K_q]:
            self._is_on_reverse = not self._is_on_reverse
        control.reverse = self._is_on_reverse
        return control

    def run_step(self):
        # We basically ignore all the parameters.
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return carla.VehicleControl()
        return self._get_keyboard_control(pygame.key.get_pressed())
