from ego_vehicle import CollisionSensor
import logging

class CollisionChecker(object):

    def __init__(self, collision_sensor):

        self.collision_sensor = collision_sensor
        self.first_iter = True
        # The parameters used for the case we want to detect collisions
        self._thresh_other = 400
        self._thresh_vehicle = 400
        self._thresh_pedestrian = 300
        self._previous_pedestrian_collision = 0
        self._previous_vehicle_collision = 0
        self._previous_other_collision = 0

        self._collision_time = -1
        self._count_collisions = 0

    def test_collision(self):
        """
            test if there is any instant collision.

        """
        collided = False
        try:
            history = self.collision_sensor.get_collision_history()
        except Exception as e:
            logging.critical(e, e.args)
            # traceback.print_exc(file=sys.stdout)

        if(len(history)>0):
            collided = True
        # if (player_measurements.collision_vehicles - self._previous_vehicle_collision) \
        #         > self._thresh_vehicle:
        #     collided = True
        # if (player_measurements.collision_pedestrians - self._previous_pedestrian_collision) \
        #         > self._thresh_pedestrian:
        #     collided = True
        # if (player_measurements.collision_other - self._previous_other_collision) \
        #         > self._thresh_other:
        #     collided = True

        # self._previous_pedestrian_collision = .collision_pedestrians
        # self._previous_vehicle_collision = player_measurements.collision_vehicles
        # self._previous_other_collision = player_measurements.collision_other

        return collided




