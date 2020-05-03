import numpy as np
from multiagent.core import World, Agent, Landmark, Wall
from multiagent.scenario import BaseScenario

class DummyAgent(Entity):
	#super class of red and grey agent that use pre-defined policies
    def __init__(self):
        super(DummyAgent, self).__init__()
        self.collide = True
        self.movable = False
        self.size = 0.05
        self.is_red = None

    def response_to_audio(self, audio):
    	#this method takes in an audio action by the blue agent
    	#output a probablistic response according to the agent type
    	raise NotImplementedError

class RedAgent(DummyAgent):
    def __init__(self):
        super(RedAgent, self).__init__()
        self.color = np.array([1.0, 0.0, 0.0])
        self.is_red = True

class GreyAgent(DummyAgent):
    def __init__(self):
        super(GreyAgent, self).__init__()
        self.color = np.array([0.2, 0.2, 0.2])
        self.is_red = False

class Room_cell(object):
	def __init__(self):
		#center of the room_cell: 2d np array
		self.center = None
		raise NotImplementedError

	def has_agent(self):
		#return if cell has agent occupied
		raise NotImplementedError

class Room_window(object):
	def __init__(self):
		#list of two np arrays contain the two end_points of the window
		self.end_points = None
		raise NotImplementedError
	def intersect(self, line: list) -> bool:
		#return if intersect with a list of two np arrays specifying a line
		raise NotImplementedError

class Room(object):
	def __init__(self):
		self.cells = []
		raise NotImplementedError
	def has_agent(self) -> bool:
		raise NotImplementedError
	def agent_location(self):
		#return the cell location, which occupied by an agent
		raise NotImplementedError


class BlueAgent(Agent):
    def __init__(self):
        super(BlueAgent, self).__init__()
        self.color = np.array([0.0, 0.0, 1.0])

class Scenario(BaseScenario):
	def make_world(self):
		self.num_blue = 5
		self.num_grey = 3
		#self.agents contains only policy agents (blue agents)
		self.agents = [BlueAgent() for i in range(self.num_blue)]
		self.dummy_agents = []
		self.walls = []
		self.rooms = []
		raise NotImplementedError
	
	def _reset_dummy_agents_location(self):
		raise NotImplementedError

	def _permute_dummy_agents_index(self):
		raise NotImplementedError

	def _set_walls(self):
		raise NotImplementedError

	def reset_world(self, world):
		self._reset_dummy_agents_location()
		self._permute_dummy_agents_index()
		raise NotImplementedError

	def benchmark_data(self, agent, world):
        # returns data for benchmarking purposes
        return reward(agent, world)

    def reward(self, agent, world):

	def observation(self, agent, world):