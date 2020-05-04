import numpy as np
from multiagent.core import World, Agent, Landmark, Wall
from multiagent.scenario import BaseScenario

class DummyAgent(Landmark):
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

class Point:
	def __init__(self, xy):
		self.x = xy[0]
		self.y = xy[1]

class Room_window(object):
	def __init__(self):
		#list of two np arrays contain the two end_points of the window
		self.end_points = None
		raise NotImplementedError

	# https: // www.geeksforgeeks.org / check - if -two - given - line - segments - intersect /
	def onSegment(self, p, q, r):
		if ((q.x <= max(p.x, r.x)) and (q.x >= min(p.x, r.x)) and
				(q.y <= max(p.y, r.y)) and (q.y >= min(p.y, r.y))):
			return True
		return False

	def orientation(self, p, q, r):
		# to find the orientation of an ordered triplet (p,q,r)
		# function returns the following values:
		# 0 : Colinear points
		# 1 : Clockwise points
		# 2 : Counterclockwise

		# See https://www.geeksforgeeks.org/orientation-3-ordered-points/amp/
		# for details of below formula.

		val = (float(q.y - p.y) * (r.x - q.x)) - (float(q.x - p.x) * (r.y - q.y))
		if (val > 0):
			# Clockwise orientation
			return 1
		elif (val < 0):
			# Counterclockwise orientation
			return 2
		else:
			# Colinear orientation
			return 0

	# The main function that returns true if
	# the line segment 'p1q1' and 'p2q2' intersect.
	def doIntersect(self, _p1, _q1, _p2, _q2):
		p1 = Point(_p1)
		q1 = Point(_q1)
		p2 = Point(_p2)
		q2 = Point(_q2)
		# Find the 4 orientations required for
		# the general and special cases
		o1 = self.orientation(p1, q1, p2)
		o2 = self.orientation(p1, q1, q2)
		o3 = self.orientation(p2, q2, p1)
		o4 = self.orientation(p2, q2, q1)

		# General case
		if ((o1 != o2) and (o3 != o4)):
			return True

		# Special Cases
		# p1 , q1 and p2 are colinear and p2 lies on segment p1q1
		if ((o1 == 0) and self.onSegment(p1, p2, q1)):
			return True
		# p1 , q1 and q2 are colinear and q2 lies on segment p1q1
		if ((o2 == 0) and self.onSegment(p1, q2, q1)):
			return True
		# p2 , q2 and p1 are colinear and p1 lies on segment p2q2
		if ((o3 == 0) and self.onSegment(p2, p1, q2)):
			return True
		# p2 , q2 and q1 are colinear and q1 lies on segment p2q2
		if ((o4 == 0) and self.onSegment(p2, q1, q2)):
			return True
		# If none of the cases
		return False

	# def intersect(self, line: list) -> bool:
	# 	#return if intersect with a list of two np arrays specifying a line
	# 	raise NotImplementedError

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
		self.num_blue = 3
		self.num_red = 1
		self.num_grey = 3
		self.num_room = 5
		self.num_wall = 10

		world = World()
		#self.agents contains only policy agents (blue agents)
		self.agents = [BlueAgent() for i in range(self.num_blue)]
		world.agents = [BlueAgent() for i in range(self.num_blue)]

		world.landmarks = [RedAgent() for i in range(self.num_red)]
		world.landmarks.append([GreyAgent() for i in range(self.num_grey)])
		self.dummy_agents = self.num_red + self.num_grey
		self.walls = [Wall() for i in range(self.num_wall)]
		self.rooms = [Room() for i in range(self.num_room)]
		
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
		return self.reward(agent, world)

	def reward(self, agent, world):
		rew_belief = 0
		reward_penality = 0
		return rew_belief + reward_penality

	def observation(self, agent, world):
		other_pos = []
		other_vel = []
		for other in world.agents:
			if other is agent: continue
			other_pos.append(other.state.p_pos - agent.state.p_pos)
			other_vel.append(other.state.p_vel)
		return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + other_pos + other_vel)