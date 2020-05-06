import numpy as np
from multiagent.core import World, Agent, Landmark, Wall, Entity
from multiagent.scenario import BaseScenario
from enum import Enum, unique

@unique
class AudioAction(Enum):
	HandsUp = 1
	Freeze = 2

@unique
class AudioResponse(Enum):
	AgentHandsUp = 1
	AgentFreeze = 2

RedResponseProbMatrix = [[0.2, 0.8], [0.3, 0.7]]
GreyResponseProbMatrix = [[0.7, 0.3], [0.8, 0.2]]

class SwiftWorld(World):
	def __init__(self):
		super(SwiftWorld, self).__init__()
		self.old_belief = None 					#old_belief records the belief of last step, used to calculate belief update reward

class DummyAgent(Entity):
	#super class of red and grey agent that use pre-defined policies
	def __init__(self):
		super(DummyAgent, self).__init__()
		self.collide = True
		self.movable = False
		self.size = 0.05
		self.is_red = None
		self.response_probability_matrix = None

	def sample_response_to_audio(self, audio: AudioAction):
		#this method takes in an audio action by the blue agent
		#output a probablistic response according to the agent type
		response_probability = self.response_probability_matrix[AudioAction.value - 1] #enum member value start from 1
		sampled_response_index = np.random.choice(len(response_probability), 1, p=response_probability)[0]
		return AudioResponse(sampled_response_index + 1)

class RedAgent(DummyAgent):
	def __init__(self):
		super(RedAgent, self).__init__()
		self.color = np.array([1.0, 0.0, 0.0])
		self.is_red = True
		self.response_probability_matrix = RedResponseProbMatrix
		self.room_index = None

class GreyAgent(DummyAgent):
	def __init__(self):
		super(GreyAgent, self).__init__()
		self.color = np.array([0.2, 0.2, 0.2])
		self.is_red = False
		self.response_probability_matrix = GreyResponseProbMatrix
		self.room_index = None
@unique
class CellLocation(Enum):
	#Enumeration of celllocations within a room
	UpperLeft = 1
	UpperRight = 2
	BottomLeft = 3
	BottomRight = 4

@unique
class CellState(Enum):
	Unexplored = 1
	ExploredNoAgent = 2
	ExploredHasAgent = 3


class Room_cell(object):
	def __init__(self, center, cell_location, cell_state=CellState.Unexplored, occupant_agent=None, belief=0.5):
		#center of the room_cell: 2d np array
		self._center = center   				# Point object specifying the coordinate of cell
		self._cell_location = cell_location 	# relative location within a room: upperleft, upperright etc.
		self._cell_state = cell_state  			# Unexplored / ExploredNoAgent / ... etc
		if occupant_agent is not None:
			self.add_agent(occupant_agent)
		else:
			self._occupant_agent = None
		self._belief = belief   				#belief of occupant_agent being red
		self._belief_update_reward = 0

	def has_agent(self):
		return self._occupant_agent is not None

	def get_location(self):
		return self._cell_location

	def get_cell_state(self):
		return self._cell_state

	def get_cell_center(self):
		return self._center

	def get_belief(self):
		return self._belief

	def update_cell_state_once_observed(self):
		#this method is called if and only if this cell has been observed by blue
		if self.has_agent():
			self._cell_state = CellState.ExploredHasAgent
		self._cell_state = CellState.ExploredNoAgent

	def update_cell_belief_upon_audio(self, audio: AudioAction):
		if not self.has_agent():
			return
		sampled_response = self._occupant_agent.sample_response_to_audio(audio)
		belief_vector = np.array([self._belief, 1 - self._belief])
		audio_index = audio.value - 1
		response_index = sampled_response.value - 1
		likelihood_vector = np.array([RedResponseProbMatrix[audio_index][response_index],
									  GreyResponseProbMatrix[audio_index][response_index]])
		belief_vector = belief_vector * likelihood_vector
		belief_vector = belief_vector / np.sum(belief_vector)
		self._belief = belief_vector[0]
		return

	def add_agent(self, agent: DummyAgent):
		assert not self.has_agent()
		self._occupant_agent = agent

class Point:
	def __init__(self, xy):
		self.x = xy[0]
		self.y = xy[1]

	def new_point(self, xy):
		#generate a new point which is offset by xy
		return Point([self.x + xy[0], self.y + xy[1]])

# class checkIntersection:
# 	def __init__(self):

# https: // www.geeksforgeeks.org / check - if -two - given - line - segments - intersect /
class checkIntersection:
	def __init__(self):
		self.check = True
	def onSegment(self, p, q, r):
		# TODO: does this method use the points defining windows?
		# TODO: if this method is also needed by other classes, then do not define as
		# a class method
		if ((q.x <= max(p.x, r.x)) and (q.x >= min(p.x, r.x)) and
				(q.y <= max(p.y, r.y)) and (q.y >= min(p.y, r.y))):
			return True
		return False

	def orientation(self, p, q, r):
		# TODO: does this method use the points defining windows?

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
		# TODO: does this method use the points defining windows?
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


class Room_window(object):
	def __init__(self, p1, p2):
		#list of two np arrays contain the two end_points of the window
		self.p1 = p1
		self.p2 = p2
		raise NotImplementedError



class Room(object):
	def __init__(self, center: Point, x_scale, y_scale):
		self.center = center
		cell_centers = [self.center.new_point([ - x_scale / 4,   y_scale / 4]),
						self.center.new_point([   x_scale / 4,   y_scale / 4]),
						self.center.new_point([ - x_scale / 4, - y_scale / 4]),
						self.center.new_point([   x_scale / 4, - y_scale / 4])]
		cell_locations = [CellLocation.UpperLeft,
						  CellLocation.UpperRight,
						  CellLocation.BottomLeft.
						  CellLocation.BottomRight]
		self.cells = [Room_cell(c_center, c_location) for c_center, c_location in zip(cell_centers, cell_locations)]

	def has_agent(self) -> bool:
		return any([cell.has_agent() for cell in self.cells])

	def agent_location(self) -> CellLocation:
		#return the cell location, which occupied by an agent
		if not self.has_agent():
			raise Exception("no agent in the room")
		cell_has_agent = filter(lambda x: x.has_agent(), self.cells)
		assert len(cell_has_agent) == 1, "room contains a most one agent, check correctness"
		return cell_has_agent[0].get_location()

	def add_agent(self, agent: DummyAgent):
		assert not self.has_agent(), "room already contains one agent"
		rand_cell_ind = np.random.choice(list(range(4)))[0]
		self.cells[rand_cell_ind].add_agent(agent)

	def get_cell_states(self):
		return [cell.get_cell_state() for cell in self.cells]

	def get_cell_centers(self):
		return [cell.get_cell_center() for cell in self.cells]

	def get_cell_beliefs(self):
		return [cell.get_cell_belief() for cell in self.cells]


class FieldOfView(object):
	#blue agent filed of view
	def __init__(self, attached_agent, half_view_angle=np.pi/3, sensing_range=0.2):
		self._half_view_angle = half_view_angle
		self._sensing_range = sensing_range
		self._attached_agent = attached_agent
	def check_within_fov(self, p): #check if a point p is within fov
		vector1 = np.subtract(p, self._attached_agent.state.p_pos)
		vector2 = np.array([np.cos(self._attached_agent.boresight), np.sin(self._attached_agent.boresight)])
		return True if np.inner(vector1, vector2)/np.linalg.norm(vector1) >= np.cos(self._half_view_angle) else False
		# raise NotImplementedError


class BlueAgent(Agent):
	def __init__(self):
		super(BlueAgent, self).__init__()
		self.color = np.array([0.0, 0.0, 1.0])
		self.FOV = FieldOfView(self)   #agent filed of view
		# self.state.boresight = np.pi/2

	def check_within_fov(self, p):
		return self.FOV.check_within_fov(p)

class Scenario(BaseScenario):
	def make_world(self):
		num_blue = 3
		num_red = 1
		num_grey = 2
		num_room = 4
		num_wall = 3*num_room + 1
		wall_orient = "H" * num_wall
		wall_axis_pos = np.zeros((num_wall))
		wall_endpoints = []

		self.num_room = num_room
		self.num_wall = num_wall
		self.num_red = num_red
		self.num_grey = num_grey

		room_centers = np.zeros((num_room, 2))
		for i in range(num_room):
			room_centers[i,:] = np.array([-0.75+i*0.5, 0.75])

		assert num_room >= num_grey + num_red, "must ensure each room only has less than 1 agent"

		world = SwiftWorld()
		#self.agents contains only policy agents (blue agents)
		world.agents = [BlueAgent() for i in range(num_blue)]

		world.dummy_agents = [RedAgent() for i in range(num_red)]
		world.dummy_agents += [GreyAgent() for i in range(num_grey)]

		
		world.walls = [Wall(orient=wall_orient[i], axis_pos=wall_axis_pos[i], endpoints=wall_endpoints[i]) for i in range(num_wall)]
		world.rooms = [Room(Point(room_centers[i,:]), 0.5, 0.5) for i in range(num_room)]

		#TODO: chuangchuang implements wall and room generation
		self._set_walls(world)
		self._set_rooms(world)
		
		raise NotImplementedError
	
	def _reset_dummy_agents_location(self, world):
		#TODO: chuangchuang implements
		#use 'add_agent' method of the room object
		for i in range(self.num_red + self.num_grey):
			room_index = world.dummy_agents[i].room_index
			world.rooms[room_index].add_agent(world.dummy_agents[i])
		# raise NotImplementedError

	def _permute_dummy_agents_index(self, world):
		#TODO: chuangchuang implements
		permuted_index = np.random.permutation(self.num_room)
		for i in range(self.num_red + self.num_grey):
			world.dummy_agents[i].room_index = permuted_index[i]
		# return np.random.permutation(self.num_room)[:self.num_red + self.num_grey]
		# raise NotImplementedError

	def _set_walls(self, world):
		# TODO: chuangchuang implements

		raise NotImplementedError

	def _set_rooms(self, world):
		# TODO: chuangchuang implements
		raise NotImplementedError

	def reset_world(self, world):
		world._permute_dummy_agents_index()
		world._reset_dummy_agents_location()
		world._initilize_room_belief()
		raise NotImplementedError

	def benchmark_data(self, agent, world):
		# returns data for benchmarking purposes
		return self.reward(agent, world)

	def reward(self, agent, world):
		rew_belief = 0
		reward_penality = 0
		raise NotImplementedError
		return rew_belief + reward_penality

	def observation(self, agent, world):
		other_pos = []
		other_vel = []
		for other in world.agents:
			if other is agent: continue
			other_pos.append(other.state.p_pos - agent.state.p_pos)
			other_vel.append(other.state.p_vel)
		raise NotImplementedError
		return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + other_pos + other_vel)