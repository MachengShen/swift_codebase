import numpy as np
from multiagent.core import World, Agent, Landmark, Wall, Entity
from multiagent.scenario import BaseScenario
from enum import Enum, unique
from .utils import doIntersect, Point

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
		self.old_cell_state_binary = None     			#old_cell_state records boolean, explored or not
		#self._time = 0

	def record_old_belief(self):
		self.old_belief = np.array([room.get_cell_beliefs() for room in self.rooms]).flatten()

	def record_old_cell_state_binary(self):
		self.old_cell_state_binary = np.array([room.get_cell_states_binary() for room in self.rooms]).flatten()

	#def increment_world_time(self):
	#	self._time += 1

	def step_belief(self):
		#this step function is the additional to the physical state update
		#should be called before calling reward
		def audio_belief_reward(audio, belief):
			#specify the audio action reward on dummy agent
			if not audio: #audio action is None
				return 0.0
			BELIEF_THRES = 0.3   #belief above which assign no penalty
			if belief > BELIEF_THRES:
				return 0.0
			if audio == AudioAction.Freeze:
				return 0.1 * (belief - BELIEF_THRES)
			assert audio == AudioAction.HandsUp, "error"
			return 0.3 * (belief - BELIEF_THRES)

		audio_rew = 0.0
		#TODO: make sure each time step, this function has been called once and only once
		#TODO: should modify environment._step()
		self.record_old_belief()
		self.record_old_cell_state_binary()  #record if cell has been explored or not
		for agent in self.agents:
			if agent.action.audio: #audio is not None
				audio_rew -= 0.1 	#penalize audio action
			for room in self.rooms:
				for cell in room.cells:
					cell_center = cell.get_cell_center()
					if agent.check_within_fov(cell_center) and doIntersect(cell_center, Point(agent.state.p_pos), room.window.p1, room.window.p2):
						cell.update_cell_state_once_observed()
						if cell.has_agent():
							cell.update_cell_belief_upon_audio(agent.action.audio)
							audio_rew += audio_belief_reward(agent.action.audio, cell.get_belief())

						#TODO: make sure agent.action has audio attribute
						#TODO: should also add audio action reward

		current_cell_state_binary = np.array([room.get_cell_states_binary() for room in self.rooms]).flatten()
		old_cell_state_binary = self.old_cell_state_binary
		explore_cell_rew = 0.2 * np.sum(current_cell_state_binary - old_cell_state_binary)

		current_belief = np.array([room.get_cell_beliefs() for room in self.rooms]).flatten()
		old_belief = self.old_belief
		delta_belief = np.abs(current_belief - old_belief)
		belief_update_rew = 5.0 * np.sum(np.sqrt(delta_belief))

		rew = explore_cell_rew + belief_update_rew + audio_rew

		return rew

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

CELL_STATE_ENCODING = {CellState.Unexplored: np.array([1, 0, 0]),
					   CellState.Unexplored: np.array([0, 1, 0]),
					   CellState.Unexplored: np.array([0, 0, 1])
					   }

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

	def has_agent(self):
		return self._occupant_agent is not None

	def get_location(self):
		return self._cell_location

	def get_cell_state(self):
		return self._cell_state

	def get_cell_state_binary(self):
		if self._cell_state == CellState.Unexplored:
			return 0.0
		assert self._cell_state == CellState.ExploredHasAgent or self._cell_state == CellState.ExploredNoAgent, "error"
		return 1.0

	def get_cell_state_encoding(self):
		return CELL_STATE_ENCODING[self.get_cell_state()]

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
		if not audio: #audio is None, agent does not take audio action
			#TODO: make sure in policy decoding steps, assign None to audio action if agent did not take it
			return

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
		self._occupant_agent.state.p_pos = self._center

	def reset_cell_states(self):
		#remove occupant agent and set cell state to unexplored, and cell belief to 0.5
		self._occupant_agent = None
		self._cell_state = CellState.Unexplored
		self._belief = 0.5


# class Point:
# 	def __init__(self, xy):
# 		self.x = xy[0]
# 		self.y = xy[1]
#
# 	def new_point(self, xy):
# 		#generate a new point which is offset by xy
# 		return Point([self.x + xy[0], self.y + xy[1]])

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
		self.window = None

	def reset_room_cell_states(self):
		for cell in self.cells:
			cell.reset_cell_states()

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

	def get_cell_states_binary(self):
		return [cell.get_cell_state_binary() for cell in self.cells]

	def get_cell_centers(self):
		return [cell.get_cell_center() for cell in self.cells]

	def get_cell_beliefs(self):
		return [cell.get_belief() for cell in self.cells]


class FieldOfView(object):
	#blue agent filed of view
	def __init__(self, attached_agent, half_view_angle=np.pi/3, sensing_range=0.2):
		self._half_view_angle = half_view_angle
		self._sensing_range = sensing_range
		self._attached_agent = attached_agent

	def check_within_fov(self, p): #check if a point p is within fov
		#input p 2x1 numpy array
		vector1 = np.subtract(p, self._attached_agent.state.p_pos)
		#TODO: boresight definition?
		vector2 = np.array([np.cos(self._attached_agent.state.boresight), np.sin(self._attached_agent.state.boresight)])
		return True if np.inner(vector1, vector2)/np.linalg.norm(vector1) >= np.cos(self._half_view_angle) else False

class BlueAgent(Agent):
	def __init__(self):
		super(BlueAgent, self).__init__()
		# self.color = np.array([0.0, 0.0, 1.0])
		self.FOV = FieldOfView(self)   #agent filed of view
		self.silent = True
		self.collide = True
		self.silent = True
		self.size = 0.15
		self.color = np.array([0.35, 0.35, 0.85])

	def check_within_fov(self, p):
		return self.FOV.check_within_fov(p)

class Scenario(BaseScenario):
	def make_world(self):
		num_blue = 3
		num_red = 1
		num_grey = 2
		num_room = 4
		arena_size = 2.0

		self.num_room = num_room
		self.num_red = num_red
		self.num_grey = num_grey

		assert num_room >= num_grey + num_red, "must ensure each room only has less than 1 agent"

		world = SwiftWorld()
		#self.agents contains only policy agents (blue agents)
		world.agents = [BlueAgent() for i in range(num_blue)]
		for i, agent in enumerate(world.agents):
			agent.name = 'agent %d' % i


		world.dummy_agents = [RedAgent() for i in range(num_red)]
		world.dummy_agents += [GreyAgent() for i in range(num_grey)]

		self._set_rooms(world, num_room, arena_size)
		self._set_walls(world, num_room, arena_size)
		self._set_room_windows(world, num_room, arena_size=arena_size)

		self.reset_world(world)  #reset_world also reset agents

		raise NotImplementedError
	

	def _set_walls(self, world, num_room, arena_size):
		num_wall = 3 * num_room + 1
		length = arena_size / num_room
		window_length = length / 2
		wall_orient = "H" * num_wall
		wall_axis_pos = np.zeros((num_wall))
		wall_endpoints = []
		for i in range(num_room):
			wall_orient[3*i:3*i+3] = 'HVH'
			wall_axis_pos[3*i:3*i+3] = np.array([arena_size/2, -arena_size/2 + length*i, arena_size/2-length])
			wall_endpoints.append((-arena_size/2 + length*i, -arena_size/2 + length*(i+1)))
			wall_endpoints.append((arena_size/2, arena_size/2-length))
			wall_endpoints.append((-arena_size/2 + length*i, -arena_size/2 + length*(i+1) - window_length))
		wall_orient[num_room-1] = 'V'
		wall_axis_pos[num_room-1] = arena_size/2
		wall_endpoints.append((arena_size/2, arena_size/2-length))

		world.walls = [Wall(orient=wall_orient[i], axis_pos=wall_axis_pos[i], endpoints=wall_endpoints[i]) for i in range(num_wall)]

	def _set_rooms(self, world, num_room, arena_size=2):
		length = arena_size / num_room
		room_centers = np.array([[-arena_size/2 + length/2 + i * length, arena_size/2 - length/2] for i in num_room])
		world.rooms = [Room(Point(room_centers[i, :]), length, length) for i in range(num_room)]
		# raise NotImplementedError

	def _set_room_windows(self, world, num_room, arena_size=2):
		length = arena_size / num_room
		window_length = length / 2
		for i, room in enumerate(world.rooms):
			room.window = Room_window(p1=Point(np.array([-arena_size/2 + length*i + window_length, arena_size/2-length])),
										  p2=Point(np.array([-arena_size/2 + length*(i+1), arena_size/2-length])))

	def _reset_blue_states(self, world):
		# raise NotImplementedError
		for agent in world.agents:
			agent.silent = True
			agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
			agent.state.p_vel = np.zeros(world.dim_p)
			agent.state.c = np.zeros(world.dim_c)

	def _permute_dummy_agents_index(self, world):
		permuted_index = np.random.permutation(self.num_room)
		for i in range(self.num_red + self.num_grey):
			world.dummy_agents[i].room_index = permuted_index[i]

	def _reset_dummy_agents_location(self, world):
		for room in world.rooms:
			room.reset_room_cell_states()

		for i, agent in enumerate(world.dummy_agents):
			world.rooms[agent.room_index].add_agent(agent)

	def reset_world(self, world):
		self._reset_blue_states(world)
		self._permute_dummy_agents_index(world)
		self._reset_dummy_agents_location(world)  #room states are also reset
		world.record_old_belief()
		world.record_old_cell_state_binary()

	def benchmark_data(self, agent, world):
		# returns data for benchmarking purposes
		return self.reward(agent, world)

	def reward(self, agent, world):
		raise Exception("use world.step_belief(), should not call for every agent")

	def observation(self, agent, world):
		# info from the other agents
		other_pos = []
		other_vel = []
		other_heading = []
		for other in world.agents:
			if other is agent: continue
			other_pos.append(other.state.p_pos - agent.state.p_pos)
			other_vel.append(other.state.p_vel)
			other_heading.append(other.state.boresight)

		cell_info = []

		def encode_boolean(bool):
			return np.array([1, 0]) if bool else np.array([0, 1])

		for room in world.rooms:
			for cell in room.cells:
				cell_pos = np.array([cell._center.x, cell._center.y])
				flag_1 = encode_boolean(agent.check_within_fov(cell_pos))
				flag_2 = encode_boolean(doIntersect(Point(agent.state.p_pos), Point(cell_pos), room.window.p1, room.window.p2))
				cell_info.extend([cell_pos, flag_1, flag_2, cell.get_cell_state_encoding(), cell.get_belief()])

		return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + [agent.state.boresight] + other_pos + other_vel + other_heading + cell_info)


