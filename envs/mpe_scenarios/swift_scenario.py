import numpy as np
from multiagent.core import World, Agent, Wall, Entity
from multiagent.scenario import BaseScenario
from enum import Enum, unique
from multiagent.utils import Point, doIntersect
from multiagent.environment import AudioAction

class SwiftWolrdStat(object):
	def __init__(self, world):
		self.world = world

	def get_stats(self):
		# return a dictionary containing the following statistics for swift world
		# num_room_explored: number of rooms that have been explored
		# num_dummy_agents: number of dummy agents that have been found
		# max_belief: the largest cell belief (should corresponds to red)
		# min_belief: the smallest cell belief (should corresponds to grey)
		num_room_explored = np.sum(np.array([room.get_cell_states_binary() for room in self.world.rooms]).flatten())
		num_dummy_agents = 0
		for room in self.world.rooms:
			for cell in room.cells:
				cell_state = cell.get_cell_state()
				if cell_state == CellState.ExploredHasAgent:
					num_dummy_agents += 1
		beliefs = np.array([room.get_cell_beliefs() for room in self.world.rooms]).flatten()
		max_belief, min_belief = np.max(beliefs), np.min(beliefs)
		stat_dict = {'num_room_explored': num_room_explored,
					'num_dummy_agents': num_dummy_agents,
					'max_belief': max_belief,
					'min_belief': min_belief
					}
		if self.world._cached_fov:
			stat_dict['num_room_within_fov'] = self.world._cached_fov
		return stat_dict


@unique
class AudioResponse(Enum):
	AgentHandsUp = 1
	AgentFreeze = 2


RedResponseProbMatrix = [[0.2, 0.8], [0.65, 0.35]]
GreyResponseProbMatrix = [[0.8, 0.2], [0.35, 0.65]]

# RedResponseProbMatrix = [[0.1, 0.9], [0.9, 0.1]]
# GreyResponseProbMatrix = [[0.8, 0.2], [0.5, 0.5]]

class SwiftWorld(World):
	def __init__(self):
		super(SwiftWorld, self).__init__()
		self.old_belief = None 					#old_belief records the belief of last step, used to calculate belief update reward
		self.old_cell_state_binary = None     			#old_cell_state records boolean, explored or not
		self.stat = None
		self._cached_fov = None

	def get_stats(self):
		return self.stat.get_stats()

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
			BELIEF_THRES = 0.4   #belief above which assign no penalty
			if belief > BELIEF_THRES:
				return 0.0
			if audio == AudioAction.Freeze:
				return np.max([- 0.5 * np.log(belief + 1e-8) * (belief - BELIEF_THRES), -5])
			assert audio == AudioAction.HandsUp, "error"
			return np.max([- 1.0 * np.log(belief + 1e-8) * (belief - BELIEF_THRES), -10])

		audio_rew = np.array([0.0])
		#TODO: make sure each time step, this function has been called once and only once
		#TODO: should modify environment._step()
		self.record_old_belief()
		self.record_old_cell_state_binary()  #record if cell has been explored or not
		num_cell_within_fov = 0

		beliefs = np.array([room.get_cell_beliefs() for room in self.rooms]).flatten()
		max_belief, min_belief = np.max(beliefs), np.min(beliefs)
		print('max_belief is: ', max_belief, 'min_belief is: ', min_belief)
		for i, agent in enumerate(self.agents):

			if agent.action.audio is None:
				# print("agent:", i, "u", agent.action.u, "r" ,agent.action.r, " audio: None")
				print("agent:", i, " audio: None")
			else:
				print("agent:", i, " audio:", agent.action.audio)

			if agent.action.audio: #audio is not None
				if agent.action.audio == AudioAction.HandsUp:
					audio_rew -= 0.1 	#penalize audio action
				else:
					assert agent.action.audio == AudioAction.Freeze
					audio_rew -= 0.4
			for room in self.rooms:
				for cell in room.cells:
					cell_center = cell.get_cell_center()
					if agent.check_within_fov(cell_center) and doIntersect(cell_center, Point(agent.state.p_pos), room.window.p1, room.window.p2):
						num_cell_within_fov += 1
						cell.update_cell_state_once_observed()
						if cell.has_agent():
							cell.update_cell_belief_upon_audio(agent.action.audio)
							audio_rew += audio_belief_reward(agent.action.audio, cell.get_belief())

						#TODO: make sure agent.action has audio attribute
						#TODO: should also add audio action reward
		self._cached_fov = num_cell_within_fov
		current_cell_state_binary = np.array([room.get_cell_states_binary() for room in self.rooms]).flatten()
		old_cell_state_binary = self.old_cell_state_binary
		explore_cell_rew = 1.5 * np.sum(current_cell_state_binary - old_cell_state_binary)

		current_belief = np.array([room.get_cell_beliefs() for room in self.rooms]).flatten()
		old_belief = self.old_belief
		delta_belief = np.abs(current_belief - old_belief)
		belief_update_rew = 10.0 * np.sum(np.sqrt(delta_belief))

		# print('belief_update_rew', belief_update_rew)
		# print('audio reward', audio_rew)

		fov_reward = 0.5 * num_cell_within_fov
		rew = explore_cell_rew + belief_update_rew + audio_rew + fov_reward
		return rew[0]

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
		response_probability = self.response_probability_matrix[audio.value - 1] #enum member value start from 1
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
					   CellState.ExploredNoAgent: np.array([0, 1, 0]),
					   CellState.ExploredHasAgent: np.array([0, 0, 1])
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
		else:
			self._cell_state = CellState.ExploredNoAgent

	def update_cell_belief_upon_audio(self, audio: AudioAction):
		if not audio: #audio is None, agent does not take audio action
			#TODO: make sure in policy decoding steps, assign None to audio action if agent did not take it
			return

		if not self.has_agent():
			return
		sampled_response = self._occupant_agent.sample_response_to_audio(audio)
		belief_vector = np.array([self._belief[0], 1 - self._belief[0]])
		audio_index = audio.value - 1
		response_index = sampled_response.value - 1
		likelihood_vector = np.array([RedResponseProbMatrix[audio_index][response_index],
									  GreyResponseProbMatrix[audio_index][response_index]])
		belief_vector = belief_vector * likelihood_vector
		belief_vector = belief_vector / np.sum(belief_vector)
		self._belief = np.array([belief_vector[0]])
		return

	def add_agent(self, agent: DummyAgent):
		assert not self.has_agent()
		self._occupant_agent = agent
		self._occupant_agent.state.p_pos = np.array([self._center.x, self._center.y])

	def reset_cell_states(self):
		#remove occupant agent and set cell state to unexplored, and cell belief to 0.5
		self._occupant_agent = None
		self._cell_state = CellState.Unexplored
		self._belief = np.array([0.5])


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
		if abs(p1.x - p2.x) < 1e-5:
			wall_orient = 'V'
			wall_axis_pos = p1.x
			endpoints = (p1.y, p2.y)
		if abs(p1.y - p2.y) < 1e-5:
			wall_orient = 'H'
			wall_axis_pos = p1.y
			endpoints = (p1.x, p2.x)
		self.wall = Wall(orient=wall_orient, axis_pos=wall_axis_pos, endpoints=endpoints)
		self.wall.color = np.array([0,0.9,0.3])

class Room(object):
	def __init__(self, center: Point, x_scale, y_scale):
		self.center = center
		cell_centers = [self.center.new_point([ - x_scale / 4,   y_scale / 4]),
						self.center.new_point([   x_scale / 4,   y_scale / 4]),
						self.center.new_point([ - x_scale / 4, - y_scale / 4]),
						self.center.new_point([   x_scale / 4, - y_scale / 4])]
		cell_locations = [CellLocation.UpperLeft,
						  CellLocation.UpperRight,
						  CellLocation.BottomLeft,
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
		rand_cell_ind = np.random.choice(list(range(4)))
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
	def __init__(self, attached_agent, half_view_angle=np.pi/4, sensing_range=1.5):
		self._half_view_angle = half_view_angle
		self._sensing_range = sensing_range
		self._attached_agent = attached_agent
		self.color = np.array([1, 0.6, 0.1])

		self.half_view_angle = half_view_angle
		self.sensing_range = sensing_range

	def check_within_fov(self, p_in): #check if a point p is within fov
		#input p 2x1 numpy array
		if isinstance(p_in, Point):
			p = np.array([p_in.x, p_in.y])
		else:
			p = p_in
		vector1 = np.subtract(p, self._attached_agent.state.p_pos)
		if np.linalg.norm(vector1) > self.sensing_range:
			return False
		vector2 = np.array([np.cos(self._attached_agent.state.boresight), np.sin(self._attached_agent.state.boresight)]).squeeze()
		return True if np.inner(vector1, vector2)/np.linalg.norm(vector1) >= np.cos(self._half_view_angle) else False

class BlueAgent(Agent):
	def __init__(self, index, use_handcraft_policy=False):
		super(BlueAgent, self).__init__()
		# self.color = np.array([0.0, 0.0, 1.0])
		self.index = index
		self.FOV = FieldOfView(self)   #agent filed of view
		self.silent = True
		self.collide = True
		self.silent = True
		self.size = 0.015
		if index == 0:
			self.color = np.array([1.0, 0.0, 0.0])
		if index == 1:
			self.color = np.array([0.0, 1.0, 0.0])
		if index == 2:
			self.color = np.array([0.0, 0.0, 1.0])
		self.BlueAgent = True
		if use_handcraft_policy:
			from envs.mpe_scenarios.handcraft_policy import handcraft_policy
			self.action_callback = handcraft_policy

	def check_within_fov(self, p):
		return self.FOV.check_within_fov(p)

class Scenario(BaseScenario):
	def make_world(self, use_handcraft_policy=False):
		num_blue = 3
		num_red = 1
		num_grey = np.random.randint(3) + 1
		num_grey = 2
		num_room = num_red + num_grey
		num_room = 4
		# num_room = np.random.randint(5)
		arena_size = 2.0

		self.num_room = num_room
		self.num_red = num_red
		self.num_grey = num_grey
		self.max_room_num_per_dim = 4
		self.arena_size = arena_size
		self.room_length = self.arena_size / self.max_room_num_per_dim

		assert num_room <= self.max_room_num_per_dim**2, "must be <= maximum room num allowed"
		assert num_room >= num_grey + num_red, "must ensure each room only has less than 1 agent"

		world = SwiftWorld()
		world.arena_size = arena_size
		world.num_room = num_room

		world.stat = SwiftWolrdStat(world)
		#self.agents contains only policy agents (blue agents)
		world.agents = [BlueAgent(i, use_handcraft_policy=use_handcraft_policy) for i in range(num_blue)]
		for blue in world.agents:
			blue.initial_mass = 0.2

		for i, agent in enumerate(world.agents):
			agent.name = 'agent %d' % i

		self._reset_blue_states(world)

		world.dummy_agents = [RedAgent() for i in range(num_red)]
		world.dummy_agents += [GreyAgent() for i in range(num_grey)]

		self._set_rooms(world, num_room, arena_size=arena_size)

		# self._set_room_windows(world, num_room, arena_size=arena_size)
		self._set_walls(world, num_room, arena_size=arena_size)

		self.reset_world(world)  #reset_world also reset agents

		return world

	def _set_rooms(self, world, num_room, arena_size):
		x_index = np.random.permutation(self.max_room_num_per_dim)[0:self.num_room] * 2 + 1
		y_index = np.random.permutation(self.max_room_num_per_dim)[0:self.num_room] * 2 + 1
		# x_index = np.random.randint(self.max_room_num_per_dim, size=self.num_room) * 2 + 1
		# y_index = np.random.randint(self.max_room_num_per_dim, size=self.num_room) * 2 + 1
		for i in range(self.num_room):
			if x_index[i] != 1 and y_index[i] != self.max_room_num_per_dim*2-1:
				if np.random.random() >= 0.5:
					x_index[i] = 1
				else:
					y_index[i] = self.max_room_num_per_dim*2-1
			if x_index[i] == 1 and y_index[i] == self.max_room_num_per_dim*2-1:
				if np.random.random() >= 0.5:
					x_index[i] = self.max_room_num_per_dim*2-1
				else:
					y_index[i] = 1


		# length = arena_size / num_room
		room_centers = np.array([[-arena_size/2 + self.room_length/2*i, -arena_size/2 + self.room_length/2*j]
								 for i, j in zip(x_index, y_index)])
		world.rooms = [Room(Point(room_centers[i, :]), self.room_length, self.room_length) for i in range(num_room)]
		world.room_centers = room_centers

	# def _set_room_windows(self, world, num_room, arena_size):
	# 	length = self.room_length
	# 	room_centers = world.room_centers
		# print(room_centers[0, :])
		window_length = self.room_length / 2
		for i, room in enumerate(world.rooms):
			if x_index[i] == 1:
				orient_angle = 0
			else:
				orient_angle = -np.pi / 2

			room.window = Room_window(
				p1=Point(room_centers[i, :] + window_length*np.array([np.cos(orient_angle), np.sin(orient_angle)])),
				p2=Point(room_centers[i, :] + np.sqrt(2)*window_length*np.array([np.cos(orient_angle - np.pi/4), np.sin(orient_angle - np.pi/4)])))

	def _set_walls(self, world, num_room, arena_size):
		num_wall = 4*num_room
		length = self.room_length
		window_length = length / 2
		wall_orient = 'HVHV' * num_room
		wall_axis_pos = np.zeros((num_wall))
		wall_endpoints = []
		room_centers = world.room_centers
		for i in range(num_room):
			wall_axis_pos[4*i:4*i+4] = np.array([room_centers[i, 1] + window_length, room_centers[i, 0] - window_length,
												 room_centers[i, 1] - window_length, room_centers[i, 0] + window_length])
			wall_endpoints.append((room_centers[i, 0] - window_length, room_centers[i, 0] + window_length))
			wall_endpoints.append((room_centers[i, 1] - window_length, room_centers[i, 1] + window_length))
			wall_endpoints.append((room_centers[i, 0] - window_length, room_centers[i, 0] + window_length))
			wall_endpoints.append((room_centers[i, 1] - window_length, room_centers[i, 1] + window_length))

		world.walls = [Wall(orient=wall_orient[i], axis_pos=wall_axis_pos[i], endpoints=wall_endpoints[i]) for i in range(num_wall)]


		arena_size_larger = 3
		boundary_wall_orient = 'VHVH'
		boundary_wall_axis_pos = np.array([-arena_size_larger / 2, -arena_size_larger / 2, arena_size_larger / 2, arena_size_larger / 2])
		boundary_wall_endpoints = []
		boundary_wall_endpoints.append((-arena_size_larger / 2, arena_size_larger / 2))
		boundary_wall_endpoints.append((-arena_size_larger / 2, arena_size_larger / 2))
		boundary_wall_endpoints.append((-arena_size_larger / 2, arena_size_larger / 2))
		boundary_wall_endpoints.append((-arena_size_larger / 2, arena_size_larger / 2))
		for i in range(len(boundary_wall_orient)):
			world.walls.append(Wall(orient=boundary_wall_orient[i], axis_pos=boundary_wall_axis_pos[i],
									endpoints=boundary_wall_endpoints[i]))

		for room in world.rooms:
			world.walls.append(room.window.wall)

	def _reset_blue_states(self, world):
		# raise NotImplementedError
		for agent in world.agents:
			agent.silent = True
			agent.state.p_pos = np.random.uniform(-(self.arena_size-self.room_length), +(self.arena_size-self.room_length), world.dim_p)
			if agent.state.p_pos[1] > 0: agent.state.p_pos[1] = 0
			if agent.state.p_pos[1] < -0.9: agent.state.p_pos[1] = -0.9
			if agent.state.p_pos[0] < 0: agent.state.p_pos[0] = 0
			if agent.state.p_pos[0] > 0.9: agent.state.p_pos[0] = 0.9
			agent.state.p_vel = np.zeros(world.dim_p)
			agent.state.boresight = np.array([np.random.uniform(-np.pi, +np.pi)])
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
				flag_1 = agent.check_within_fov(cell_pos)
				flag_2 = doIntersect(Point(agent.state.p_pos), Point(cell_pos), room.window.p1, room.window.p2)
				fov_flag = encode_boolean(flag_1 and flag_2)
				cell_info.extend([cell_pos, fov_flag, cell.get_cell_state_encoding(), cell.get_belief()])

		return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + [agent.state.boresight] + other_pos + other_vel + other_heading + cell_info)


