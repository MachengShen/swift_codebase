from .swift_scenario_3d import CellState, AudioAction
from multiagent.core import Action


from .swift_scenario_3d import CellState
import numpy as np
from multiagent.utils import Point, doIntersect

# def handcraft_policy(agent, world):

DELTA = 1/8
def handcraft_policy(agent, world)->Action:
    # row_index = world.row_index
    # world.rooms = [world.rooms[row_index_] for row_index_ in row_index]
    world.rooms = world.agent_rooms
    # action: in 2D, one hot \in R^9, 0~4: translation; 5: rotation; 6~8: audio
    # action: in 3D, one hot \in R^12, 0~6: translation; 7~8: rotation; 9~11: audio

    # get_unexplored_room_index
    # agent not close to a unexplored room, or explored but with an agent?
    # build the room-agent correspondence, then translate to the corresponding room

    # agent_list[0] goes to room_index[0],
    # if len(agent_list) > len(room_index), redundant agents goes to room with belief close to 0.5

    # so far, agent is close to a certain window
    # if agent in agent_list: translate else rotate
    action = Action()

    dist_thres = world.arena_size / world.num_room / 4 / (1*np.sqrt(2))
    dist_thres = world.arena_size / world.num_room / 4 / 2
    # dist_thres = 1

    agent_list, room_index, uncertainty_sort_index = \
        get_translate_agent_list(agent, world, dist_thres)

    translation_action = np.zeros(2*world.dim_p + 1)
    action.u = np.zeros(world.dim_p)
    if len(agent_list) > len(room_index):
        for i in range(len(agent_list) - len(room_index)):
            room_index.append(uncertainty_sort_index[i])

    for i, blue_agent in enumerate(world.agents):
        if agent is blue_agent:
            agent_index = i
            break

    print(dist_thres, agent_index, agent_list)
    for count, idx in enumerate(agent_list):
        if agent_index == idx:
            room = world.rooms[room_index[count]]
            # room_window_center = 0.5 * np.array([room.window.p1.x + room.window.p2.x,
            #                                      room.window.p1.y + room.window.p2.y])
            room_window_center = get_window_center(room.window)

            dx = room_window_center[0] - agent.state.p_pos[0] + 0
            dy = room_window_center[1] - agent.state.p_pos[1] + 0
            dz = room_window_center[2] - agent.state.p_pos[2] + 0
            dxyz = np.array([dx, dy, dz])
            if np.random.uniform(0, 1) >= 0.05:
                dim_u = np.argmax(np.abs(dxyz))
            else:
                dim_u = np.random.randint(world.dim_p)
            if dxyz[dim_u] > 0:
                translation_action[2*dim_u + 2] = 1
                    # action.u = 2
            else:
                translation_action[2*dim_u + 1] = 1
                    # action.u = 1

            # if np.abs(dx) > np.abs(dy):
            #     if dx > 0:
            #         translation_action[2] = 1
            #         # action.u = 2
            #     else:
            #         translation_action[1] = 1
            #         # action.u = 1
            # else:
            #     if dy > 0:
            #         translation_action[4] = 1
            #         # action.u = 4
            #     else:
            #         translation_action[3] = 1
                    # action.u = 3

            action.u[0] = -1 * (translation_action[1] - translation_action[2])
            action.u[1] = -1 * (translation_action[3] - translation_action[4])
            action.u[2] = -1 * (translation_action[5] - translation_action[6])
            action.u = 1*action.u
            action.r = np.array([0.0, 0.0])
            action.audio = None
            # print(action.u)
            return action

            # if action[0] == 1: agent.action.u[0] = -1.0
            # if action[0] == 2: agent.action.u[0] = +1.0
            # if action[0] == 3: agent.action.u[1] = -1.0
            # if action[0] == 4: agent.action.u[1] = +1.0
            # agent.action.u[0] += action[0][1] - action[0][2]
            # agent.action.u[1] += action[0][3] - action[0][4]
    flag_rotate, rotate_action, cell_belief = if_rotate(agent, world, dist_thres)
    if flag_rotate:
        action.r = rotate_action
        action.u = np.array([0.0, 0.0, 0.0])
        action.audio = None
        print(action.r)
        return action

    audio_action = get_audio_action(agent, world, dist_thres, cell_belief)
    action.u = np.array([0.0, 0.0, 0.0])
    action.r = np.array([0.0, 0.0])
    action.audio = audio_action
    return action

def get_window_center(window):
    endpoints = np.array(window.endpoints)
    if window.orient == 'x':
        dim_normal = 0
        room_window_center = np.array([window.axis_pos, np.mean(endpoints[0:2]), np.mean(endpoints[2:])])
    elif window.orient == 'y':
        dim_normal = 1
        room_window_center = np.array([np.mean(endpoints[0:2]), window.axis_pos, np.mean(endpoints[2:])])
    elif window.orient == 'z':
        dim_normal = 2
        room_window_center = np.array([np.mean(endpoints[0:2]), np.mean(endpoints[2:]), window.axis_pos])
    else:
        raise ValueError
    return room_window_center


def is_room_all_explored(room):
    count = 0
    for cell in room.cells:
        if cell.get_cell_state().name == CellState.Unexplored.name:
            count += 1
    flag_all_cells_explored = False if count > 0 else True
    return flag_all_cells_explored


def if_room_has_dummy_inside(room):
    for cell in room.cells:
        if cell.get_cell_state().name == CellState.ExploredHasAgent.name:
            return True
    return False


def if_blue_with_room(agent, world, dist_thres):
    for room_index, room in enumerate(world.rooms):
        room_window_center = get_window_center(room.window)
        # room_window_center = 0.5 * np.array([room.window.p1.x + room.window.p2.x,
        #                                      room.window.p1.y + room.window.p2.y])
        if np.linalg.norm(agent.state.p_pos - room_window_center) <= dist_thres:
            if not is_room_all_explored(room) or if_room_has_dummy_inside(room):
                # print(np.linalg.norm(agent.state.p_pos - room_window_center)<= dist_thres,
                #       not is_room_all_explored(room), if_room_has_dummy_inside(room))
                return True, room_index
    # print(np.linalg.norm(agent.state.p_pos - room_window_center) <= dist_thres,
    #       not is_room_all_explored(room), if_room_has_dummy_inside(room))
    return False, None


def get_translate_agent_list(agent, world, dist_thres):
    # if an agent is far from all unexplored(explored but with agents) room within a threshold
    # basically within a threshold to the window center, then the agent can see all the cells with a proper boresight angle
    # get_unexplored_room_list()
    # get_room_most_uncertain()
    def if_room_close_to_blue(room, world):
        window = room.window
        room_window_center = get_window_center(window)
        # room_window_center = 0.5*np.array([room.window.p1.x + room.window.p2.x,
        #                                room.window.p1.y + room.window.p2.y])
        for blue_agent in world.agents:
            if np.linalg.norm(blue_agent.state.p_pos - room_window_center) <= dist_thres:
                    return True
        return False

    def get_room_most_uncertain(world):
        uncertainy = np.zeros((len(world.rooms)))
        for i, room in enumerate(world.rooms):
            uncertainy[i] = 0
            for cell in room.cells:
                uncertainy[i] += np.abs(cell.get_belief() - 0.5)
        uncertainty_sort_index = np.argsort(-uncertainy)
        return uncertainty_sort_index

    room_index = []
    # for i, room in enumerate(world.rooms):
    for i, room in enumerate(world.agent_rooms):
        if not is_room_all_explored(room) or if_room_has_dummy_inside(room):
            if not if_room_close_to_blue(room, world):
                room_index.append(i)

    agent_list = []
    for i, blue_agent in enumerate(world.agents):
        flag, _ = if_blue_with_room(blue_agent, world, dist_thres)
        if not flag:
            agent_list.append(i)

    uncertainty_sort_index = get_room_most_uncertain(world)
    return agent_list, room_index, uncertainty_sort_index


def if_rotate(agent, world, dist_thres):
    def rotation_action(agent, my_room):
        rotate_action = np.array([0, 0], dtype=float)
        rotate_unit = 1. #1 or np.pi/2
        my_room_center = np.array([my_room.center.x, my_room.center.y, my_room.center.z])
        prj_point = np.array([agent.state.p_pos[0], agent.state.p_pos[1], my_room_center[2]])
        angle_to_plane = np.arcsin((agent.state.p_pos[2]-my_room_center[2]) / np.linalg.norm(agent.state.p_pos-my_room_center))
        angle_in_xy = np.pi + np.arctan2(prj_point[1] - agent.state.p_pos[1],
                                        prj_point[0] - agent.state.p_pos[0])
        angle = np.array([angle_in_xy, angle_to_plane])
        delta_angle = agent.state.boresight - angle
        dim_r = np.argmax(np.abs(delta_angle))
        # angle_to_room_center = np.pi + np.arctan2(my_room.center.y - agent.state.p_pos[1],
        #                                 my_room.center.x - agent.state.p_pos[0])
        if agent.state.boresight[dim_r] >= angle[dim_r]:
            # rotate_action = np.array([1, 0])
            rotate_action[dim_r] = -rotate_unit
        else:
            # rotate_action = np.array([0, 1])
            rotate_action[dim_r] = +rotate_unit
        return rotate_action

    # if agent not in translate_agent_index,
    # and no dummy agent in this room are in FOV
    # not all cell state are explored
    # rotate to the vector connecting agent and window center

    _, room_index = if_blue_with_room(agent, world, dist_thres)
    my_room = world.rooms[room_index]
    for cell in my_room.cells:
        cell_center = cell.get_cell_center()
        # print('in handcraft policy, cell_center:', cell_center.x, cell_center.y, cell_center.z)
        cell_center_array = np.array([cell_center.x, cell_center.y, cell_center.z])
        flag_see_the_cell = agent.check_within_fov(cell_center_array) \
                            and doIntersect(cell_center_array, agent.state.p_pos, my_room.window)
        if flag_see_the_cell and cell.get_cell_state().name == CellState.ExploredHasAgent.name:
            cell_belief = cell.get_belief()
            return False, 0.0, cell_belief
    if not is_room_all_explored(my_room):
        return True, rotation_action(agent, my_room), None
    else:
        return True, 0.0, None
    # raise Exception("cannot see the dummy inside the room!")

    # if not flag_all_cells_explored and not flag_dummy_in_FOV:
    # if not flag_dummy_in_FOV:
    #     if not flag_all_cells_explored:
    #         flag_rotate = True
    # else:
    #     flag_rotate = False
    # print(flag_all_cells_explored,flag_dummy_in_FOV)
    # rotate_action = np.array([0, 0])



def get_audio_action(agent, world, dist_thres, cell_belief):
    # not in previous two lists
    # and an agent has a dummy agent within FOV
    # and belief is within a threshold
    # def _agent_near_window(agent, room, dist_thres):   #this function should be compatible with the thredhold check in get_translate_agent_list()
    #     # raise NotImplementedError
    #     flag = False
    #     room_window_center = 0.5 * np.array([room.window.p1.x + room.window.p2.x,
    #                                          room.window.p1.y + room.window.p2.y])
    #     if np.linalg.norm(agent.state.p_pos - room_window_center) <= dist_thres:
    #         flag = True
    # for room in world.rooms:
    #     if _agent_near_window(agent, room, dist_thres):
    #         for cell in room.cells:
    #             if not agent.check_within_fov(cell.get_cell_center()):
    #                 continue
    #             if cell.get_cell_state() == CellState.ExploredHasAgent:
    # cell_belief = cell.get_belief()
    if cell_belief < 0.1 or cell_belief > 0.849:
        return None
    if cell_belief < 0.5:
        return AudioAction.Freeze
    else:
        return AudioAction.HandsUp
    # return None
    # raise Exception("cannot find a valid action, check if there is any dummy agent within fov")

# def get_audio_agent_list():
#     # not in previous two lists
#     # and an agent has a dummy agent within FOV
#     # and belief is within a threshold
#     index = []
#     return index


# how to translate
# def get_unexplored_room_list():
#     # if a room has a cell unexplored, this room is labelled as unexplored
#     # get_room_without_agent_nearby_index
#     index = []
#     return index
#
#
# def get_room_most_uncertain():
#     # for room/cell with agent, get its belief and find the one closest to 0.5
#     index = []
#     return index

