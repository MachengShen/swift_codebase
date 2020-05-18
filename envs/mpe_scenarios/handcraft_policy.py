from .swift_scenario import CellState, AudioAction
from multiagent.core import Action


def handcraft_policy(agent, world)->Action:
    # action: one hot \in R^9, 0~4: translation; 5: rotation; 6~8: audio
    # TODO: as before, I can do translation and rotation; Macheng can do audio]
    # get_unexplored_room_index
    # agent not close to a unexplored room, or explored but with an agent?
    # build the room-agent correspondence, then translate to the corresponding room

    # agent_list[0] goes to room_index[0],
    # if len(agent_list) > len(room_index), redundant agents goes to room with belief close to 0.5

    # so far, agent is close to a certain window
    # if agent in agent_list: translate else rotate

    action = Action()




    return action





def get_translate_agent_list():
    # if an agent is far from all unexplored(explored but with agents) room within a threshold
    # basically within a threshold to the window center, then the agent can see all the cells with a proper boresight angle
    index = []
    return index


def if_rotate():
    # if agent not in translate_agent_index, and no agent in this room are in FOV
    # rotate to the vector connecting agent and window center
    # check cell state
    index = []
    return index


def get_audio_action(agent, world):
    # not in previous two lists
    # and an agent has a dummy agent within FOV
    # and belief is within a threshold
    def _agent_near_window():   #this function should be compatible with the thredhold check in get_translate_agent_list()
        raise NotImplementedError

    for room in world.rooms:
        if _agent_near_window(agent, room):
            for cell in room.cells:
                if not agent.check_within_fov(cell.get_cell_center()):
                    continue
                if cell.get_cell_state() == CellState.ExploredHasAgent:
                    cell_belief = cell.get_belief()
                    if cell_belief < 0.1 or cell_belief > 0.95:
                        return None
                    if cell_belief < 0.5:
                        return AudioAction.Freeze
                    else:
                        return AudioAction.HandsUp
    raise Exception("cannot find a valid action, check if there is any dummy agent within fov")

# def get_audio_agent_list():
#     # not in previous two lists
#     # and an agent has a dummy agent within FOV
#     # and belief is within a threshold
#     index = []
#     return index


# how to translate
def get_unexplored_room_list():
    # if a room has a cell unexplored, this room is labelled as unexplored
    # get_room_without_agent_nearby_index
    index = []
    return index


def get_room_most_uncertain():
    # for room/cell with agent, get its belief and find the one closest to 0.5
    index = []
    return index
