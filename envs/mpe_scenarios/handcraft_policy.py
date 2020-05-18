


def handcraft_policy(agent, world):
    # action: one hot \in R^9, 0~4: translation; 5: rotation; 6~8: audio
    # TODO: as before, I can do translation and rotation; Macheng can do audio]
    # get_unexplored_room_index
    # agent not close to a unexplored room, or explored but with an agent?
    # build the room-agent correspondence, then translate to the corresponding room

    # agent_list[0] goes to room_index[0],
    # if len(agent_list) > len(room_index), redundant agents goes to room with belief close to 0.5

    # so far, agent is close to a certain window
    # if agent in agent_list: translate else rotate




    action = []
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


def get_audio_action():
    # not in previous two lists
    # and an agent has a dummy agent within FOV
    # and belief is within a threshold
    return 0

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
