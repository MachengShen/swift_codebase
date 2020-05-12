from .swift_scenario import Scenario, CellState
import numpy as np

class TestClass:
    def test_blue_index_init(self):
        world = Scenario().make_world()
        #test init room index
        for agent in world.dummy_agents:
            assert agent.room_index is not None

    def test_belief_init(self):
        #check belief init correctly
        world = Scenario().make_world()
        world.record_old_belief()
        belief = world.old_belief
        assert np.abs(np.sum(belief) - 4 * 0.5 * len(world.rooms)) < 1E-5

    def test_cell_state_init(self):
        world = Scenario().make_world()
        total_agent_count = 0
        for room in world.rooms:
            for cell in room.cells:
                assert cell.get_cell_state() == CellState.Unexplored
                assert cell.get_cell_state_binary() == 0.0
                if cell.has_agent():
                    total_agent_count += 1
                    assert cell._occupant_agent.room_index is not None
                    assert np.abs(np.sum(cell._occupant_agent.state.p_pos - np.array([cell._center.x, cell._center.y]))) < 1e-5
        assert total_agent_count == len(world.dummy_agents)





