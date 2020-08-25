from .swift_scenario import Scenario, CellState, AudioAction
import numpy as np


class TestClass:
    def test_blue_index_init(self):
        world = Scenario().make_world()
        # test init room index
        for agent in world.dummy_agents:
            assert agent.room_index is not None

    def test_belief_init(self):
        # check belief init correctly
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
                    assert np.abs(np.sum(cell._occupant_agent.state.p_pos -
                                         np.array([cell._center.x, cell._center.y]))) < 1e-5
        assert total_agent_count == len(world.dummy_agents)

    def test_belief_reward(self):
        world = Scenario().make_world()
        world.record_old_belief()
        world.record_old_cell_state_binary()
        world.step_belief()
        assert np.abs(world.step_belief()) < 1e-4

    def test_cell_update(self):
        world = Scenario().make_world()
        world.record_old_belief()
        world.record_old_cell_state_binary()
        assert np.sum(world.old_cell_state_binary) < 1e-3

        for room in world.rooms:
            for cell in room.cells:
                cell.update_cell_state_once_observed()
                assert cell.get_cell_state() is not CellState.Unexplored
                cell.update_cell_belief_upon_audio(AudioAction.Freeze)
                if not cell.has_agent():
                    assert np.abs(cell.get_belief() - 0.5) < 1e-5
                else:
                    assert np.abs(cell.get_belief() - 0.5) > 1e-5
        current_cell_state_binary = np.array(
            [room.get_cell_states_binary() for room in world.rooms]).flatten()
        assert np.abs(np.sum(current_cell_state_binary) -
                      4 * len(world.rooms)) < 1e-5

    def test_reset(self):
        scenario = Scenario()
        world = scenario.make_world()
        world.record_old_belief()
        world.record_old_cell_state_binary()
        for room in world.rooms:
            for cell in room.cells:
                cell.update_cell_state_once_observed()
                cell.update_cell_belief_upon_audio(AudioAction.Freeze)

        scenario.reset_world(world)

        belief = world.old_belief
        assert np.abs(np.sum(belief) - 4 * 0.5 * len(world.rooms)) < 1E-5

        total_agent_count = 0
        for room in world.rooms:
            for cell in room.cells:
                assert cell.get_cell_state() == CellState.Unexplored
                assert cell.get_cell_state_binary() == 0.0
                if cell.has_agent():
                    total_agent_count += 1
                    assert cell._occupant_agent.room_index is not None
                    assert np.abs(np.sum(cell._occupant_agent.state.p_pos -
                                         np.array([cell._center.x, cell._center.y]))) < 1e-5
        assert total_agent_count == len(world.dummy_agents)
