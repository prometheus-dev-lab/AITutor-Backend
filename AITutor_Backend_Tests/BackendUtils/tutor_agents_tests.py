# Tests LRU policy
import unittest
import uuid
from AITutor_Backend.src.BackendUtils.tutor_agents import TutorAgentNode, LRUAgentCache


class LRUAgentCacheTests(unittest.TestCase):
    def test_lru_functionality(self):

        # Testing the LRUAgentCache with a capacity of 2 to observe eviction behavior
        cache = LRUAgentCache(2)
        agent_1 = "Agent 1"
        agent_2 = "Agent 2"
        agent_3 = "Agent 3"

        # Add agent1 and check its retrieval
        uuid_1 = uuid.uuid4()
        output_agent = cache.add_agent(uuid_1, agent_1)

        assert output_agent is None, "Unexpected Non-null value."
        agent_retrieval = cache.get_agent(uuid_1)
        assert (
            agent_retrieval == agent_1
        ), f"Unexpected Value: Expected {agent_1}, Actual {agent_retrieval}"

        # Add agent2, which should cause eviction of agent1 due to capacity limit
        uuid_2 = uuid.uuid4()
        cache.add_agent(uuid_2, agent_2)
        agent_retrieval = cache.get_agent(uuid_2)  # Check retrieval of agent2
        assert (
            agent_retrieval == agent_2
        ), f"Unexpected Value: Expected {agent_2}, Actual {agent_retrieval}"

        uuid_3 = uuid.uuid4()
        output_agent = cache.add_agent(uuid_3, agent_3)

        assert (
            output_agent is not None
        ), "Error implementing LRU Policy, no Agent was removed."

        assert (
            output_agent.agent == agent_1
        ), f"Error implementing LRU Policy. Output: {output_agent.agent} Expected: {agent_1}"

        # Check if agent_1 is still accessible
        agent_retrieval = cache.get_agent(uuid_1)
        assert (
            agent_retrieval is None
        ), f"Unexpected Behavior: Expected agent_1 to be unretrieavable after removal."
