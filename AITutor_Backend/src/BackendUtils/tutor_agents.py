### implement LRU cache for tutors
from typing import Any
import heapq
import time


class TutorAgentNode(object):
    def __init__(self, uuid, agent):
        self.uuid = uuid
        self.agent = agent
        self.prev = None
        self.next = None

        self._time_used = time.time()

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self._time_used = time.time()
        return self.agent

    def __gt__(self, other: "TutorAgentNode"):
        return self._time_used > other._time_used

    def __lt__(self, other: "TutorAgentNode"):
        return self._time_used < other._time_used

    def __eq__(self, other: "TutorAgentNode"):
        return self._time_used > other._time_used


class LRUAgentCache:
    def __init__(self, capacity=32):
        self.capacity = capacity
        self.__agent_map = {}
        self.__agent_heap = []

    def add_agent(self, uuid, agent) -> TutorAgentNode | None:
        removal_agent = None
        if uuid in self.__agent_map:
            raise Exception(f"Internal Error, agent already exists under uuid: {uuid}")

        if len(self.__agent_map) >= self.capacity:  # Remove an Agent by LRU policy
            heapq.heapify(self.__agent_heap)
            # Remove the least recently used agent
            removal_agent = heapq.heappop(self.__agent_heap)
            del self.__agent_map[removal_agent.uuid]

        # Initialize new data and ptrs to the new_agent
        new_agent = TutorAgentNode(uuid, agent)
        self.__agent_map[uuid] = new_agent
        self.__agent_heap.append(new_agent)

        # Returns None or the Node pointing to the removed agent so that it can be stored back to memory:
        return removal_agent

    def get_agent(self, uuid: str) -> Any | None:
        if uuid in self.__agent_map:
            return self.__agent_map[
                uuid
            ]()  # Uses call method to get the value and update the time used

        return None
