from collections import deque
import numpy as np
import queue
from game import BoardState, GameSimulator, Rules

class Problem:
    """
    This is an interface which GameStateProblem implements.
    You will be using GameStateProblem in your code. Please see
    GameStateProblem for details on the format of the inputs and
    outputs.
    """

    def __init__(self, initial_state, goal_state_set: set):
        self.initial_state = initial_state
        self.goal_state_set = goal_state_set

class GameStateProblem(Problem):
    """
    Search over (encoded_state_tuple, player_idx) with unit-cost moves.
    """

    def __init__(self, initial_board_state, goal_board_state, player_idx):
        init_state = tuple((tuple(initial_board_state.state), int(player_idx)))
        goal_states = set([
            tuple((tuple(goal_board_state.state), 0)),
            tuple((tuple(goal_board_state.state), 1)),
        ])
        super().__init__(init_state, goal_states)

        self.sim = GameSimulator(None)
        self.search_alg_fnc = None
        self.set_search_alg()

    def set_search_alg(self, alg=""):
        """
        If you decide to implement several search algorithms, and you wish to switch between them,
        pass a string as a parameter to alg, and then set:
            self.search_alg_fnc = self.your_method
        to indicate which algorithm you'd like to run.

        DONE: You need to set self.search_alg_fnc here
        """
        self.search_alg_fnc = self.bfs

    def get_actions(self, state):
        """
        Returns a set of valid actions that can be taken from this state
        """
        s_enc, p = state
        self.sim.game_state.state = np.array(s_enc, dtype=int)
        self.sim.game_state.decode_state = self.sim.game_state.make_state()
        return self.sim.generate_valid_actions(p)

    def execute(self, state, action):
        """
        Transitions from the state to the next state that results from taking the action
        """
        s, p = state
        k, v = action
        offset_idx = p * 6
        next_state = tuple((
            tuple(s[i] if i != offset_idx + k else v for i in range(len(s))),
            (p + 1) % 2
        ))
        return next_state
    
    def is_goal(self, state):
        """
        Checks if the state is a goal state in the set of goal states
        """
        return state in self.goal_state_set

    def bfs(self):
        start = self.initial_state
        if start in self.goal_state_set:
            return [(start, None)]

        q = deque([start])
        parent = {start: (None, None)}
        visited = {start}

        while q:
            cur = q.popleft()
            if cur in self.goal_state_set:
                break

            for act in self.get_actions(cur):
                nxt = self.execute(cur, act)
                if nxt not in visited:
                    visited.add(nxt)
                    parent[nxt] = (cur, act)
                    q.append(nxt)

        goal = None
        for g in self.goal_state_set:
            if g in parent:
                goal = g
                break

        if goal is None:
            return [(start, None)]

        states = []
        s = goal
        while s is not None:
            states.append(s)
            s = parent[s][0]
        states.reverse()

        result = []
        for i in range(len(states) - 1):
            act = parent[states[i+1]][1]
            result.append((states[i], act))
        result.append((states[-1], None))
        return result
