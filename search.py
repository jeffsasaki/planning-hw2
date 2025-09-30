import numpy as np
from collections import deque
from game import GameSimulator

class Problem:
    """
    Minimal interface: carry an initial state and a goal set.
    """
    def __init__(self, initial_state, goal_state_set: set):
        self.initial_state = initial_state
        self.goal_state_set = goal_state_set

class GameStateProblem(Problem):
    """
    Search over (encoded_state_tuple, player_idx) with unit-cost moves.
    """

    def __init__(self, initial_board_state, goal_board_state, player_idx):
        # Represent states as ((tuple_of_12_ints), player_idx)
        init_state = tuple((tuple(initial_board_state.state), int(player_idx)))
        goal_states = set([
            tuple((tuple(goal_board_state.state), 0)),
            tuple((tuple(goal_board_state.state), 1)),
        ])
        super().__init__(init_state, goal_states)

        self.sim = GameSimulator(None)
        self.search_alg_fnc = None
        self.set_search_alg()

    def set_search_alg(self, alg: str = ""):
        # For this assignment we expose a single optimal algorithm: BFS.
        self.search_alg_fnc = self.bfs

    # -------- helpers used by search --------

    def get_actions(self, state: tuple):
        """
        Return the set of legal actions from state (relative to the state's player).
        """
        s_enc, p = state
        # load simulator board to this state's board
        self.sim.game_state.state = np.array(s_enc, dtype=int)
        self.sim.game_state.decode_state = self.sim.game_state.make_state()
        return self.sim.generate_valid_actions(p)

    def execute(self, state: tuple, action: tuple):
        """
        Transition function: apply action to state and return next state.
        """
        s, p = state
        k, v = action
        offset_idx = p * 6
        next_state = tuple((
            tuple(s[i] if i != offset_idx + k else v for i in range(len(s))),
            (p + 1) % 2
        ))
        return next_state

    # --------- BFS (optimal in steps) ---------

    def bfs(self):
        """
        Breadth-first search over the turn-based state space.
        Returns a list of (state, action) pairs from start to goal, with
        the final pair having action=None.
        """
        start = self.initial_state
        if start in self.goal_state_set:
            return [(start, None)]

        q = deque([start])
        parent = {start: (None, None)}  # state -> (prev_state, action_taken_to_get_here)
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

        # Find a reached goal
        goal = None
        for g in self.goal_state_set:
            if g in parent:
                goal = g
                break

        if goal is None:
            # No solution (shouldn't happen in provided tests)
            return [(start, None)]

        # Reconstruct state sequence then attach actions-from-state
        states = []
        s = goal
        while s is not None:
            states.append(s)
            s = parent[s][0]
        states.reverse()

        result = []
        for i in range(len(states) - 1):
            # action stored on child leads from states[i] -> states[i+1]
            act = parent[states[i+1]][1]
            result.append((states[i], act))
        result.append((states[-1], None))
        return result
