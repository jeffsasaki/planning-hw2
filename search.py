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

    def get_actions(self, state):
        """
        Returns a set of valid actions that can be taken from this state
        """
        pass

    def execute(self, state, action):
        """
        Transitions from the state to the next state that results from taking the action
        """
        pass

    def is_goal(self, state):
        """
        Checks if the state is a goal state in the set of goal states
        """
        return state in self.goal_state_set

class GameStateProblem(Problem):
    
    def __init__(self, initial_board_state, goal_board_state, player_idx):
        """
        player_idx is 0 or 1, depending on which player will be first to move from this initial state.

        Inputs for this constructor:
            - initial_board_state: an instance of BoardState
            - goal_board_state: an instance of BoardState
            - player_idx: an element from {0, 1}

        How Problem.initial_state and Problem.goal_state_set are represented:
            - initial_state: ((game board state tuple), player_idx ) <--- indicates state of board and who's turn it is to move
              ---specifically it is of the form: tuple( ( tuple(initial_board_state.state), player_idx ) )

            - goal_state_set: set([tuple((tuple(goal_board_state.state), 0)), tuple((tuple(goal_board_state.state), 1))])
              ---in otherwords, the goal_state_set allows the goal_board_state.state to be reached on either player 0 or player 1's
              turn.
        """
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

    def get_actions(self, state: tuple):
        """
        From the given state, provide the set possible actions that can be taken from the state
        Inputs: 
            state: (encoded_state, player_idx), where encoded_state is a tuple of 12 integers,
                and player_idx is the player that is moving this turn
        Outputs:
            returns a set of actions
        """
        s_enc, p = state
        self.sim.game_state.state = np.array(s_enc, dtype=int)
        self.sim.game_state.decode_state = self.sim.game_state.make_state()

        return self.sim.generate_valid_actions(p)

    def execute(self, state: tuple, action: tuple):
        """
        From the given state, executes the given action

        The action is given with respect to the current player

        Inputs: 
            state: is a tuple (encoded_state, player_idx), where encoded_state is a tuple of 12 integers,
                and player_idx is the player that is moving this turn
            action: (relative_idx, position), where relative_idx is an index into the encoded_state
                with respect to the player_idx, and position is the encoded position where the indexed piece should move to.
        Outputs:
            the next state tuple that results from taking action in state
        """
        s, p = state
        k, v = action
        offset_idx = p * 6
        return tuple((tuple( s[i] if i != offset_idx + k else v for i in range(len(s))), (p + 1) % 2))
 
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

    def is_goal(self, state):
        """
        Checks if the state is a goal state in the set of goal states
        """
        return state in self.goal_state_set