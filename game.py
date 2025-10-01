import numpy as np

KNIGHT_STEPS = [(1,2),(2,1),(2,-1),(1,-2),(-1,-2),(-2,-1),(-2,1),(-1,2)]

class BoardState:
    """
    Represents a state in the game
    """

    def __init__(self):
        """
        Initializes a fresh game state
        """
        self.N_ROWS = 8
        self.N_COLS = 7
        self.state = np.array([1,2,3,4,5,3,50,51,52,53,54,52])
        self.decode_state = [self.decode_single_pos(d) for d in self.state]

    def update(self, idx, val):
        """
        Updates both the encoded and decoded states
        """
        self.state[idx] = val
        self.decode_state[idx] = self.decode_single_pos(self.state[idx])

    def make_state(self):
        """
        Creates a new decoded state list from the existing state array
        """
        return [self.decode_single_pos(d) for d in self.state]

    def encode_single_pos(self, cr: tuple):
        """
        Encodes a single coordinate (col, row) -> Z

        Input: a tuple (col, row)
        Output: an integer in the interval [0, 55] inclusive

        DONE: You need to implement this.
        """
        c, r = cr
        if not (0 <= c < self.N_COLS and 0 <= r < self.N_ROWS):
            raise ValueError("encode_single_pos: coordinate out of bounds")
        return r * self.N_COLS + c

    def decode_single_pos(self, n: int):
        """
        Decodes a single integer into a coordinate on the board: Z -> (col, row)

        Input: an integer in the interval [0, 55] inclusive
        Output: a tuple (col, row)

        DONE: You need to implement this.
        """
        if not (0 <= int(n) < self.N_COLS * self.N_ROWS):
            n = int(n)
            c = max(0, min(self.N_COLS - 1, n % self.N_COLS))
            r = max(0, min(self.N_ROWS - 1, n // self.N_COLS))
            return (c, r)
        n = int(n)
        return (n % self.N_COLS, n // self.N_COLS)

    def is_termination_state(self):
        """
        Checks if the current state is a termination state. Termination occurs when
        one of the player's move their ball to the opposite side of the board.

        You can assume that `self.state` contains the current state of the board, so
        check whether self.state represents a termainal board state, and return True or False.
        
        DONE: You need to implement this.
        """
        if not self.is_valid():
            return False

        white_ball_enc = int(self.state[5])
        black_ball_enc = int(self.state[11])
        wc, wr = self.decode_single_pos(white_ball_enc)
        bc, br = self.decode_single_pos(black_ball_enc)
        return (wr == self.N_ROWS - 1) or (br == 0)

    def is_valid(self):
        """
        Checks if a board configuration is valid. This function checks whether the current
        value self.state represents a valid board configuration or not. This encodes and checks
        the various constrainsts that must always be satisfied in any valid board state during a game.

        If we give you a self.state array of 12 arbitrary integers, this function should indicate whether
        it represents a valid board configuration.

        Output: return True (if valid) or False (if not valid)
        
        DONE: You need to implement this.
        """
        s = self.state
        if len(s) != 12:
            return False
        if np.any((s < 0) | (s >= self.N_COLS * self.N_ROWS)):
            return False

        white_blocks = s[0:5].tolist()
        black_blocks = s[6:11].tolist()
        blocks = white_blocks + black_blocks
        if len(set(blocks)) != len(blocks):
            return False

        white_ball_ok = int(s[5]) in set(white_blocks)
        black_ball_ok = int(s[11]) in set(black_blocks)
        if not (white_ball_ok and black_ball_ok):
            return False

        return True

class Rules:

    @staticmethod
    def single_piece_actions(board_state, piece_idx):
        """
        Returns the set of possible actions for the given piece, assumed to be a valid piece located
        at piece_idx in the board_state.state.

        Inputs:
            - board_state, assumed to be a BoardState
            - piece_idx, assumed to be an index into board_state, identfying which piece we wish to
              enumerate the actions for.

        Output: an iterable (set or list or tuple) of integers which indicate the encoded positions
            that piece_idx can move to during this turn.
        
        DONE: You need to implement this.
        """
        s = board_state.state
        ncols, nrows = board_state.N_COLS, board_state.N_ROWS

        if piece_idx < 5:
            if int(s[5]) == int(s[piece_idx]):
                return set()
        elif 6 <= piece_idx <= 10:
            if int(s[11]) == int(s[piece_idx]):
                return set()

        occupied = set(s[0:5].tolist() + s[6:11].tolist())

        c, r = board_state.decode_single_pos(int(s[piece_idx]))
        moves = set()
        for dc, dr in KNIGHT_STEPS:
            nc, nr = c + dc, r + dr
            if Rules.bounded(nc, nr, ncols, nrows):
                enc = board_state.encode_single_pos((nc, nr))
                if enc not in occupied:
                    moves.add(enc)
        return moves

    @staticmethod
    def single_ball_actions(board_state, player_idx):
        """
        Returns the set of possible actions for moving the specified ball, assumed to be the
        valid ball for plater_idx  in the board_state

        Inputs:
            - board_state, assumed to be a BoardState
            - player_idx, either 0 or 1, to indicate which player's ball we are enumerating over
        
        Output: an iterable (set or list or tuple) of integers which indicate the encoded positions
            that player_idx's ball can move to during this turn.
        
        DONE: You need to implement this.
        """
        s = board_state.state
        offset = player_idx * 6
        my_blocks = s[offset:offset+5].tolist()
        ball_pos = int(s[offset+5])
        occupied = set(s[0:5].tolist() + s[6:11].tolist())

        adj = {b: set() for b in my_blocks}
        for i in range(len(my_blocks)):
            for j in range(len(my_blocks)):
                if i == j: 
                    continue
                a, b = my_blocks[i], my_blocks[j]
                if Rules.clear_line(board_state, a, b, occupied):
                    adj[a].add(b)

        reachable = set()
        frontier = [ball_pos]
        seen = {ball_pos}
        while frontier:
            cur = frontier.pop(0)
            for nxt in adj.get(cur, ()):
                if nxt not in seen:
                    seen.add(nxt)
                    frontier.append(nxt)
                    reachable.add(nxt)

        if ball_pos in reachable:
            reachable.remove(ball_pos)
        return reachable

    @staticmethod
    def clear_line(board_state: BoardState, src_enc: int, dst_enc: int, occupied: set) -> bool:
        if src_enc == dst_enc:
            return False
        sc, sr = board_state.decode_single_pos(src_enc)
        dc, dr = board_state.decode_single_pos(dst_enc)

        dc_diff = dc - sc
        dr_diff = dr - sr

        if not (dc_diff == 0 or dr_diff == 0 or abs(dc_diff) == abs(dr_diff)):
            return False

        step_c = (0 if dc_diff == 0 else (1 if dc_diff > 0 else -1))
        step_r = (0 if dr_diff == 0 else (1 if dr_diff > 0 else -1))

        c, r = sc + step_c, sr + step_r
        while (c, r) != (dc, dr):
            enc = board_state.encode_single_pos((c, r))
            if enc in occupied:
                return False
            c += step_c
            r += step_r
        return True

class GameSimulator:
    """
    Responsible for handling the game simulation
    """

    def __init__(self, players):
        self.game_state = BoardState()
        self.current_round = -1 ## The game starts on round 0; white's move on EVEN rounds; black's move on ODD rounds
        self.players = players

    def run(self):
        """
        Runs a game simulation
        """
        while not self.game_state.is_termination_state():
            ## Determine the round number, and the player who needs to move
            self.current_round += 1
            player_idx = self.current_round % 2
            ## For the player who needs to move, provide them with the current game state
            ## and then ask them to choose an action according to their policy
            action, value = self.players[player_idx].policy( self.game_state.make_state() )
            print(f"Round: {self.current_round} Player: {player_idx} State: {tuple(self.game_state.state)} Action: {action} Value: {value}")

            if not self.validate_action(action, player_idx):
                ## If an invalid action is provided, then the other player will be declared the winner
                if player_idx == 0:
                    return self.current_round, "BLACK", "White provided an invalid action"
                else:
                    return self.current_round, "WHITE", "Black probided an invalid action"

            ## Updates the game state
            self.update(action, player_idx)

        ## Player who moved last is the winner
        if player_idx == 0:
            return self.current_round, "WHITE", "No issues"
        else:
            return self.current_round, "BLACK", "No issues"

    def generate_valid_actions(self, player_idx: int):
        """
        Given a valid state, and a player's turn, generate the set of possible actions that player can take

        player_idx is either 0 or 1

        Input:
            - player_idx, which indicates the player that is moving this turn. This will help index into the
              current BoardState which is self.game_state
        Outputs:
            - a set of tuples (relative_idx, encoded position), each of which encodes an action. The set should include
              all possible actions that the player can take during this turn. relative_idx must be an
              integer on the interval [0, 5] inclusive. Given relative_idx and player_idx, the index for any
              piece in the boardstate can be obtained, so relative_idx is the index relative to current player's
              pieces. Pieces with relative index 0,1,2,3,4 are block pieces that like knights in chess, and
              relative index 5 is the player's ball piece.
            
        DONE: You need to implement this.
        """
        s = self.game_state
        assert player_idx in (0, 1)
        actions = set()
        offset = player_idx * 6

        for rel in range(5):
            dests = Rules.single_piece_actions(s, offset + rel)
            for enc in dests:
                actions.add((rel, int(enc)))

        for enc in Rules.single_ball_actions(s, player_idx):
            actions.add((5, int(enc)))

        return actions

    def validate_action(self, action: tuple, player_idx: int):
        """
        Checks whether or not the specified action can be taken from this state by the specified player

        Inputs:
            - action is a tuple (relative_idx, encoded position)
            - player_idx is an integer 0 or 1 representing the player that is moving this turn
            - self.game_state represents the current BoardState

        Output:
            - if the action is valid, return True
            - if the action is not valid, raise ValueError
        
        DONE: You need to implement this.
        """
        if not isinstance(action, tuple) or len(action) != 2:
            raise ValueError("Action must be a tuple (relative_idx, encoded_position).")
        rel, pos = action
        if not isinstance(rel, int) or rel < 0 or rel > 5:
            raise ValueError("relative_idx must be an integer in [0,5].")
        if not isinstance(pos, (int, np.integer)):
            raise ValueError("encoded_position must be an integer.")
        if pos < 0 or pos >= self.game_state.N_COLS * self.game_state.N_ROWS:
            raise ValueError("encoded_position out of bounds.")

        # Ensure current state is valid
        if not self.game_state.is_valid():
            raise ValueError("Current board state is invalid.")

        valid_actions = self.generate_valid_actions(player_idx)
        if action not in valid_actions:
            raise ValueError("Action is not legal for the current player and state.")
        return True

    def update(self, action: tuple, player_idx: int):
        """
        Uses a validated action and updates the game board state
        """
        offset_idx = player_idx * 6 ## Either 0 or 6
        idx, pos = action
        self.game_state.update(offset_idx + idx, pos)

    @staticmethod
    def bounded(c, r, ncols, nrows):
        return 0 <= c < ncols and 0 <= r < nrows
