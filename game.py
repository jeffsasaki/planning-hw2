import numpy as np

# -------------------------------
# Board & Rules
# -------------------------------

class BoardState:
    """
    Represents a state in the game.
    Encoded state layout (length 12):
      indices 0..4  : white block pieces
      index   5     : white ball (must equal one of indices 0..4)
      indices 6..10 : black block pieces
      index   11    : black ball (must equal one of indices 6..10)
    Positions are encoded as integers 0..55 on a 7x8 grid where
    encode(col,row) = row*7 + col, with (0,0) at lower-left.
    """

    def __init__(self):
        self.N_ROWS = 8
        self.N_COLS = 7
        # initial configuration from the README
        self.state = np.array([1,2,3,4,5,3,50,51,52,53,54,52], dtype=int)
        self.decode_state = [self.decode_single_pos(d) for d in self.state]

    def update(self, idx, val):
        """Updates both the encoded and decoded states"""
        self.state[idx] = int(val)
        self.decode_state[idx] = self.decode_single_pos(self.state[idx])

    def make_state(self):
        """Creates a new decoded state list from the existing state array"""
        return [self.decode_single_pos(d) for d in self.state]

    # ---------- encoding helpers ----------

    def encode_single_pos(self, cr: tuple) -> int:
        """
        Encodes a single coordinate (col, row) -> Z
        Input: a tuple (col, row)
        Output: an integer in the interval [0, 55] inclusive
        """
        c, r = cr
        if not (0 <= c < self.N_COLS and 0 <= r < self.N_ROWS):
            raise ValueError("encode_single_pos: coordinate out of bounds")
        return r * self.N_COLS + c

    def decode_single_pos(self, n: int) -> tuple:
        """
        Decodes a single integer into a coordinate on the board: Z -> (col, row)
        Input: an integer in the interval [0, 55] inclusive
        Output: a tuple (col, row)
        """
        if not (0 <= int(n) < self.N_COLS * self.N_ROWS):
            # For robustness during partial/invalid tests, clamp to bounds but
            # the overall validity will be handled by is_valid().
            n = int(n)
            c = max(0, min(self.N_COLS - 1, n % self.N_COLS))
            r = max(0, min(self.N_ROWS - 1, n // self.N_COLS))
            return (c, r)
        n = int(n)
        return (n % self.N_COLS, n // self.N_COLS)

    # ---------- state checks ----------

    def _block_positions(self):
        """Return sets of encoded positions: (white_blocks, black_blocks)."""
        s = self.state.tolist()
        white_blocks = set(s[0:5])
        black_blocks = set(s[6:11])
        return white_blocks, black_blocks

    def is_termination_state(self) -> bool:
        """
        Termination occurs when:
          - White's ball reaches the top row (row == 7), OR
          - Black's ball reaches the bottom row (row == 0),
        and the state itself is valid.
        """
        if not self.is_valid():
            return False

        white_ball_enc = int(self.state[5])
        black_ball_enc = int(self.state[11])
        wc, wr = self.decode_single_pos(white_ball_enc)
        bc, br = self.decode_single_pos(black_ball_enc)
        return (wr == self.N_ROWS - 1) or (br == 0)

    def is_valid(self) -> bool:
        """
        A configuration is valid iff:
          - All 12 entries are integers in [0, 55]
          - The 10 block pieces (0..4 and 6..10) occupy distinct squares
          - Each ball equals the position of one of its own blocks
        """
        # bounds
        s = self.state
        if len(s) != 12:
            return False
        if np.any((s < 0) | (s >= self.N_COLS * self.N_ROWS)):
            return False

        # uniqueness of blocks
        white_blocks = s[0:5].tolist()
        black_blocks = s[6:11].tolist()
        blocks = white_blocks + black_blocks
        if len(set(blocks)) != len(blocks):
            return False

        # balls must be on one of their own blocks
        white_ball_ok = int(s[5]) in set(white_blocks)
        black_ball_ok = int(s[11]) in set(black_blocks)
        if not (white_ball_ok and black_ball_ok):
            return False

        return True


class Rules:
    """Enumerates legal moves for pieces and balls."""

    KNIGHT_STEPS = [(1,2),(2,1),(2,-1),(1,-2),(-1,-2),(-2,-1),(-2,1),(-1,2)]

    @staticmethod
    def _in_bounds(c, r, ncols, nrows):
        return 0 <= c < ncols and 0 <= r < nrows

    @staticmethod
    def single_piece_actions(board_state: BoardState, piece_idx: int):
        """
        Return encoded destinations for the given block piece (knight moves).
        Disallow: moves off board, into any occupied square, or moving a block
        currently holding its ball.
        """
        s = board_state.state
        ncols, nrows = board_state.N_COLS, board_state.N_ROWS

        # If the piece holds its color's ball, it cannot move
        if piece_idx < 5:
            if int(s[5]) == int(s[piece_idx]):
                return set()
        elif 6 <= piece_idx <= 10:
            if int(s[11]) == int(s[piece_idx]):
                return set()

        # occupied by any block (both colors)
        occupied = set(s[0:5].tolist() + s[6:11].tolist())

        c, r = board_state.decode_single_pos(int(s[piece_idx]))
        moves = set()
        for dc, dr in Rules.KNIGHT_STEPS:
            nc, nr = c + dc, r + dr
            if Rules._in_bounds(nc, nr, ncols, nrows):
                enc = board_state.encode_single_pos((nc, nr))
                if enc not in occupied:
                    moves.add(enc)
        return moves

    @staticmethod
    def _clear_los(board_state: BoardState, src_enc: int, dst_enc: int, occupied: set) -> bool:
        """Return True if line-of-sight (queen move) from src to dst is clear of blockers (excluding endpoints)."""
        if src_enc == dst_enc:
            return False
        sc, sr = board_state.decode_single_pos(src_enc)
        dc, dr = board_state.decode_single_pos(dst_enc)

        dc_diff = dc - sc
        dr_diff = dr - sr

        # must be queen-aligned
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

    @staticmethod
    def single_ball_actions(board_state: BoardState, player_idx: int):
        """
        Return encoded destinations (friendly block squares) that can end up
        holding the ball this turn, allowing unlimited chained passes along
        unobstructed queen lines.
        Excludes the current holder.
        """
        s = board_state.state
        offset = player_idx * 6
        my_blocks = s[offset:offset+5].tolist()
        ball_pos = int(s[offset+5])

        # blockers are all block pieces (both colors)
        occupied = set(s[0:5].tolist() + s[6:11].tolist())

        # Build adjacency among friendly blocks if line-of-sight is clear
        adj = {b: set() for b in my_blocks}
        for i in range(len(my_blocks)):
            for j in range(len(my_blocks)):
                if i == j: 
                    continue
                a, b = my_blocks[i], my_blocks[j]
                if Rules._clear_los(board_state, a, b, occupied):
                    adj[a].add(b)

        # BFS from current holder over this graph
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

        # Do not include the current holder
        if ball_pos in reachable:
            reachable.remove(ball_pos)
        return reachable


class GameSimulator:
    """Responsible for handling the game simulation / validation helpers used by search."""

    def __init__(self, players):
        self.game_state = BoardState()
        self.current_round = -1  # starts at round 0
        self.players = players

    def generate_valid_actions(self, player_idx: int):
        """
        Given a valid state and the player's turn, generate all actions:
        returns a set of (relative_idx, encoded_position).
        relative_idx 0..4 -> block piece indices; 5 -> ball pass.
        """
        s = self.game_state
        assert player_idx in (0, 1)
        actions = set()

        offset = player_idx * 6

        # block moves
        for rel in range(5):
            dests = Rules.single_piece_actions(s, offset + rel)
            for enc in dests:
                actions.add((rel, int(enc)))

        # ball passes
        for enc in Rules.single_ball_actions(s, player_idx):
            actions.add((5, int(enc)))

        return actions

    def validate_action(self, action: tuple, player_idx: int):
        """
        Return True if action is valid; otherwise raise ValueError with a descriptive reason.
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
        """Uses a validated action and updates the game board state"""
        offset_idx = player_idx * 6  # 0 or 6
        idx, pos = action
        self.game_state.update(offset_idx + idx, pos)
