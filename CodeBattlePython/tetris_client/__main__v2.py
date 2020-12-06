from copy import deepcopy
from dataclasses import dataclass
import logging
import random
import sys
import time
import traceback
from collections import defaultdict
from typing import Text, List, Optional, Dict

from tetris_client import (
    Board,
    Element,
    GameClient,
    Point,
    TetrisAction,
)


@dataclass
class State(object):
    board: List[List[str]]
    cols_heights: List[int]
    score: int
    fill_score: int
    first_shift: Optional[int]
    first_rotation: Optional[int]


logging.basicConfig(format="%(asctime)s %(levelname)s:%(message)s", level=logging.INFO)

ALL_ACTIONS = [x for x in TetrisAction if x.value != "act(0,0)"]

EMPTY = '.'

ROTATION_COUNTS = {
    'O': 1,
    'I': 2,
    'S': 2,
    'Z': 2,
    'J': 4,
    'L': 4,
    'T': 4,
}

SCORE_FOR_HOLE = 50
MULTIPLIER_FILLED_ROWS = -2
TOP_STATES_COUNT = 3
HEIGHTS_DIFFS_THRESHOLDS = [2, 4, 6]

# Table of scores for filled rows for different max heights.
# Table for board_size = 18
SCORES_FOR_FILLED_ROWS = [
    [0, -130,-100, -60, 1000],
    [0, -120, -90, -55, 1000],
    [0, -110, -80, -50, 1000],
    [0, -100, -70, -45, 1000],
    [0,  -90, -60, -40, 1000],
    [0,  -80, -50, -35, 1000],
    [0,  -70, -40, -30, 1000],
    [0,  -60, -35, -25, 1000],
    [0,  -50, -30, -20, 1000],
    [0,  -40, -25, -15, 1000],
    [0,  -30, -20, -10, 1000],
    [0,  -20, -10,   0, 1000],
    [0,  -10,   0,  30, 1000],
    [0,    0,  25,  60, 1000],
    [0,   20,  50,  90, 1000],
    [0,   40,  75, 120, 1000],
    [0,   60, 100, 150, 1000],
    [0,   80, 125, 180, 1000],
]

# timing
total_get_board_time = 0
total_get_score_time = 0
total_get_cols_heights_time = 0

# TEXT_LOST2 = """
# ████████████████████████████████████████████████████
# █────██────██───██────██────██─██─██───██─██─██────█
# █─██─██─██─███─███─██─██─██─██─██─██─████─██─██─██─█
# █─██─██─██─███─███────██────██────██───██────██─██─█
# █─██─██─██─███─███─█████─██─█████─██─████─██─██─██─█
# █─██─██────███─███─█████─██─█████─██───██─██─██────█
# ████████████████████████████████████████████████████
# """.strip()

TEXT_EXCEPTION = """
╔════╗─╔═══╗╔═══╗─╔════╗─╔════╗─╔═════╗─╔═════╗─╔═════╗─╔═════╗─╔═╗───╔═╗
║ ╔══╝─╚═╗ ║║ ╔═╝─║ ╔══╝─║ ╔══╝─║ ╔═╗ ║─╚═╗ ╔═╝─╚═╗ ╔═╝─║ ╔═╗ ║─║ ╚═╗─║ ║
║ ╚══╗───║ ╚╝ ║───║ ║────║ ╚══╗─║ ╚═╝ ║───║ ║─────║ ║───║ ║ ║ ║─║ ╔╗╚╗║ ║
║ ╔══╝───║ ╔╗ ║───║ ║────║ ╔══╝─║ ╔═══╝───║ ║─────║ ║───║ ║ ║ ║─║ ║╚╗╚╝ ║
║ ╚══╗─╔═╝ ║║ ╚═╗─║ ╚══╗─║ ╚══╗─║ ║───────║ ║───╔═╝ ╚═╗─║ ╚═╝ ║─║ ║─╚╗  ║
╚════╝─╚═══╝╚═══╝─╚════╝─╚════╝─╚═╝───────╚═╝───╚═════╝─╚═════╝─╚═╝──╚══╝
"""

TEXT_LOST = """
╔═════╗─╔════╗─╔═════╗─╔═════╗─╔═════╗─╔═╗─╔═╗─╔════╗─╔═╗─╔═╗─╔═════╗
║ ╔═╗ ║─║ ╔╗ ║─╚═╗ ╔═╝─║ ╔═╗ ║─║ ╔═╗ ║─║ ║─║ ║─║ ╔══╝─║ ║─║ ║─║ ╔═╗ ║
║ ║─║ ║─║ ║║ ║───║ ║───║ ╚═╝ ║─║ ╚═╝ ║─║ ╚═╝ ║─║ ╚══╗─║ ╚═╝ ║─║ ║ ║ ║
║ ║─║ ║─║ ║║ ║───║ ║───║ ╔═══╝─║ ╔═╗ ║─╚═══╗ ║─║ ╔══╝─║ ╔═╗ ║─║ ║ ║ ║
║ ║─║ ║─║ ╚╝ ║───║ ║───║ ║─────║ ║─║ ║─────║ ║─║ ╚══╗─║ ║─║ ║─║ ╚═╝ ║
╚═╝─╚═╝─╚════╝───╚═╝───╚═╝─────╚═╝─╚═╝─────╚═╝─╚════╝─╚═╝─╚═╝─╚═════╝
""".strip()


def get_cols_heights(board: List[List[str]]) -> List[int]:
    global total_get_cols_heights_time
    time_cols_heights_start = time.time()

    width = len(board[0])
    height = len(board)

    cols_heights = [0] * width

    for x in range(width):
        for y in range(height - 1, -1, -1):
            p = Point(x, y)

            if board[y][x] != EMPTY:
                cols_heights[x] = y + 1
                break

    time_cols_heights_end = time.time()
    total_get_cols_heights_time += time_cols_heights_end - time_cols_heights_start

    return cols_heights


def get_cols_heights_updated_up(base_cols_heights: List[int], add_points: List[Point]) -> List[int]:
    updated_cols_heights = base_cols_heights.copy()
    for p in add_points:
        updated_cols_heights[p.get_x()] = max(updated_cols_heights[p.get_x()], p.get_y() + 1)
    return updated_cols_heights


def print_custom_board(board: List[str]) -> None:
    print('\n'.join(board))


def shift_to_bottom(cols_heights: List[int], coords: List[Point]) -> List[Point]:
    bottom_shift = min(p.get_y() - cols_heights[p.get_x()] for p in coords)
    return [p.shift_bottom(bottom_shift) for p in coords]


def get_board(gcb: Board, remove_points: List[Point], add_points: List[Point], add_char: Optional[str] = None) -> List[List[str]]:
    global total_get_board_time
    time_board_start = time.time()

    board = [list(line) for line in reversed(gcb._line_by_line())]

    for p in remove_points:
        if not p.is_out_of_board(gcb._size):
            board[p.get_y()][p.get_x()] = EMPTY

    for p in add_points:
        if not p.is_out_of_board(gcb._size):
            board[p.get_y()][p.get_x()] = add_char

    time_board_end = time.time()
    total_get_board_time += time_board_end - time_board_start

    return board


def remove_filled_rows(board: List[List[str]], added_points_dict: Dict[int, int]) -> List[List[str]]:
    new_board = deepcopy(board)
    board_size = len(board)

    filled_rows_count = 0
    for y in range(board_size):
        count_blocks = sum(new_board[y][x] != EMPTY for x in range(board_size))

        if count_blocks == 0:
            break

        if count_blocks == board_size:
            if y in added_points_dict:
                del added_points_dict[y]

            filled_rows_count += 1
        else:
            if y in added_points_dict:
                added_points_dict[y - filled_rows_count] = added_points_dict[y]
                del added_points_dict[y]

            new_board[y - filled_rows_count] = new_board[y]

    for s in range(1, filled_rows_count + 1):
        new_board[y - s] = [EMPTY for _ in range(board_size)]

    return new_board, filled_rows_count


def update_board(board: List[List[str]], remove_points: List[Point], add_points: List[Point], add_char: Optional[str] = None) -> None:
    global total_get_board_time
    time_board_start = time.time()

    board_size = len(board)

    if any(p.is_out_of_board(board_size) for p in remove_points) or \
            any(p.is_out_of_board(board_size) for p in add_points):
        return False

    # TODO: check if add_point intersects with remove_points and add policies.
    for p in remove_points:
        board[p.get_y()][p.get_x()] = EMPTY

    for p in add_points:
        board[p.get_y()][p.get_x()] = add_char

    if add_points:
        assert isinstance(add_char, str) and len(add_char) == 1, 'add_char must be str of len 1'

    time_board_end = time.time()
    total_get_board_time += time_board_end - time_board_start

    return True


get_scores__calls_count = 0


def get_score(
    board: List[List[str]],
    final_cols_heights: List[int],
    added_points_dict: Dict[int, List[int]],
    cur_char: str
) -> int:

    global get_scores__calls_count
    get_scores__calls_count += 1

    global total_get_score_time
    time_score_start = time.time()

    board_size = len(board)

    # Height (turn) scores
    heights_score = 0

    for y, x in added_points_dict.items():
        if y > 5:
            heights_score += (y - 5) ** 2

    # Holes under blocks (board)
    holes_score = 0

    for x in range(board_size):
        count_empty = 0

        for y in range(final_cols_heights[x]):
            cur_is_empty = board[y][x] == EMPTY
            if cur_is_empty:
                count_empty += 1
            else:
                if y > 8:
                    heights_score += y - 8
                holes_score += SCORE_FOR_HOLE * count_empty
                # count_empty = 0 max(0, count_empty // 2)

    # Smoothness (board)
    heights_diff_score = 0
    
    # max_diff = max(final_cols_heights) - min(final_cols_heights)
    # if max_diff > 5:
    #     heights_diff_score += (max_diff - 5) ** 2

    for shift in range(1, len(HEIGHTS_DIFFS_THRESHOLDS) + 1):
        for x in range(shift, len(final_cols_heights)):
            diff = abs(final_cols_heights[x] - final_cols_heights[x - shift])
            if diff > HEIGHTS_DIFFS_THRESHOLDS[shift - 1]:
                heights_diff_score += diff * diff

    score = holes_score + heights_diff_score + heights_score

    time_score_end = time.time()
    total_get_score_time += time_score_end - time_score_start

    # print("coords:  {:36s}  ||  holes: {:3d}  h_diffs: {:3d}  heights: {:3d}  total: {:3d}".format(
    #     str(final_coords), emptyness_score, heights_diff_score, heights_score, score
    # ))

    return score


def turn(gcb: Board) -> List[TetrisAction]:
    # this function must return list actions from TetrisAction: tetris_client/internals/tetris_action.py
    #     LEFT = 'left'
    #     RIGHT = 'right'
    #     DOWN = 'down'
    #     ACT = 'act'
    #     ACT_2 = 'act(2)'
    #     ACT_3 = 'act(3),'
    #     ACT_0_0 = 'act(0,0)'
    # change return below to your code (right now its returns random aciton):
    # код ниже является примером и сэмплом для демонстрации - после подстановки корректного URI к своей игре
    # запустите клиент и посмотрите как отображаются изменения в UI игры и что приходит как ответ от API
    # for elem in dcb.

    global get_scores__calls_count
    get_scores__calls_count = 0

    global total_get_board_time
    global total_get_score_time
    global total_get_cols_heights_time

    start_time = time.time()

    total_get_board_time = 0
    total_get_score_time = 0
    total_get_cols_heights_time = 0

    elements = [gcb.get_current_element()] + [Element(c) for c in gcb.get_future_figures()]
    cur_figure_p = gcb.get_current_figure_point()

    base_board = get_board(gcb, elements[0].get_all_coords(cur_figure_p), [])
    prev__states = [State(
        board=base_board,
        cols_heights=get_cols_heights(base_board),
        score=0,
        fill_score=0,
        first_shift=None,
        first_rotation=None,
    )]

    board_size = gcb._size

    for cur__el in elements:
        cur__char = cur__el.get_char()

        cur__states: List[State] = []

        for prev__state in prev__states:
            prev_state__max_height = max(prev__state.cols_heights)

            for cur__rotation in range(ROTATION_COUNTS[cur__char]):
                cur__rotate_coords = cur__el.get_all_coords_after_rotation(cur_figure_p, cur__rotation)

                min_x = min(p.get_x() for p in cur__rotate_coords)
                max_x = max(p.get_x() for p in cur__rotate_coords)

                shifts = list(range(-min_x, board_size - max_x))
                for cur__shift in shifts:
                    if cur__shift < 0:
                        cur__final_coords = [p.shift_left(-cur__shift) for p in cur__rotate_coords]
                    elif cur__shift > 0:
                        cur__final_coords = [p.shift_right(cur__shift) for p in cur__rotate_coords]
                    else:
                        cur__final_coords = cur__rotate_coords

                    cur__final_coords = shift_to_bottom(prev__state.cols_heights, cur__final_coords)

                    # add first figure
                    if not update_board(prev__state.board, [], cur__final_coords, cur__char):
                        # invalid state
                        continue

                    # process state
                    added_points_dict = defaultdict(list)
                    for p in cur__final_coords:
                        added_points_dict[p.get_y()].append(p.get_x())

                    cur__board, cur__filled_rows_count = remove_filled_rows(prev__state.board, added_points_dict)
                    cur__cols_heights = get_cols_heights_updated_up(prev__state.cols_heights, cur__final_coords)
                    cur__cols_heights = [h - cur__filled_rows_count for h in cur__cols_heights]

                    cur__score = prev__state.fill_score + get_score(cur__board, cur__cols_heights, added_points_dict, cur__char)

                    cur__states.append(State(
                        board=cur__board,
                        cols_heights=cur__cols_heights,
                        score=cur__score,
                        fill_score=SCORES_FOR_FILLED_ROWS[prev_state__max_height - 1][cur__filled_rows_count] * MULTIPLIER_FILLED_ROWS,
                        first_shift=prev__state.first_shift if prev__state.first_shift is not None else cur__shift,
                        first_rotation=prev__state.first_rotation if prev__state.first_rotation is not None else cur__rotation,
                    ))

                    # remove first figure
                    update_board(prev__state.board, cur__final_coords, [])

        # i have a lot of states
        cur__states.sort(key=lambda state: state.score)

        prev__states = cur__states[:TOP_STATES_COUNT]

    if not cur__states:
        print(TEXT_LOST)
        return [TetrisAction.ACT_0_0]

    print("    get_scores  calls count: ", get_scores__calls_count)

    best_state = cur__states[0]

    actions = []
    if best_state.first_rotation == 1:
        actions.append(TetrisAction.ACT)
    elif best_state.first_rotation == 2:
        actions.append(TetrisAction.ACT_2)
    elif best_state.first_rotation == 3:
        actions.append(TetrisAction.ACT_3)

    if best_state.first_shift > 0:
        actions.extend([TetrisAction.RIGHT] * best_state.first_shift)
    elif best_state.first_shift < 0:
        actions.extend([TetrisAction.LEFT] * -best_state.first_shift)

    actions.append(TetrisAction.DOWN)

    end_time = time.time()
    print("  TIME.  get_board: {:.6f}  get_score: {:.6f}  get_cols_heights: {:.6f}  total: {:.6f}".format(
        total_get_board_time, total_get_score_time, total_get_cols_heights_time, end_time - start_time),
    )

    return actions


def wrap_turn(gcb: Board) -> List[TetrisAction]:
    try:
        return turn(gcb)
    except Exception as ex:
        print(traceback.format_exc(), file=sys.stderr)
        print(TEXT_EXCEPTION)
        return []


def main(uri: Text):
    """
    uri: url for codebattle game
    """
    gcb = GameClient(uri)
    gcb.run(wrap_turn)


if __name__ == "__main__":
    # в uri переменную необходимо поместить url с игрой для своего пользователя
    # put your game url in the 'uri' path 1-to-1 as you get in UI

    uri = "http://{}/codenjoy-contest/board/player/{}?code={}&gameName=tetris".format(
        # "codebattle2020.westeurope.cloudapp.azure.com",
        # "rq5k7kulq3t78tcsvice",
        # "3502227942623102662",

        "localhost:8080",
        # "ur4i4ozcsyjusmekp36y",
        # "6681050020408655667",
        "0",
        "000000000000",
    )

    main(uri)
