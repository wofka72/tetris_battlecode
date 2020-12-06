import random
import logging
import time
from copy import deepcopy
from typing import Text, List, Optional

from tetris_client import (
    Board,
    Element,
    GameClient,
    Point,
    TetrisAction,
)

logging.basicConfig(format="%(asctime)s %(levelname)s:%(message)s", level=logging.INFO)

ALL_ACTIONS = [x for x in TetrisAction if x.value != "act(0,0)"]

EMPTY = '.'
prev_cols_heights = None

ROTATION_COUNTS = {
    'O': 1,
    'I': 2,
    'S': 2,
    'Z': 2,
    'J': 4,
    'L': 4,
    'T': 4,
}

SCORE_FOR_HOLE = 10
MULTIPLIER_FULL_ROWS = -3
SCORES_FOR_FULL_ROWS = [0, 10, 30, 50, 100]

total_get_board_time = 0
total_get_score_time = 0
total_get_cols_heights_time = 0


def get_first_level_actions(gcb: Board) -> List[TetrisAction]:
    first_empty_p = Point(0, 0)

    while not first_empty_p.is_out_of_board():
        if gcb.get_element_at(first_empty_p).get_char() == EMPTY:
            break
        first_empty_p = first_empty_p.shift_right(1)

    fig_point = gcb.get_current_figure_point()
    diff = fig_point.get_x() - first_empty_p.get_x()

    actions = []

    if diff > 0:
        actions = [TetrisAction.LEFT] * diff
    if diff < 0:
        actions = [TetrisAction.RIGHT] * -diff

    actions.append(TetrisAction.DOWN)

    return actions


def start_debug_print(gcb: Board) -> None:
    elem = gcb.get_current_figure_type()
    print(gcb.get_future_figures())
    print(gcb.get_current_figure_point())
    print(gcb.get_current_figure_type())
    print(gcb.find_element(elem))

    # predict_figure_points_after_rotation - предсказывает положение фигуры после вращения
    print('rotate prediction: ', gcb.predict_figure_points_after_rotation(rotation=3))


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


def remove_full_rows(board: List[List[str]]) -> List[List[str]]:
    new_board = deepcopy(board)
    board_size = len(board)

    shift = 0
    for y in range(board_size):
        count_blocks = sum(new_board[y][x] != EMPTY for x in range(board_size))
        
        if count_blocks == 0:
            break

        if count_blocks == board_size:
            shift += 1
        else:
            new_board[y - shift] = new_board[y]

    for s in range(1, shift + 1):
        new_board[y - s] = [EMPTY for _ in range(board_size)]

    sub_score = MULTIPLIER_FULL_ROWS * SCORES_FOR_FULL_ROWS[shift]
    return new_board, sub_score


def update_board(board: List[List[str]], remove_points: List[Point], add_points: List[Point], add_char: Optional[str] = None) -> None:
    global total_get_board_time
    time_board_start = time.time()

    board_size = len(board)

    for p in remove_points:
        if not p.is_out_of_board(board_size):
            board[p.get_y()][p.get_x()] = EMPTY

    if add_points:
        assert isinstance(add_char, str) and len(add_char) == 1, 'add_char must be str of len 1'

    for p in add_points:
        if not p.is_out_of_board(board_size):
            board[p.get_y()][p.get_x()] = add_char

    time_board_end = time.time()
    total_get_board_time += time_board_end - time_board_start


def get_score(
    board: List[List[str]],
    final_cols_heights: List[int],
    cur_char: str
) -> int:

    global total_get_score_time
    time_score_start = time.time()

    board_size = len(board)

    emptyness_score = 0

    for x in range(board_size):
        count_empty = 0

        for y in range(board_size):
            cur_is_empty = board[y][x] == EMPTY
            if cur_is_empty:
                count_empty += 1
            else:
                emptyness_score += SCORE_FOR_HOLE * count_empty

    full_rows_score = 0
    heights_score = 0

    count_full_rows = 0
    for y in range(board_size):
        count_blocks = sum(board[y][x] != EMPTY for x in range(board_size))
        if count_blocks == 0:
            break

        if y > 5:
            heights_score += count_blocks * (y - 5) * (y - 5)

        if count_blocks == board_size:
            count_full_rows += 1

    full_rows_score = SCORES_FOR_FULL_ROWS[count_full_rows] * MULTIPLIER_FULL_ROWS
    heights_score = int(heights_score)

    heights_diff_score = 0

    for x in range(1, len(final_cols_heights)):
        diff = abs(final_cols_heights[x] - final_cols_heights[x - 1])
        heights_diff_score += diff * diff

    score = emptyness_score + full_rows_score + heights_diff_score + heights_score

    # print("coords:  {:36s}  ||  e: {:3d}  fr: {:3d}  hd: {:3d}  h: {:3d}  total: {:3d}".format(
        # str(final_coords), emptyness_score, full_rows_score, heights_diff_score, heights_score, score
    # ))

    time_score_end = time.time()
    total_get_score_time += time_score_end - time_score_start

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

    # actions = get_first_level_actions(gcb)
    # start_debug_print(gdb)

    global total_get_board_time
    global total_get_score_time
    global total_get_cols_heights_time

    start_time = time.time()
    total_get_board_time = 0
    total_get_score_time = 0
    total_get_cols_heights_time = 0

    cur_figure_p = gcb.get_current_figure_point()
    cur_figure_coords = gcb.get_current_element().get_all_coords(cur_figure_p)

    actions = [TetrisAction.DOWN]

    cur_el = gcb.get_current_element()
    cur_char = cur_el.get_char()

    future_figures = gcb.get_future_figures()
    next_el = Element(future_figures[0])
    next_char = next_el.get_char()

    base_board = get_board(gcb, cur_figure_coords, [])

    base_board_cols_heights = get_cols_heights(base_board)

    # print("base_board_cols_heights:  ", base_board_cols_heights)

    best_score = None
    best_shift = None
    best_rot = None

    for rot in range(ROTATION_COUNTS[cur_el.get_char()]):
        cur_rotate_coords = cur_el.get_all_coords_after_rotation(cur_figure_p, rot)

        min_x = min(p.get_x() for p in cur_rotate_coords)
        max_x = max(p.get_x() for p in cur_rotate_coords)

        shifts = list(range(-min_x, gcb._size - max_x))
        for shift in shifts:
            if shift < 0:
                pre_final_coords = [p.shift_left(-shift) for p in cur_rotate_coords]
            elif shift > 0:
                pre_final_coords = [p.shift_right(shift) for p in cur_rotate_coords]
            else:
                pre_final_coords = cur_rotate_coords

            pre_final_coords = shift_to_bottom(base_board_cols_heights, pre_final_coords)

            # add first figure
            update_board(base_board, [], pre_final_coords, cur_char)
            prefinal_board, sub_score = remove_full_rows(base_board)
            base_board_cols_heights_next = get_cols_heights_updated_up(base_board_cols_heights, pre_final_coords)

            for rot_next in range(ROTATION_COUNTS[next_el.get_char()]):
                cur_rotate_coords_next = next_el.get_all_coords_after_rotation(cur_figure_p, rot_next)

                min_x_next = min(p.get_x() for p in cur_rotate_coords_next)
                max_x_next = max(p.get_x() for p in cur_rotate_coords_next)

                shifts_next = list(range(-min_x_next, gcb._size - max_x_next))
                for shift_next in shifts_next:
                    if shift_next < 0:
                        final_coords = [p.shift_left(-shift_next) for p in cur_rotate_coords_next]
                    elif shift_next > 0:
                        final_coords = [p.shift_right(shift_next) for p in cur_rotate_coords_next]
                    else:
                        final_coords = cur_rotate_coords_next

                    final_coords = shift_to_bottom(base_board_cols_heights_next, final_coords)

                    # add final_coords to board
                    update_board(prefinal_board, [], final_coords, cur_char)
                    final_cols_heights = get_cols_heights_updated_up(base_board_cols_heights, final_coords)

                    score = sub_score + get_score(prefinal_board, final_cols_heights, next_char)

                    # remove final_coords from board
                    update_board(prefinal_board, final_coords, [])

                    if best_score is None or best_score > score:
                        best_score = score
                        best_shift = shift
                        best_rot = rot

            # remove first figure
            update_board(base_board, pre_final_coords, [])

    # prev_cols_heights = cols_heights

    print("BEST.  score: {},  shift: {},  rot: {}".format(best_score, best_shift, best_rot))
    actions = []

    if best_rot == 1:
        actions.append(TetrisAction.ACT)
    elif best_rot == 2:
        actions.append(TetrisAction.ACT_2)
    elif best_rot == 3:
        actions.append(TetrisAction.ACT_3)

    if best_shift > 0:
        actions.extend([TetrisAction.RIGHT] * best_shift)
    elif best_shift < 0:
        actions.extend([TetrisAction.LEFT] * -best_shift)

    actions.append(TetrisAction.DOWN)

    end_time = time.time()
    print("  TIME.  get_board: {:.6f}  get_score: {:.6f}  get_cols_heights: {:.6f}  total: {:.6f}".format(
        total_get_board_time, total_get_score_time, total_get_cols_heights_time, end_time - start_time),
    )

    return actions


def main(uri: Text):
    """
    uri: url for codebattle game
    """
    gcb = GameClient(uri)
    gcb.run(turn)


if __name__ == "__main__":
    # в uri переменную необходимо поместить url с игрой для своего пользователя
    # put your game url in the 'uri' path 1-to-1 as you get in UI

    uri = "http://{}/codenjoy-contest/board/player/{}?code={}&gameName=tetris".format(
        # "codebattle2020.westeurope.cloudapp.azure.com",
        # "rq5k7kulq3t78tcsvice",
        # "3502227942623102662",

        "localhost:8080",
        "ur4i4ozcsyjusmekp36y",
        "6681050020408655667",
    )

    main(uri)
