import random
import time

def find_tile_location(state,tile):
    '''
    This function finds the location of tile in input matrix
    :param state: input state matrix
    :param tile: tile number ; for this eg 1,2....8
    :return:
    '''
    for row in range(len(state)):
        for col in range(len(state[0])):
            if tile == state[row][col]:
                return [row,col]

def h1():
    '''
    This function return constant heuristic as 0
    :return: constant 0
    '''
    return 0

def h2(initial,goal):
    '''
    This function computes cost of number of misplaced tiles
    :param initial: start state matrix
    :param goal: target matrix
    :return: cost of misplaced tiles
    '''
    cost = 0
    for row in range(len(initial)):
        for col in range(len(initial[0])):
            if initial[row][col] != goal[row][col] and initial[row][col]!='B':
                cost+=1
    return cost

def h3(initial,goal):
    '''
    This function computes the manhattan distance
    :param initial:start state matrix
    :param goal:target matrix
    :return:total cost of manhattan distance tiles
    '''
    cost = 0
    for row in range(len(initial)):
        for col in range(len(initial[0])):
            if initial[row][col] != 'B':
                initial_tile_position = find_tile_location(initial,initial[row][col])
                goal_tile_position = find_tile_location(goal,initial[row][col])
                temp_cost = abs(initial_tile_position[0] - goal_tile_position[0]) + \
                            abs(initial_tile_position[1]-goal_tile_position[1])
                cost += temp_cost
    return cost


def h4(initial,goal):
    '''
    This function overestimates the actual cost
    :param initial: start state matrix
    :param goal:target matrix
    :return: overestimated cost
    '''
    return 10*h3(initial,goal)


def create_initial_state(state_values):
    """
    :param state_values: list of values
    containing number from 1 to 8 and 'B'
    :return: state matrix generated from shuffled state_values
    """
    random.shuffle(state_values)
    rows = 3
    cols = 3
    state_matrix = []
    for row in range(rows):
        temp_row = state_values[row*3:(row+1)*3]
        state_matrix.append(temp_row)
    return state_matrix

def find_blank_tile(state):
    """
    This function iterates on every tile and gives
    the location of blank tile
    :param state: state is input matrix
    :return: [row,col] row and col at which blank tile
            is present
    """
    rows = len(state)
    cols = len(state[0])
    for row in range(rows):
        for col in range(cols):
            if state[row][col] == 'B':
                return [row,col]

def generate_new_states(state):
    """
    This function generate new states based on information
    present in question.
    The blank can have 4 moves up,down,left and right
    :param state: state is input matrix
    :return: list of matrices by performing 4 moves
    """
    new_states = []  # output variable to store the generated states

    rows = len(state) # rows present in state matrix
    cols = len(state[0]) # cols present in state matrix

    # We can move either one block up,down, left or right
    moves_possible = [[-1,0],[1,0],[0,-1],[0,1]] # [up,down,left,right]

    # find the blank space represented as 'B'
    blank_space = find_blank_tile(state)   # here we get [row,col] of blank space

    # iterate over all the moves and find the states
    for move in moves_possible:
        new_state = [row[:] for row in state]
        new_row = blank_space[0]+move[0]
        new_col = blank_space[1]+move[1]
        if new_row >= 0 and new_row < rows and new_col >= 0 and new_col < cols:
            #swap the value of new row and col with new legal move
            temp = new_state[new_row][new_col]
            new_state[new_row][new_col] = new_state[blank_space[0]][blank_space[1]]
            new_state[blank_space[0]][blank_space[1]]=temp
            new_states.append(new_state)
    return new_states

def convert_arr_tuple(matrix):
    """
    This function convert matrix to a tuple
    :param arr: matrix is input state matrix
    :return: tuple of tuple of rows of matrix
    """
    return tuple(map(tuple,matrix))

def convert_tuple_arr(node):
    """
    This function converts tuple to a matrix
    :param arr: matrix is input state tuple
    :return: list of list of rows of tuple
    """
    return list(map(list,node))

def heuristic_map(heuristic,start,target):
    '''
    This function maps the start and target to right heuristic function
    :param heuristic: which heuristic is used 1,2,3or 4
    :param initial: start state matrix
    :param goal:target matrix
    :return:
    '''
    if heuristic == 1:
        return h1()
    elif heuristic == 2:
        return h2(start,target)
    elif heuristic == 3:
        return h3(start,target)
    else:
        return h4(start,target)

def check_monotonicity(path,heuristic,target,cost):
    '''
    This function checks the condition h(n) <= cost(n,m)+h(n)
    :param path: optimal a* path
    :param heuristic: which heuristic function to be used.
    :param target: target matrix
    :param cost: cost of nodes from n to m
    :return: None
    '''
    flag = True
    start_state = path.pop(0)
    for neighbor in path:
        if heuristic_map(heuristic,convert_tuple_arr(start_state),target) > \
                (cost[neighbor] + heuristic_map(heuristic,convert_tuple_arr(neighbor),target)):
            flag = False
            break
    if flag:
        print(f"Heuristic h({heuristic}) : Monotonicity check passed")
    else:
        print(f"Heuristic h({heuristic}) : Monotonicity check failed")
    return

def check_states_in_better_included_inferior(path1,path2):
    '''
    This function evaluates whether states of better heuristic
    are already explored by inferior heuristic
    :param path1:  closed list of inferior heuristic
    :param path2:  closed list of better heuristic
    :return:
    '''
    flag = True
    for node in path2:
        if node not in path1:
            flag = False
    if flag:
        print("Passed :All the states expanded by better heuristics should also be expanded by inferior heuristics")
    else:
        print("Failed :All the states expanded by better heuristics should also be expanded by inferior heuristics")
    return

def a_star_algorithm(start, target,heuristic=1):
    '''
    This function implements the a* algorithm.
    :param start: start of the search
    :param target: target matrix of search
    :param heuristic: which heuristic to use
    :return: None
    '''
    start_time = time.time()
    print(f"Start time : {start_time}")
    # This function keeps track of states which are found
    # and yet to be explored
    open_list = set([convert_arr_tuple(start)])
    # This function keeps track of states which are explored
    closed_list = set([])

    # This function keeps track of the cost
    cost = {}

    # Initialize the cost from start state
    cost[convert_arr_tuple(start)] = 0

    # parents contains an adjacency map of all nodes
    parents = {}
    parents[convert_arr_tuple(start)] = convert_arr_tuple(start)

    # count : is used to keep track of iterations taken to reach the goal state
    count = len(open_list)
    while len(open_list) > 0:
        # here n represents the current node
        current_node = None
        # find a node with the lowest value of f() - evaluation function
        for v in open_list:
            v_mat = convert_tuple_arr(v)
            if current_node == None or cost[v] + heuristic_map(heuristic,v_mat,target) <= cost[current_node] + heuristic_map(heuristic,convert_tuple_arr(current_node),target):
                current_node = v;

        if current_node == None or count==1e6:
            end_time = time.time()
            print(f'Failure. Path does not exist! and terminated after trying for iterations {count}')
            print("Start State : ")
            print(start)
            print("Target State")
            print(target)
            print(f"End time : {end_time}")
            print(f"Total time lapsed : {end_time - start_time} seconds")
            return None,None

        # if the current node is the stop_node
        # then we begin reconstructin the path from it to the start_node
        if current_node == convert_arr_tuple(target):
            print("Success")
            print("Start State : ")
            print(start)
            print("Target State")
            print(target)
            print(f"Total no of iterations needed to reach goal : {count}")
            print(f'Total number of steps needed to explore optimal path: {cost[current_node]}')
            reconst_path = []

            while parents[current_node] != current_node:
                reconst_path.append(current_node)
                current_node = parents[current_node]

            reconst_path.append(convert_arr_tuple(start))

            reconst_path.reverse()
            check_monotonicity(reconst_path, heuristic, target,cost)
            print(f"Iterations : {count}")
            print('Path found: {}'.format(reconst_path))
            end_time = time.time()
            print(f"End time : {end_time}")
            print(f"Total time lapsed : {end_time-start_time} seconds")
            return reconst_path, closed_list

        # for all neighbors of the current node do
        new_states= generate_new_states(convert_tuple_arr(current_node))
        for m in new_states:
            # if the current node isn't in both open_list and closed_list
            # add it to open_list and note n as it's parent
            m_tuple = convert_arr_tuple(m)
            if m_tuple not in open_list and m_tuple not in closed_list:
                open_list.add(m_tuple)
                parents[m_tuple] = current_node
                cost[m_tuple] = cost[current_node] + 1

            # otherwise, check if it's quicker to first visit n, then m
            # and if it is, update parent data and g data
            # and if the node was in the closed_list, move it to open_list
            else:
                if cost[m_tuple] > cost[current_node] + 1:
                    cost[m_tuple] = cost[current_node] + 1
                    parents[m_tuple] = current_node

                    if m_tuple in closed_list:
                        closed_list.remove(m_tuple)
                        open_list.add(m_tuple)

        # remove n from the open_list, and add it to closed_list
        # because all of his neighbors were inspected
        open_list.remove(current_node)
        closed_list.add(current_node)
        count+=1

def main():
    state = [[3,2,1],[4,5,6],[8,7,'B']]
    #state =  [[1,2,1],[4,5,6],[8,7,'B']]
    target = [[1,2,3],[4,5,6],[7,8,'B']]
    state_values = [1,2,3,4,5,6,7,8,'B']

    # First try for the given state matrix in assignment .
    # If it fails generate random matrix for state
    print("Solution for heuristic 1 :")
    recons_path1,closed1 = a_star_algorithm(state,target,1)
    print("Solution for heuristic 2 :")
    recons_path2,closed2 = a_star_algorithm(state,target,2)
    print("Solution for heuristic 3 :")
    recons_path3,closed3 = a_star_algorithm(state,target,3)
    print("\nSolution for heuristic 4 :")
    recons_path4,closed4 = a_star_algorithm(state,target,4)
    check_states_in_better_included_inferior(closed3,closed4)
    check_states_in_better_included_inferior(closed4, closed3)
    if not(recons_path1 and recons_path2 and recons_path3 and recons_path4):
        max_states_to_be_generated = 362880
        while max_states_to_be_generated:
                # generate new state matrix
                state = create_initial_state(state_values)
                print("Solution for heuristic 1 :")
                recons_path1,closed1 = a_star_algorithm(state,target,1)
                print("Solution for heuristic 2 :")
                recons_path2,closed2 = a_star_algorithm(state,target,2)
                print("Solution for heuristic 3 :")
                recons_path3,closed3 = a_star_algorithm(state,target,3)
                print("\nSolution for heuristic 4 :")
                recons_path4,closed4 = a_star_algorithm(state,target,4)
                check_states_in_better_included_inferior(closed3,closed4)
                if not(recons_path1 and recons_path2 and recons_path3 and recons_path4):
                    max_states_to_be_generated-=1
                else:
                    break
        if max_states_to_be_generated == 0:
            print(f"No solution found after trying generated states randomly for {max_states_to_be_generated} times")



heuristics = {1: 'h1', 2: 'h2', 3: 'h3', 4: 'h4'}
main()
