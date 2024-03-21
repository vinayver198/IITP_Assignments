import random

def create_initial_state(state_values):
    """
    :param state_values: list of values containing number from 1 to 8 and 'B'
    :return: state matrix generated from shuffled state_values
    """
    random.shuffle(state_values)
    rows = 3
    cols = 3
    state_matrix = []
    print("New Initial state is : ")
    for row in range(rows):
        temp_row = state_values[row*3:(row+1)*3]
        state_matrix.append(temp_row)
    print(state_matrix)
    return state_matrix

def find_blank_tile(state):
    """
    This function iterates on every tile and gives the location of
    blank tile
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

def generate_states(state):
    """
    This function generate states based on information present in question.
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
    This function matrix to a tuple
    :param arr: matrix is input state matrix
    :return: tuple of tuple of rows of matrix
    """
    return tuple(map(tuple,matrix))

def dfs(start,target):
    """
    This function performs depth first search on input state matrix
    :param start: it is the initial state matrix
    :param target: target matrix to reach
    :return: prints the number of steps if target is reached else returns None
    """
    stack = []
    visited = set()

    stack.append((start,0))
    while stack:
        curr,steps = stack.pop(-1)
        if curr == target:
            print(f"DFS : State found with steps {steps}")
            return True

        visited.add(convert_arr_tuple(curr))
        # generate states by moving blank space
        new_states = generate_states(curr)

        for new_state in new_states:
            temp = convert_arr_tuple(new_state)
            if temp not in visited:
                stack.append((new_state,steps+1)) # increment the steps by 1 to know total steps taken to reach the target
    return False

def bfs(start,target):
    """
    This function performs breadth first search on input state matrix
    :param start:it is the initial state matrix
    :param target: target matrix to reach
    :return:prints the number of steps if target is reached else returns None
    """
    queue = []
    visited = set()

    queue.append((start,0)) # append the state and current steps to reach the target
    count = 0
    while queue:
        curr,steps = queue.pop(0)
        count+=1
        if curr == target:
            print(f"BFS : State found with steps {steps}")
            return True

        visited.add(convert_arr_tuple(curr))
        # generate states by moving blank space
        new_states = generate_states(curr)

        for new_state in new_states:
            temp = convert_arr_tuple(new_state)
            if temp not in visited:
                queue.append((new_state,steps+1)) # increment the steps by 1 to know total steps taken to reach the target
    return False

def main():
    state = [[3,2,1],[4,5,6],[8,7,'B']]
    target = [[1,2,3],[4,5,6],[7,8,'B']]
    state_values = [1,2,3,4,5,6,7,8,'B']

    '''
    First try for the state given in assignment , if it fails start with random state and 
    find the initial state till we reach the final state
    '''

    print("Initial State matrix is :")
    print(state)
    print("Target matrix is : ")
    print(target)

    bfs_out = bfs(state, target)
    dfs_out = dfs(state, target)
    if not bfs_out and not dfs_out:
        count =0
        flag = False
        while count<1001:
            initial_state_matrix = create_initial_state(state_values)
            print("Target to reach is ")
            print(target)
            bfs_out = bfs(initial_state_matrix,target)
            dfs_out = dfs(initial_state_matrix, target)
            if not bfs_out and not dfs_out:
                count+=1
            else:
                flag = True
                break
        if not flag:
            print("Solution not found after 1000 iterations")

main()
