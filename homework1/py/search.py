# search.py
# ---------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

"""
In search.py, you will implement generic search algorithms which are called
by Pacman agents (in searchAgents.py).
"""

import util
from util import heappush, heappop
class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
      """
      Returns the start state for the search problem
      """
      util.raiseNotDefined()

    def isGoalState(self, state):
      """
      state: Search state

      Returns True if and only if the state is a valid goal state
      """
      util.raiseNotDefined()

    def getSuccessors(self, state):
      """
      state: Search state

      For a given state, this should return a list of triples,
      (successor, action, stepCost), where 'successor' is a
      successor to the current state, 'action' is the action
      required to get there, and 'stepCost' is the incremental
      cost of expanding to that successor
      """
      util.raiseNotDefined()

    def getCostOfActions(self, actions):
      """
      actions: A list of actions to take

      This method returns the total cost of a particular sequence of actions.  The sequence must
      be composed of legal moves
      """
      util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other
    maze, the sequence of moves will be incorrect, so only use this for tinyMaze
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s,s,w,s,w,w,s,w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.
    Your search algorithm needs to return a list of actions that reaches
    the goal. Make sure that you implement the graph search version of DFS,
    which avoids expanding any already visited states. 
    Otherwise, your implementation may run infinitely!
    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:
    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """

    # This stack will store Tuple type values.
    # Each tuple contains ("location", "path").
    # "location" indicates what x, y value pacman is located currently according to x-y coordinate system.
    # "path" indicates a unique path from initial state to current state.
    #        It will concatenate every direction/action from initial state to current state.
    stack = util.Stack()

    # visited locations. It helps to avoid expanding any already visited states.
    visited = []

    # if initial state is goal state, then return.
    if problem.isGoalState(problem.getStartState()):
        return []

    # push initial state into stack
    state = (problem.getStartState(), [])
    stack.push(state)

    while (True):
        # if the stack is empty, it indicates we have reached the whole graph/tree, but we still can not find the goal state.
        if stack.isEmpty():
            return []

        # variable location is current location
        # variable path is a whole path from initial state to current path
        # The path that pacman needs to take. It will be returned after DFS algorithm.
        current_state = stack.pop()
        location = current_state[0]
        path = current_state[1]

        # add current location into visited list.
        visited.append(location)

        # Checking whether we have find the goal state
        if problem.isGoalState(location):
            print(path)
            return path

        # retrieve all possible locations and directions based on current state.
        successors = problem.getSuccessors(location)

        if successors:
            for successor in successors:
                # Checking whether the location has been visited.
                # If the location has not been visited, we push it into stack.
                if successor[0] not in visited:
                    # Since it is the "graph" search version, not the tree search version of each algorithm.
                    # There are some successor may also already exist in our Stack.
                    if successor[0] in (temp[0] for temp in stack.list):
                        break

                    # The successor_path will be unique path from initial state to current successor.
                    # Then push current successor location and path into stack.
                    successor_path = path + [successor[1]]
                    successor_state = (successor[0], successor_path)
                    stack.push(successor_state)


def breadthFirstSearch(problem):
    # The only difference between BFS and DFS is BFS will be implemented by data structure Queue.

    # This queue will store Tuple type values.
    # Each tuple contains ("location", "path").
    # "location" indicates what x, y value pacman is located currently according to x-y coordinate system.
    # "path" indicates a unique path from initial state to current state.
    #        It will concatenate every direction/action from initial state to current state.
    queue = util.Queue()

    # visited locations. It helps to avoid expanding any already visited states.
    visited = []

    # if initial state is goal state, then return.
    if problem.isGoalState(problem.getStartState()):
        return []

    # push initial state into queue
    state = (problem.getStartState(), [])
    queue.push(state)

    while (True):
        # if the queue is empty, it indicates we have reached the end of graph/tree, but we still can not find the goal state.
        if queue.isEmpty():
            return []

        # variable location is current location
        # variable path is a whole path from initial state to current path
        # The path that pacman needs to take.
        current_state = queue.pop()
        location = current_state[0]
        path = current_state[1]

        # add current location into visited list.
        visited.append(location)

        # Checking whether we have find the goal state
        if problem.isGoalState(location):
            print(path)
            return path

        # retrieve all possible locations and directions based on current state.
        successors = problem.getSuccessors(location)

        if successors:
            for successor in successors:
                # Checking whether the location has been visited.
                # If the location has not been visited, we push it into queue.
                if successor[0] not in visited:
                    # Since it is the "graph" search version, not the tree search version of each algorithm.
                    # There are some successor may also already exist in our Queue.
                    if successor[0] in (temp[0] for temp in queue.list):
                        break

                    # The successor_path will be unique from initial state to current successor.
                    # Then push current successor location and path into queue.
                    successor_path = path + [successor[1]]
                    successor_state = (successor[0], successor_path)
                    queue.push(successor_state)


def uniformCostSearch(problem):
    # UCS algorithm can be implemented by using PriorityQueue

    # This priorityQueue will store Tuple type values, and also an integer indicates priority for heap sort.
    # In UCS algorithm, the value of priority will the total cost from initial state to current state.
    # We only need to pop the smallest value of priority everytime.

    # Each tuple contains ("location", "path").
    # "location" indicates what x, y value pacman is located currently according to x-y coordinate system.
    # "path" indicates a unique path from initial state to current state.
    #        It will concatenate every direction/action from initial state to current state.
    priorityQueue = util.PriorityQueue()

    # visited locations. It helps to avoid expanding any already visited states.
    visited = []

    # if initial state is goal state, then return.
    if problem.isGoalState(problem.getStartState()):
        return []

    # push initial state into priorityQueue
    state = (problem.getStartState(), [])
    priorityQueue.push(state, 0)

    while (True):
        # if the priorityQueue is empty, it indicates we have reached the end of graph/tree, but we still can not find the goal state.
        if priorityQueue.isEmpty():
            return []

        # variable location is current location
        # variable path is a whole path from initial state to current path
        # The path that pacman needs to take.
        current_state = priorityQueue.pop()
        location = current_state[0]
        path = current_state[1]

        # add current location into visited list.
        visited.append(location)

        # Checking whether we have find the goal state
        if problem.isGoalState(location):
            print(path)
            return path

        # retrieve all possible locations and directions based on current state.
        successors = problem.getSuccessors(location)

        if successors:
            for successor in successors:
                # Checking whether the location has been visited.
                # If the location has not been visited, we push it into priorityQueue.
                if successor[0] not in visited:
                    # Since it is the "graph" search version, not the tree search version of each algorithm.
                    # There are some successor may also already exist in our PriorityQueue.
                    # If current successor is not exist in our PriorityQueue, we just simply add it into our PriorityQueue.
                    # If current successor exists in our PriorityQueue, we may need to re-calculate its total cost.
                    if successor[0] not in (temp[0] for temp in priorityQueue.heap):
                        # The successor_path will be unique from initial state to current successor.
                        # Then push current successor location and path into PriorityQueue.
                        # The priority will be the total cost from initial state to current successor.
                        successor_path = path + [successor[1]]
                        successor_total_cost = problem.getCostOfActions(successor_path)
                        successor_state = (successor[0], successor_path)
                        priorityQueue.push(successor_state, successor_total_cost)

                    else:
                        old_total_cost = 0
                        for temp in priorityQueue.heap:
                            if successor[0] == temp[2][0]:
                                old_total_cost = problem.getCostOfActions(temp[2][1])

                        # The successor_path will be unique from initial state to current successor.
                        # Then push current successor location and path into PriorityQueue.
                        # The priority will be the total cost from initial state to current successor.
                        successor_path = path + [successor[1]]
                        new_total_cost = problem.getCostOfActions(successor_path)

                        # If the new_total_cost is less than old_total_cost, we would update to new path and total_cost/priority.
                        if new_total_cost < old_total_cost:
                            successor_state = (successor[0], successor_path)
                            priorityQueue.update(successor_state, new_total_cost)



def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    # This stack will store Tuple type values.
    # Each tuple contains ("location", "path").
    # "location" indicates what x, y value pacman is located currently according to x-y coordinate system.
    # "path" indicates a unique path from initial state to current state.
    #        It will concatenate every direction/action from initial state to current state.
    priorityQueue = PriorityQueueWithHeuristic(problem, path_cost_heuristic_cost)

    # visited locations. It helps to avoid expanding any already visited states.
    visited = []

    # if initial state is goal state, then return.
    if problem.isGoalState(problem.getStartState()):
        return []

    # push initial state into PriorityQueue
    priorityQueue.push((problem.getStartState(), []), heuristic)

    while (True):
        # if the stack is empty, it indicates we have reached the whole graph/tree, but we still can not find the goal state.
        if priorityQueue.isEmpty():
            return []

        # variable location is current location
        # variable path is a whole path from initial state to current path
        # The path that pacman needs to take.
        current_state = priorityQueue.pop()
        location = current_state[0]
        path = current_state[1]

        # add current location into visited list.
        visited.append(location)

        # Checking whether we have find the goal state
        if problem.isGoalState(location):
            print(path)
            return path

        # retrieve all possible locations and directions based on current state.
        successors = problem.getSuccessors(location)

        if successors:
            for successor in successors:
                # Checking whether the location has been visited.
                # If the location has not been visited, we push it into stack.
                if successor[0] not in visited:
                    # Since it is the "graph" search version, not the tree search version of each algorithm.
                    # There are some successor may also already exist in our PriorityQueue.
                    if successor[0] in (temp[0] for temp in priorityQueue.heap):
                        break

                    # The successor_path will be unique path from initial state to current successor.
                    # Then push current successor location and path into stack.
                    successor_path = path + [successor[1]]
                    successor_state = (successor[0], successor_path)
                    priorityQueue.push(successor_state, heuristic)


class PriorityQueueWithHeuristic(util.PriorityQueue):
    def  __init__(self, problem, priorityFunction):
        "priorityFunction (item) -> priority"
        self.priorityFunction = priorityFunction      # store the priority function
        util.PriorityQueue.__init__(self)        # super-class initializer
        self.problem = problem
    def push(self, state, heuristic):
        "Adds an item to the queue with priority from the priority function"
        util.PriorityQueue.push(self, state, self.priorityFunction(self.problem, state, heuristic))


def path_cost_heuristic_cost(problem, state, heuristic):
    # This method will count (cost of path until now + estimated cost to the goal)
    # Calculate f(n) = g(n) + h(n)
    # g(n) = cost of path until now
    # h(n) = heuristic function

    g = problem.getCostOfActions(state[1])
    h = heuristic(state[0], problem)
    return g + h

# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
