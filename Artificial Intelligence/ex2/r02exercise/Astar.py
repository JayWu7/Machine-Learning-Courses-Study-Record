#
# Author: Jussi Rintanen, (C) Aalto University
# Only for student use on the Aalto course CS-E4800/CS-EJ4801.
# Do not redistribute.
#
# NOTE: Copy this file to 'Astar.py' before modifying.
#
# NOTE: It is recommended to only modify the block of the code
# indicated by "### Insert your code here ###"
#
# NOTE: Do not change the name of the class or the methods, as
# the automated grader relies on the names.
#
# Functions in classes representing state space search problems:
#   __init__    To create a state (a starting state for search)
#   __repr__    To construct a string that represents the state
#   __hash__    Hash function for states
#   __eq__      Equality for states
#   successors  Returns [(a1,s1,c1),...,(aN,sN,cN)] where each si is
#               the successor state when action called ai is taken,
#               and ci is the associated cost.
#               Here the name ai of an action is a string.
import time
import queue
import itertools
DEBUG = False
# DEBUG=True
# A*
# def ASTAR(initialstate, goaltest, h):
#     predecessor = dict()  # dictionary for predecessors
#     g = dict()  # dictionary for holding cost-so-far
#     if goaltest(initialstate):
#         print('Initial state is a goal state, terminating...')
#         return None
#     goal = None  # record the goal state
#     priority_Q = queue.Queue(maxsize=0)
#
#     g[str(initialstate)] = 0
#     predecessor[str(initialstate)] = initialstate
#
#     print('A* : Initial state is ' + str(initialstate))
#     priority_Q.put([h(initialstate), initialstate])
#
#     while not priority_Q.empty():
#         fq, n = priority_Q.get()
#         q_list = []
#         if goal and g[str(goal)] <= fq:
#             pass
#         else:
#             for aname, s, c in n.successors():
#                 if str(s) not in g or g[str(n)] + c < g[str(s)]:
#                     g[str(s)] = g[str(n)] + c
#                     predecessor[str(s)] = n
#                     if not goaltest(s):
#                         q_list.append([h(s) + g[str(s)], s])
#                     else:
#                         goal = s
#         while not priority_Q.empty():
#             q_list.append(priority_Q.get())
#         q_list.sort()
#         for item in q_list:
#             priority_Q.put(item)
#
#
#     plan, cost = [], g[str(goal)]
#     back_state = goal
#     while back_state != initialstate:
#         plan.append(back_state)
#         back_state = predecessor[str(back_state)]
#     return plan, cost
def ASTAR(initialstate, goaltest, h):
    predecessor = dict()  # dictionary for predecessors
    g = dict()  # dictionary for holding cost-so-far
    if goaltest(initialstate):
        print('Initial state is a goal state, terminating...')
        return None
    priority_Q = queue.PriorityQueue(maxsize=0)
    g[str(initialstate)] = 0
    predecessor[str(initialstate)] = None
    print('A* : Initial state is ' + str(initialstate))
    priority_Q.put((h(initialstate), initialstate))
    global best, goal
    plan, best = list(), float('inf')
    passed = {str(initialstate)}
    while not priority_Q.empty():
        fq, n = priority_Q.get()
        if best < fq:
            break

        for aname, s, c in n.successors():
            if str(s) not in g or g[str(n)] + c < g[str(s)]:
                g[str(s)] = g[str(n)] + c
                predecessor[str(s)] = n
                if goaltest(s):
                    goal = s
                    best = g[str(s)]
                elif str(s) not in passed:
                    priority_Q.put((h(s) + g[str(s)], s))
                    passed.add(str(s))
    back_state = goal
    while back_state:
        plan.append(back_state)
        back_state = predecessor[str(back_state)]
    print('My cost:{}'.format(best))
    return plan, best