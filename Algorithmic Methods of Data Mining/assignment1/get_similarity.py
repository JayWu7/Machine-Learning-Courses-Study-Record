'''
calculate the similarity of the given question of problem1.4
'''


def calculate(X, Y):
    similarity = 0
    for x, y in enumerate(X, Y):
        similarity += abs(x - y)
    return similarity / len(X)
