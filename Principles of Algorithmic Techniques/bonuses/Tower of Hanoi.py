'''
thinking:
First, move the N-1 disks from A to B
Then, move the disk N from A to C
Finally, move the N-1 disks from B to C by the same solving procedure.
'''


def hanoi(n, a, b, c):
    if n == 1:
        print(a, '->', c)
    else:
        hanoi(n - 1, a, c, b)
        hanoi(1, a, b, c)
        hanoi(n - 1, b, a, c)


if __name__ == '__main__':
    n = int(input('Please enter the amount of disks: '))
    if n > 0:
        hanoi(n, 'A', 'B', 'C')
