# import sys
#
# if __name__ == "__main__":
#     s = sys.stdin.readline().strip()
#     t = sys.stdin.readline().strip()
#     print(s=='')
#     if sorted(s) != sorted(t):
#         print(-1)
#     else:
#         res = 0
#         dict_t = {k:v for v,k in enumerate(t)}
#         for i in range(len(s)):
#             if dict_t[s[i]] > i:
#                 res += 1
#         print(res)


import sys

if __name__ == "__main__":
    # 读取第一行的n
    n = int(sys.stdin.readline().strip())
    l = [int(i) for i in sys.stdin.readline().strip().split()]
    r = [int(i) for i in sys.stdin.readline().strip().split()]

    min_d = {}
    for li, ri in zip(l, r):
        for c in range(li, ri + 1):
            if c in min_d:
                min_d[c] += 1
            else:
                min_d[c] = 1

    abo = bot = 0
    for k, v in min_d.items():
        abo += k * v
        bot += v
    min_e = round(abo / bot, 6)
    print(min_e)