'''
It's a divide and conquer algorithm.
First divide the current nums into two parts until the rest num amount of current nums is only one
Then merge the right and left nums to a new nums which in order.
recurse this procedure.
'''


def merge_sort(nums):
    if len(nums) < 2:
        return nums

    mid = len(nums) // 2

    left_part = merge_sort(nums[:mid])  # left part
    right_part = merge_sort(nums[mid:])

    # merge them
    i = j = k = 0

    while i < len(left_part) and j < len(right_part):
        if left_part[i] < right_part[j]:
            nums[k] = left_part[i]
            i += 1
        else:
            nums[k] = right_part[j]
            j += 1
        k += 1

    while i < len(left_part):
        nums[k] = left_part[i]
        i += 1
        k += 1

    while j < len(right_part):
        nums[k] = right_part[j]
        j += 1
        k += 1

    return nums


a = merge_sort([1, 3, 2])
print(a)
