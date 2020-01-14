from math import log


class Node:
    def __init__(self, key):
        self.key = key
        self.parent = None
        self.child = None
        self.left = None
        self.right = None
        self.degree = 0
        self.mark = False


class FibonacciHeap:

    def __init__(self):
        self.root_list = None
        self.min_node = None
        self.nodes_amount = 0

    # insert new node in O(1) time
    def insert(self, key):
        n = Node(key)
        n.left = n.right = n
        self.merge_by_root_list(n)
        self.nodes_amount += 1
        if not self.min_node or n.key < self.min_node.key:
            self.min_node = n
        return n

    # get min node in O(1) time
    def get_min(self):
        return self.min_node.key

    def iteration(self, head):
        node = stop = head
        flag = 0
        while True:
            if flag == 1 and node == stop:
                break
            elif node == stop:
                flag = 1
            yield node
            node = node.right

    def extract_min(self):
        min_node = self.min_node
        if min_node:
            if min_node.child:
                children = [c for c in self.iteration(min_node.child)]
                for i in range(0, len(children)):
                    self.merge_by_root_list(children[i])
                    children[i].parent = None
            self.remove_from_root_list(min_node)
            if min_node == min_node.right:
                self.min_node = None
                self.root_list = None
            else:
                self.min_node = min_node.right
                self.consolidate()
            self.nodes_amount -= 1
        return min_node.key

    def decrease_key(self, node, key):
        if key > node.key:
            return None
        node.key, p = key, node.parent
        if p and node.key < p.key:
            self.cut(node, p)
            self.cascading_cut(p)
        if node.key < self.min_node.key:
            self.min_node = node

    def merge(self, another_heap):
        heap = FibonacciHeap()
        heap.root_list = self.root_list
        heap.min_node = self.min_node
        last_node = another_heap.root_list.left
        another_heap.root_list.left = heap.root_list.left
        heap.root_list.left.right = another_heap.root_list
        heap.root_list.left = last_node
        heap.root_list.left.right = heap.root_list
        if another_heap.min_node.key < heap.min_node.key:
            heap.min_node = another_heap.min_node
        heap.nodes_amount = self.nodes_amount + another_heap.nodes_amount
        return heap

    def consolidate(self):
        bound = [None for _ in range(int(log(self.nodes_amount) * 2))]
        nodes = [n for n in self.iteration(self.root_list)]
        for i in range(len(nodes)):
            n = nodes[i]
            d = n.degree
            while bound[d]:
                m = bound[d]
                if n.key > m.key:
                    n, m = m, n
                self.heap_link(m, n)
                bound[d] = None
                d += 1
            bound[d] = n

        for i in range(len(bound)):
            if bound[i] and bound[i].key < self.min_node.key:
                self.min_node = bound[i]

    def heap_link(self, m, n):
        self.remove_from_root_list(m)
        m.left = m.right = m
        self.merge_by_child_list(n, m)
        m.mark = False
        n.degree += 1
        m.parent = n

    def merge_by_root_list(self, node):
        if not self.root_list:
            self.root_list = node
        else:
            node.right = self.root_list.right
            node.left = self.root_list
            self.root_list.right.left = node
            self.root_list.right = node

    def merge_by_child_list(self, p, node):
        if not p.child:
            p.child = node
        else:
            node.right = p.child.right
            node.left = p.child
            p.child.right.left = node
            p.child.right = node

    def remove_from_root_list(self, node):
        if node == self.root_list:
            self.root_list = node.right
        node.left.right = node.right
        node.right.left = node.left

    def remove_from_child_list(self, p, node):
        if p.child == p.child.right:
            p.child = None
        elif p.child == node:
            p.child = node.right
            node.right.parent = p
        node.left.right = node.right
        node.right.left = node.left


f = FibonacciHeap()


f.insert(10)
f.insert(1)
f.insert(90)
f.insert(23)
f.insert(22)
f.insert(17)
f.insert(3)

print(f.get_min()) # 1
print(f.extract_min()) # 1
print(f.extract_min()) # 3
print(f.extract_min()) # 10

m = FibonacciHeap()
m.insert(8)
m.insert(19)
m.insert(12)

f = f.merge(m)

t = f.root_list.right
f.decrease_key(t, 8)

print([n.key for n in f.iteration(f.root_list)]) # [17, 8, 12, 19]

print(f.extract_min()) # 8
print(f.extract_min()) # 12
print(f.extract_min()) # 17
