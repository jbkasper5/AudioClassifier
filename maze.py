import random
import collections
class Maze:
    def __init__(self, xdim, ydim):
        self.grid = [[((xdim * j) + i) for i in range(xdim)] for j in range(ydim)]
        self.al = AdjacencyList(num_vertices = (xdim * ydim))
        self.generate_maze(xdim = xdim, ydim = ydim)
        self.attempts = 0
        self.xdim = xdim
        while True:
            if self.is_valid(num_vertices = (xdim * ydim)):
                break
            else:
                self.generate_maze(xdim = xdim, ydim = ydim)

    def generate_maze(self, xdim, ydim):
        # RULES: for a given index, valid connections are index + 1, index - 1, index - xdim, index + xdim
        # Edge cases:
            # index is a multiple of xdim
            # index is either negative or greater than xdim * ydim
        while True:
            self.start = random.randint(0, ydim - 1) * xdim
            self.finish = random.randint(0, ydim - 1) * xdim + (xdim - 1)
            startValid = False
            finishValid = False
            if self.start < xdim or self.start % xdim == 0 or self.start + xdim > (xdim * ydim) or (self.start + 1) % xdim == 0:
                startValid = True
            if self.finish < xdim or self.finish % xdim == 0 or self.finish + xdim > (xdim * ydim) or (self.finish + 1) % xdim == 0:
                finishValid = True
            if(startValid and finishValid and self.start != self.finish):
                break

        for i in range(len(self.al.list)):
            if (i - xdim) > 0:
                if(random.random() < 0.29):
                    # print(f"Adding connection between {i} and {i - xdim}")
                    self.al.add(i, i - xdim)
            if (i + xdim < (xdim * ydim)):
                if(random.random() < 0.29):
                    # print(f"Adding connection between {i} and {i + xdim}")
                    self.al.add(i, i + xdim)
            if (i % xdim != 0 and i - 1 >= 0):
                if(random.random() < 0.29):
                    # print(f"Adding connection between {i} and {i - 1}")
                    self.al.add(i, i - 1)
            if ((i + 1) % xdim != 0 and i + 1 < (xdim * ydim)):
                if(random.random() < 0.29):
                    # print(f"Adding connection between {i} and {i + 1}")
                    self.al.add(i, i + 1)

    def is_valid(self, num_vertices) -> bool:
        self.visited = [False] * num_vertices
        self.deque = collections.deque()
        out = self.BFS(vertex = self.start)
        return out

    def BFS(self, vertex) -> bool:
        self.attempts += 1
        node = self.al.list[self.start]
        self.visited[vertex] = True
        if(node == None):
            return False
        while True:
            while(node != None):
                neighbor = node.vertex
                if neighbor == self.finish:
                    return True
                if self.visited[neighbor]:
                    node = node.next
                    continue
                else:
                    self.deque.append(neighbor)
                    node = node.next
                    continue
            if len(self.deque) == 0:
                return False
            else:
                new_neighbor = self.deque.popleft()
                self.visited[new_neighbor] = True
                node = self.al.list[new_neighbor]

    def print_al(self):
        conns = 0
        print(f"Start is {self.start} and finish is {self.finish}")
        for i in range(len(self.al.list)):
            if self.al.list[i] == None:
                print(f"Vertex {i} has no conenctions.")
            else:
                curr = self.al.list[i]
                while(curr != None):
                    print(f"Vertex {i} has a connection with vertex {curr.vertex}")
                    conns += 1
                    curr = curr.next
            print()
        print(f"There are {conns} connections")
                
    def get_connections(self, index):
        curr = self.al.list[index]
        conns = []
        while curr != None:
            if curr.vertex == index - 1:
                conns.append('l')
            elif curr.vertex == index + 1:
                conns.append('r')
            elif curr.vertex == index + self.xdim:
                conns.append('d')
            elif curr.vertex == index - self.xdim:
                conns.append('u')
            curr = curr.next
        return conns

    def go(self, index, direction) -> int:
        if direction == 'u':
            new_index = index - self.xdim
        elif direction == 'd':
            new_index = index + self.xdim
        elif direction == 'l':
            new_index = index - 1
        elif direction == 'r':
            new_index = index + 1
        
        if new_index == self.finish:
            return -1
        return new_index

class AdjacencyList:
    def __init__(self, num_vertices):
        self.list = [None] * num_vertices

    def add(self, v1, v2):
        self.add_connection(v1, v2)
        self.add_connection(v2, v1)

    def add_connection(self, v1, v2):
        curr = self.list[v1]
        if(curr == None):
            self.list[v1] = ALNode(v2)
            return
        elif curr.vertex == v2:
            return
        while(curr.next != None):
            if(curr.vertex == v2):
                return
            curr = curr.next

        if(curr.vertex == v2):
            return
        else:
            curr.next = ALNode(v2)
class ALNode:
    def __init__(self, vertex = -1):
        self.next = None
        self.vertex = vertex

