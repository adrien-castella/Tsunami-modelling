import numpy as np

class Modelling:
    def __init__(self, init_grid=np.array([[0]]), c=2, h=1, dt=1):
        self.c = c
        self.h = h
        self.dt = dt
        self.grid = np.array([init_grid])
    
    def next_grid(self, k):
        # compute grid in next time step (nested for loop)
        ngrid = np.array([[0]]) # initialize the next grid?

        for i in range(0, len(self.grid[k])):
            for j in range(0, len(self.grid[k][i])):
                # use the "update formula"
                print("hello")
        return np.array([ngrid])
    
    def solveEq(self):
        # for loop for time steps and updating grid
        k = 0

        while(k < 10): # some stopping condition
            # use next_grid to compute next step
            ngrid = self.next_grid(k)
            print(self.grid)
            print(type(ngrid))
            self.grid = np.append(self.grid,
                        ngrid,
                        axis = 0)
            print(k) # not needed
            k = k + 1
        print("hello123")
    
    def set_param(self, c = 0, h = 0, dt = 0):
        if not c == 0:
            self.c = c
        if not h == 0:
            self.h = h
        if not dt == 0:
            self.dt = dt

instance = Modelling()
instance.set_param(h = 1)
instance.solveEq()