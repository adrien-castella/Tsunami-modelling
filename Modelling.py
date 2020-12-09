import numpy as np

class Modelling:
    def __init__(self, init_grid=np.array([[0]])):
        self.c = c
        self.h = h
        self.dt = dt
        self.grid = np.array([init_grid])
    
    def next_grid(self, k):
        # compute grid in next time step (nested for loop)
        ngrid = 0 # initialize the next grid?

        for i in range(0, length(self.grid[k])):
            for j in range(0, length(self.grid[k][i])):
                # use the "update formula"
                print("hello")
        return ngrid
    
    def solveEq(self):
        # for loop for time steps and updating grid
        k = 0

        while(True): # some stopping condition
            # use next_grid to compute next step
            self.grid = np.append(self.grid,
                        np.array[next_grid(k)],
                        axis = 0)
            print(k) # not needed
            k = k + 1
    
    def set_params(self, c = self.c, h = self.h, dt = self.dt):
        self.c = c
        self.h = h
        self.dt = dt
    

instance = Modelling()
instance.solveEq()