import numpy as np

class Modelling:
    def __init__(self, init_grid=np.array([[0]]), c=2, h=1, dt=1):
        self.c = c
        self.h = h
        self.dt = dt
        # add grid at time = 0
        self.grid = np.array([init_grid])
    
    def next_grid(self, k):
        # compute grid in next time step (nested for loop)
        ngrid = np.array([[0]]) # initialize the next grid?

        for i in range(0, len(self.grid[k])):
            for j in range(0, len(self.grid[k][i])):
                # use the "update formula"
                print("hello")
        return np.array([ngrid])
    
    def solveEq(self, max):
        # for loop for time steps and updating grid
        k = 0

        while(k < max): # some stopping condition
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
    
    # set the initial parameters
    def set_param(self, c = 0, h = 0, dt = 0):
        if not c == 0:
            self.c = c
        if not h == 0:
            self.h = h
        if not dt == 0:
            self.dt = dt
    
    def check_stability(self): # add given parameters
        # check whether problem is stable with given parameters
        print("your life is not stable")

    def min_stability(self):
        # find minimal stable parameters, whatever that may mean
        for i in range(0,10):
            self.check_stability()
            print("test this for stability")

class Grid:
    def __init__(self):
        # class structure for Grids, might be helpful
        # could improve computational time


# create grid
grid = Grid()

# create instance of modelling class
instance = Modelling() # need to give the grid here like (grid)

# find minimum stability conditions
instance.min_stability()

# set the parameters
instance.set_param(h = 1) # can add c = ?, h = ?, dt = ?

# solve the wave equations numerically
instance.solveEq() # should take as parameter max time