import numpy as np
import math

class Grid:
    def __init__(self, grid=0, n=2, m=2, max_time=1):
        # class structure for Grids, might be helpful
        # could improve computational time
        self.grid = None
        if not grid == 0:
            self.grid = np.array([grid])
        self.n = n
        self.m = m
        self.max_time = max_time
        self.initialize_grid()

    def get_grid(self, i):
        return self.grid[i-1].copy()
    
    def get_max(self):
        return self.max_time
    
    def set_grid(self, i, new_grid):
        self.grid[i-1] = new_grid
    
    def initialize_grid(self):
        initial = np.array([np.zeros((self.n,self.m))])
        if self.grid is None:
            self.grid = initial.copy()
        else:
            initial = np.array([self.grid[0]]).copy()
        
        for i in range(1, self.max_time):
            self.grid = np.append(self.grid, initial.copy(), axis = 0)


    # set a rectangle in the grid to 
    def set_rect(self, grid_num, i_init, j_init, x_size, y_size, value):
        if i_init + x_size > n or j_init + y_size > m:
            print("error in provided parameters. Rectangle does not fit.")
        elif grid_num > max_time:
            print("error in time step. There is not grid at this time.")
        else:
            for i in range(i_init, i_init + x_size):
                for j in range(j_init, j_init + y_size):
                    self.grid[grid_num][i][j] = value
    
    def set_unif_rect(self, grid_num, i_init, j_init, x_size, y_size, value_max, value_min):
        if i_init + x_size > n or j_init + y_size > m:
            print("error in provided parameters. Rectangle does not fit.")
        elif grid_num > max_time:
            print("error in time step. There is not grid at this time.")
        else:
            b_right = i_init + x_size
            b_up = j_init + y_size
            step = 2*(value_max - value_min)/max(x_size, y_size)

            for i in range(i_init, i_init + x_size):
                for j in range(j_init, j_init + y_size):
                    x_dist = max(math.abs(i - i_init), math.abs(i - b_right))
                    y_dist = max(math.abs(j - j_init), math.abs(j - b_up))
                    distance = min(x_dist, y_dist)

                    self.grid[grid_num][i][j] = value_max - (distance*step)
    
    def set_rect_inc_dec(self, grid_num, i_init, j_init, x_size, y_size, value_max, value_min = 0, inc = True, axis = True):
        if i_init + x_size > n or j_init + y_size > m:
            print("error in provided parameters. Rectangle does not fit.")
        elif grid_num > max_time:
            print("error in time step. There is not grid at this time.")
        else:
            step = (value_max - value_min)/max(x_size, y_size)

            for i in range(i_init, i_init + x_size):
                for j in range(j_init, j_init + y_size):
                    if axis:
                        distance = i - i_init
                    else:
                        distance = j - i_init
                    
                    if inc:
                        self.grid[grid_num][i][j] = value_min + distance*step
                    else:
                        self.grid[grid_num][i][j] = value_max - distance*step


class Modelling:
    def __init__(self, init_grid=Grid(), c=2, h=1, dt=1):
        self.c = c
        self.h = h
        self.dt = dt
        self.max = grid.get_max()
        # add grid at time = 0
        self.grid = grid
    
    def next_grid(self, k):
        # compute grid in next time step (nested for loop)
        print(k)
        print(self.max)
        grid_prev = self.grid.get_grid(k-1)
        grid_now = self.grid.get_grid(k)
        ngrid = self.grid.get_grid(k+1)

        for i in range(0, self.grid.n):
            for j in range(0, len(self.grid.m)):
                # use the "update formula"
                print("hello")
        self.grid.set_grid = ngrid
    
    def solveEq(self):
        # for loop for time steps and updating grid
        k = 1

        while(k < self.max): # some stopping condition
            # use next_grid to compute next step
            self.next_grid(k)
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