import numpy as np
import math

class Grid:
    def __init__(self, grid=None, n=2, m=2, max_time=1):
        # class structure for Grids, might be helpful
        # could improve computational time
        self.grid = grid
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
        
        self.grid = initial
        for i in range(1, self.max_time):
            self.grid = np.append(self.grid, initial.copy(), axis = 0)


    # set a rectangle in the grid to 
    def set_rect(self, grid_num, i_init, j_init, x_size, y_size, value):
        if i_init + x_size > self.n or j_init + y_size > self.m:
            print("error in provided parameters. Rectangle does not fit.")
        elif grid_num > self.max_time:
            print("error in time step. There is not grid at this time.")
        else:
            for i in range(i_init, i_init + x_size):
                for j in range(j_init, j_init + y_size):
                    self.grid[grid_num][i][j] = value
    
    def set_unif_rect(self, grid_num, i_init, j_init, x_size, y_size,
                      value_max, value_min):
        if i_init + x_size > self.n or j_init + y_size > self.m:
            print("error in provided parameters. Rectangle does not fit.")
        elif grid_num > self.max_time:
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
    
    def set_rect_inc_dec(self, grid_num, i_init, j_init, x_s, y_s,
                         v_max, v_min = 0, inc = True, axis = True):
        if i_init + x_s > self.n or j_init + y_s > self.m:
            print("error in provided parameters. Rectangle does not fit.")
        elif grid_num > self.max_time:
            print("error in time step. There is not grid at this time.")
        else:
            step = (v_max - v_min)/max(x_s, y_s)

            for i in range(i_init, i_init + x_s):
                for j in range(j_init, j_init + y_s):
                    if axis:
                        distance = i - i_init
                    else:
                        distance = j - i_init

                    print(distance)
                    
                    if inc:
                        self.grid[grid_num][i][j] = v_min + distance*step
                    else:
                        self.grid[grid_num][i][j] = v_max - distance*step
    
    def set_circ_unif(self, grid_num, c_i, c_j, radius, v_max, v_min):
        if c_i > self.n or c_i > self.m:
            print("error in provided parameters. Rectangle does not fit.")
        elif grid_num > self.max_time:
            print("error in time step. There is not grid at this time.")
        else:
            diff = v_max - v_min
            
            for i in range(self.n):
                for j in range(self.m):
                    dist = math.sqrt(math.pow(i - c_i, 2) + math.pow(j - c_j, 2))

                    if dist <= radius:
                        dist = dist/radius
                        value = int((v_min + diff*dist)*100)
                        self.grid[grid_num][i][j] = value/100.0
    
    def print_grid(self):
        print(self.grid)


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
        grid_prev = self.grid.get_grid(k-1)
        grid_now = self.grid.get_grid(k)
        ngrid = self.grid.get_grid(k+1)

        for i in range(self.grid.n):
            for j in range(self.grid.m):
                # use the "update formula"
                1 + 1
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
        1+1

    def min_stability(self):
        # find minimal stable parameters, whatever that may mean
        for i in range(0,10):
            self.check_stability()
            1+1

# create grid
# parameters
#   grid: grid at time = 0
#   n: x dimension of grid
#   m: y dimension of grid
#   max_time: time up to which we are considering the model
size = 14
init_grid = np.array([np.full((size,size),3.0)])
grid = Grid(grid = init_grid, n = size, m = size, max_time = 5)
grid.set_circ_unif(0,3,7,5,3,0)
grid.initialize_grid()
grid.print_grid()

# create instance of modelling class
instance = Modelling() # need to give the grid here like (grid)

# find minimum stability conditions
instance.min_stability()

# set the parameters
instance.set_param(h = 1) # can add c = ?, h = ?, dt = ?

# solve the wave equations numerically
instance.solveEq() # should take as parameter max time