import numpy as np
import math
from math import pow

class Grid:
    # the __init__ method is called at the initialization of a class instance
    # it initializes all global parameters of the class
    # grid: a numpy list of the first x initial grids
    # n: the x? dimension of the grid size
    # m: the y? dimension of the grid size
    # max_time: the maximum number of time steps that will be taken
    # aka, the max size of the list
    # dec: indicates the the number of decimals after the point
    def __init__(self, grid=None, n=2, m=2, max_time=1, dec = 2):
        # class structure for Grids, might be helpful
        if grid is not None:
            self.grid = np.around(grid, dec)
        else:
            self.grid = grid
        self.n = n
        self.m = m
        self.max_time = max_time
        self.dec = dec
        # note this function call will force all grids in the list to be identical (initially)
        self.initialize_grid()

    def get_grid(self, i):
        return self.grid[i].copy()
    
    def get_max(self):
        return self.max_time

    def get(self, k, i, j):
        return self.grid[k][i][j].copy()
    
    def set_v(self, k, i, j, v):
        self.grid[k][i][j] = np.around(v,self.dec)
    
    def set_grid(self, i, new_grid):
        self.grid[i] = np.around(new_grid,self.dec)
    
    def set_max(self, max_t):
        self.max_time = max_t
    
    # initializes the list of grids. Makes a list of size max_time
    # composed of grids identical to the first grid
    # by default the grids are filled with zeros
    def initialize_grid(self):
        initial = np.array([np.zeros((self.n,self.m))])
        if self.grid is None:
            self.grid = initial.copy()
        else:
            initial = np.array([self.grid[0]]).copy()
        
        self.grid = initial
        for i in range(1, self.max_time):
            self.grid = np.append(self.grid, initial.copy(), axis = 0)

    # set a rectangular shaped portion of the grid at some time step to some value
    # grid_num: (integer) the time step which is being modified
    # i_init: the x?-coord of the rectangles top left corner (integer)
    # j_init: the y?-coord of the rectangles top left corner (integer)
    # x_size: the x-size of the rectangle (integer)
    # y_size: the y-size of the rectangle (integer)
    # value: the value being input into the rectangle
    # add: (boolean) whether the rectangle is replacing the current values
    #      or being added to them
    def set_rect(self, grid_num, i_init, j_init, x_size, y_size, value, add = False):
        if i_init + x_size > self.n or j_init + y_size > self.m:
            print("error in provided parameters. Rectangle does not fit.")
        elif grid_num > self.max_time:
            print("error in time step. There is not grid at this time.")
        else:
            for i in range(i_init, i_init + x_size):
                for j in range(j_init, j_init + y_size):
                    if add:
                        self.grid[grid_num][i][j] += np.around(value, self.dec)
                    else:
                        self.grid[grid_num][i][j] = np.around(value, self.dec)
    
    # in a rectangular portion of the grid have a linear (in/de)crease in values
    # from the center outwards uniformly
    # grid_num: (integer) the time step which is being modified
    # i_init: the x?-coord of the rectangles top left corner (integer)
    # j_init: the y?-coord of the rectangles top left corner (integer)
    # x_size: the x-size of the rectangle (integer)
    # y_size: the y-size of the rectangle (integer)
    # value_max: the maximum value being input into the rectangle
    # value_min: the minimum value being input into the rectangle
    # add: (boolean) whether the rectangle is replacing the current values
    #      or being added to them
    # inc: (boolean) whether the values should be increasing or decreasing outwards
    def set_unif_rect(self, grid_num, i_init, j_init, x_size, y_size,
                      value_max, value_min, add = False, inc = True):
        if i_init + x_size > self.n or j_init + y_size > self.m:
            print("error in provided parameters. Rectangle does not fit.")
        elif grid_num > self.max_time:
            print("error in time step. There is not grid at this time.")
        else:
            x_diff = (value_max - value_min)/(float(x_size)/2)
            y_diff = (value_max - value_min)/(float(y_size)/2)
            
            for i in range(i_init, i_init + x_size):
                for j in range(j_init, j_init + y_size):
                    dist = [[abs(i_init - i), abs(i_init - i + x_size)],
                            [abs(j_init - j), abs(j_init - j + y_size)]]
                    
                    value = 0
                    diff = 0
                    if (min(dist[0]) < min(dist[1])):
                        diff = x_diff
                        dist = min(dist[0])
                    else:
                        diff = y_diff
                        dist = min(dist[1])

                    if inc:
                        value = value_min + diff*dist
                    else:
                        value = value_max - diff*dist

                    if add:
                        self.grid[grid_num][i][j] += np.around(value, self.dec)
                    else:
                        self.grid[grid_num][i][j] = np.around(value, self.dec)
    
    # in a rectangular portion of the grid have a linear (in/de)crease in values
    # from the bottom-up
    # grid_num: (integer) the time step which is being modified
    # i_init: the x?-coord of the rectangles top left corner (integer)
    # j_init: the y?-coord of the rectangles top left corner (integer)
    # x_s: the x-size of the rectangle (integer)
    # y_s: the y-size of the rectangle (integer)
    # v_max: the maximum value being input into the rectangle (float)
    # v_min: the minimum value being input into the rectangle (float)
    # inc: (boolean) whether the values should be increasing or decreasing bottom-up
    # axis: (boolean) whether we increase in the x or the y direction
    # add: (boolean) whether the rectangle is replacing the current values
    #      or being added to them
    def set_rect_inc_dec(self, grid_num, i_init, j_init, x_s, y_s,
                         v_max, v_min = 0, inc = True, axis = True, add = False):
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
                    
                    value = 0
                    if inc:
                        value = (v_min + distance*step)
                    else:
                        value = (v_max - distance*step)

                    if add:
                        self.grid[grid_num][i][j] += np.around(value, self.dec)
                    else:
                        self.grid[grid_num][i][j] = np.around(value, self.dec)
    
    # in a circle portion of the grid have a linear (in/de)crease in values
    # from the center outwards
    # grid_num: (integer) the time step which is being modified
    # c_i: the x?-coord of the center of the circle (integer)
    # c_j: the y?-coord of the center of the circle (integer)
    # radius: the radius of the circle (float)
    # v_max: the maximum value being input into the circle (float)
    # v_min: the minimum value being input into the circle (float)
    # add: (boolean) whether the circle is replacing the current values
    #      or being added to them
    # inc: (boolean) whether the values should be increasing or decreasing outward
    def set_circ_unif(self, grid_num, c_i, c_j, radius, v_max, v_min,
                      add = False, inc = True):
        if c_i > self.n or c_j > self.m:
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
                        value = 0
                        if (inc):
                            value = v_min + diff*dist
                        else:
                            value = v_max - diff*dist
                        
                        if add:
                            self.grid[grid_num][i][j] += np.around(value, self.dec)
                        else:
                            self.grid[grid_num][i][j] = np.around(value, self.dec)
    
    # add the shape of a wave into the grid. Maximal values at r_c away
    # from the center and dissipating outward from the boundary to distance
    # r_d
    # grid_num: (integer) the time step which is being modified
    # c_i: the x?-coord of the center of the circle (integer)
    # c_j: the y?-coord of the center of the circle (integer)
    # r_c: the radius of the circle (float)
    # r_d: the radius of the cirlce boundary (float)
    # v_max: the maximum value being input into the wave (float)
    # v_min: the minimum value being input into the wave (float)
    # add: (boolean) whether the wave is replacing the current values
    #      or being added to them
    def set_wave(self, grid_num, c_i, c_j, r_c, r_d, v_max, v_min,
                 add = False):
        if c_i > self.n or c_j > self.m:
            print("error in provided parameters. Rectangle does not fit.")
        elif grid_num > self.max_time:
            print("error in time step. There is not grid at this time.")
        else:
            diff = v_max - v_min
            
            for i in range(self.n):
                for j in range(self.m):
                    dist = math.sqrt(math.pow(i - c_i, 2) + math.pow(j - c_j, 2))

                    if dist >= r_c-r_d and dist <= r_c+r_d:
                        dist = abs((dist - r_c)/r_d)
                        value = v_max - diff*dist
                        if not add:
                            self.grid[grid_num][i][j] = np.around(value, self.dec)
                        else:
                            self.grid[grid_num][i][j] += np.around(value, self.dec)

    # print the grids with indices in the set {n \in N : n \in [i,j)}
    # i: (integer) initial index
    # j: (integer) final index
    def print_grid(self, i = 0, j = None):
        if j is None:
            j = len(self.grid)
        for n in range(i, j):
            print(self.grid[n])


class Modelling:
    def __init__(self, init_grid=Grid(), c=2, h=1, dt=1):
        self.c = c
        self.h = h
        self.dt = dt
        self.max = grid.get_max()
        # add grid at time = 0
        self.grid = grid
    
    def next_grid(self, k):
        # need to set up boundary conditions here
        self.boundary(k+1)

        for i in range(1, self.grid.n-1):
            for j in range(1, self.grid.m-1):
                # use the "update formula"
                self.get_next(k,i,j)
    
    def boundary(self, k):
        print("ERROR: MISSING CODE HERE")
    
    # compute next step in i, j
    def get_next(self, k, i, j):
        c = pow(self.c * self.dt / self.h, 2)
        t1 = 2*self.grid.get(k,i,j) - self.grid.get(k-1,i,j)
        t2 = self.grid.get(k,i+1,j) - 2*self.grid.get(k,i,j) + self.grid.get(k,i-1,j)
        t3 = self.grid.get(k,i,j+1) - 2*self.grid.get(k,i,j) + self.grid.get(k,i,j-1)

        self.grid.set_v(k+1, i, j, t1 + c*t2 + c*t3)
    
    def solveEq(self):
        # for loop for time steps and updating grid
        k = 1
        self.grid.initialize_grid()

        while(k < self.max): # some stopping condition
            # use next_grid to compute next step
            self.next_grid(k-1)
            k = k + 1
        #print("hello123")
    
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
#   dec: number of decimals after the point
size = 16
init_grid = np.array([np.full((size,size),1.00234)])
grid = Grid(grid = init_grid, n = size, m = size, max_time = 15, dec = 2)

grid.set_rect(grid_num = 0, i_init = 2, j_init = 2, x_size = size-4,
         y_size = size-4, value = 0, add = False)
grid.set_unif_rect(grid_num = 1, i_init = 3, j_init = 3, x_size = 12,
                   y_size = 16, value_max = 5, value_min = 1,
                   add = True, inc = True)
grid.set_rect_inc_dec(grid_num = 2, i_init = 3, j_init = 3, x_s = 12,
                      y_s = 16, v_max = 6, v_min = 1, inc = True,
                      axis = False, add = False)
grid.set_circ_unif(grid_num = 3, c_i = 11, c_j = 11, radius = 7,
                   v_max = 5, v_min = 1, add = False, inc = False)
grid.set_wave(grid_num = 4, c_i = 12, c_j = 12, r_c = 5,
              r_d = 5, v_max = 5, v_min = 1, add = True)
grid.print_grid()

grid.set_grid(2, init_grid)

grid.set_rect(grid_num = 0, i_init = 0, j_init = 0, x_size = size,
              y_size = size, value = 0, add = False)
grid.initialize_grid()
grid.set_wave(grid_num = 0, c_i = 8, c_j = 8, r_c = 5, r_d = 2,
              v_max = 3, v_min = 0, add = True)

# create instance of modelling class
instance = Modelling(init_grid = grid) # need to give the grid here like (grid)

# find minimum stability conditions
instance.min_stability()

# set the parameters
instance.set_param(h = 0.1, dt = 0.01) # can add c = ?, h = ?, dt = ?

# solve the wave equations numerically
instance.solveEq() # should take as parameter max time

instance.grid.print_grid()