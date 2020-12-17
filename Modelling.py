import numpy as np
import math
from math import pow
import seaborn as sns
import matplotlib.pyplot as plt
import time

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
    
    def what_is_max(self):
        return np.amax(self.grid)

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
            print("error in time step. There is no grid at this time.")
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
    def set_unif_rect(self, grid_num, c_1, c_2, c_3, c_4, v_max, v_min, add = True, inc = True):
        rect = Polygon([c_1,c_2,c_3,c_4],1,v_min,v_max)

        for i in range(0,self.n):
            for j in range(0,self.m):
                #print("i: ", i, "; j: ", j)
                if (rect.contains(i,j)):
                    value  = 10
                    #rect.get_value(i,j)
                    if add:
                        self.grid[grid_num][i][j] += value
                    else:
                        self.grid[grid_num][i][j]  = value

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
            print("error in time step. There is no grid at this time.")
        else:
            if (axis):
                step = (v_max - v_min)/x_s
            else:
                step = (v_max - v_min)/y_s

            for i in range(i_init, i_init + x_s):
                for j in range(j_init, j_init + y_s):
                    if axis:
                        distance = math.sqrt(math.pow(i - i_init, 2))
                    else:
                        distance = math.sqrt(math.pow(j - j_init,2))
                    
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
    def set_circ_unif(self, grid_num, c_i, c_j, radius, v_max, v_min, c_min = 0,
                      add = False, inc = True):
        if c_i > self.n or c_j > self.m:
            print("error in provided parameters. Rectangle does not fit.")
        elif grid_num > self.max_time:
            print("error in time step. There is no grid at this time.")
        else:
            diff = v_max - v_min
            
            for i in range(self.n):
                for j in range(self.m):
                    dist = math.sqrt(math.pow(i - c_i, 2) + math.pow(j - c_j, 2))

                    if (dist < radius):
                        value = 0
                        if dist > c_min and dist <= radius:
                            dist_new = (dist-c_min)/(radius - c_min)
                            if (inc):
                                value = v_min + diff*dist_new
                            else:
                                value = v_max - diff*dist_new
                        elif dist <= c_min:
                            if (inc):
                                value = v_min
                            else:
                                value = v_max
                            
                        if add:
                            self.grid[grid_num][i][j] += np.around(value, self.dec)
                        elif radius <= radius:
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
    def set_wave(self, grid_num, c_i, c_j, R, h, c=1, add = False):
        c = float(c)/6.283
        if c_i > self.n or c_j > self.m:
            print("error in provided parameters. Rectangle does not fit.")
        elif grid_num > self.max_time:
            print("error in time step. There is no grid at this time.")
        else:
            for i in range(self.n):
                for j in range(self.m):
                    dist = math.sqrt(math.pow(i - c_i, 2) + math.pow(j - c_j, 2))
                    if dist < R + (1/2) + (math.pi/2):
                        value = self.wave_function(dist/c, R/c, h)
                        if not add:
                            self.grid[grid_num][i][j] = np.around(value, self.dec)
                        else:
                            self.grid[grid_num][i][j] += np.around(value, self.dec)
    
    #def set_triangle(self, grid_num, p_1, p_2, p_3, )

    # print the grids with indices in the set {n \in N : n \in [i,j)}
    # i: (integer) initial index
    # j: (integer) final index
    def print_grid(self, i = 0, j = None):
        if j is None:
            j = len(self.grid)
        for n in range(i, j):
            print(self.grid[n])

    def plot_grid(self, k, name, vmin=0, vmax=5000,center=0):
        ax = sns.heatmap(self.grid[k], vmin = vmin, vmax = vmax, center = center,
                         square = True)#, xticklabels = False, yticklabels = False,
                         #cbar = False)
        plt.savefig(name+".png")
        plt.clf()
    
    def wave_function(self, x, R, h):
        if x < R:
            a = x - R - (math.cos(x-R)/2)
            return math.cos(a)*(x/R)*h
        elif x >= R and x < R + (1/2):
            a = x - R - (1/2)
            return math.cos(a)*h
        elif x >= R + (1/2) and x < R + (1/2) + math.pi:
            a = 2*(x - R - (1/2))
            return math.cos(a)*(h/2) + (h/2)
        else:
            return 0
    
    def magnitude(self, v):
        summation = 0
        for i in v:
            summation += pow(i,2)
        return math.sqrt(summation)

class Polygon:
    # corners: corners of a polygon
    # option: 1,2, or 3. Uniform, increasing, or constant colouring
    # side: Applies to option 2. Increasing as you approach which side?
    # side 1 is segment between corners 1 -> 2, analogously for the rest
    def __init__(self, corners=[[0,0],[0,1],[1,0],[1,1]], option=1, v_min=0, v_max=1, side=1):
        self.corners = corners
        self.option = option
        self.side = side
        self.v_min = v_min
        self.v_max = v_max
    
    # returns whether the point is contained inside the polygon
    def contains(self, x, y):
        #print("x and y: ",x, " and ", y)
        extreme = [math.inf, y]
        count = 0
        n = len(self.corners)

        for i in range(0,n):
            next = (i+1)%n
            if self.check_intersect(self.corners[i], self.corners[next], [x,y], extreme):
                if self.orientation(self.corners[i], [x,y], self.corners[next]) == 0:
                    return self.onSegment(self.corners[i], [x,y], self.corners[next])
                
                count += 1
        #print("Counter: ", count)
        #print(count % 2)
        return count % 2 == 1
    
    # gets distance from nearst segment
    # returns -1 if it is not contained in the polygon
    def distance(self, x, y):
        dist = np.array([])
        n = len(self.corners)
        for i in range(n):
            dist = np.append(dist, [self.distance_to(self.corners[i], self.corners[(i+1)%n], [x,y])])
        return [min(dist), 1]
    
    # distance to segment c_1 -> c_2
    def distance_to(self, c_1, c_2, pt):
        a = c_1[0]; b = c_1[1]
        c = c_2[0] - c_1[0]; d = c_2[1] - c_1[1]
        term1 = d*pt[0] - c*pt[1] - a*d + b*c
        term2 = math.sqrt(pow(d,2) + pow(c,2))
        return abs(term1)/term2
    
    # gets the value of the given point based on the chosen strategy (option)
    def get_value(self, x, y):
        if self.option == 1:
            dist = self.distance(x, y)
            step = self.get_step(dist[1])
            if step <= 0:
                print("Error, step size too small")
                exit(1)
            return self.v_min + (dist[0]/step)*(self.v_max - self.v_min)
        elif self.option == 2:
            dist = 0
            if self.side == len(corners)-1:
                dist = self.distance_to(corners[self.side], corners[0], [x,y])
            elif self.side < len(corners)-1:
                dist = self.distance_to(corners[self.side], corners[self.side+1], [x,y])
            else:
                print("Invalid side!")
                exit(1)
            step = self.get_step(self.side)
            if step <= 0:
                print("Error, step size too small")
                exit(1)
            return self.v_min + (dist/step)*(self.v_max - self.v_min)
        else:
            return self.v_max
        
    def get_step(self, i):
        vec = np.subtract(self.corners[i-1], self.corners[i])
        mag = math.sqrt(pow(vec[0],2) + pow(vec[1],2))
        return mag/2
    
    def orientation(self, p, q, r):
        a = q[1] - p[1]; b = r[0] - q[0]
        c = q[0] - p[0]; d = r[1] - q[1]
        value = 0
        if (not a == 0) and (not b == 0):
            value = a*b
        if (not c == 0) and (not d == 0):
            if value == c*d:
                value = 0
            else:
                value = value - c*d
        
        if value == 0:
            return 0
        elif value > 0:
            return 1
        return 2
    
    def onSegment(self, p, q, r):
        if (q[0] <= max(p[0], r[0]) and q[0] >= min(p[0], r[0]) and 
            q[1] <= max(p[1], r[1]) and q[1] >= min(p[1], r[1])):
            return True
        return False
    
    def check_intersect(self, p1, q1, p2, q2):
        o1 = self.orientation(p1, q1, p2)
        o2 = self.orientation(p1, q1, q2)
        o3 = self.orientation(p2, q2, p1)
        o4 = self.orientation(p2, q2, q1)

        if (o1 != o2) and (o3 != o4):
            return True
        
        if o1 == 0 and self.onSegment(p1, p2, q1):
            return True
        
        if o2 == 0 and self.onSegment(p1, q2, q1):
            return True
        
        if o3 == 0 and self.onSegment(p2, p1, q2):
            return True
        
        if o4 == 0 and self.onSegment(p2, q1, q2):
            return True
        
        return False

class Modelling:
    def __init__(self, init_grid=Grid(), d=Grid(), h=1, dt=1):
        self.d = d
        self.h = h
        self.dt = dt
        self.max = grid.get_max()
        # add grid at time = 0
        self.grid = grid
    
    def next_grid(self, k):
        # need to set up boundary conditions here
        for i in range(0, self.grid.n-1):
            for j in range(0, self.grid.m-1):
                # use the "update formula"
                self.get_next(k,i,j)
    
    def get_elmt(self, k, i, j):
        if (i > self.grid.n):
            return self.grid.get(k,i-1,j)
        elif (j > self.grid.m):
            return self.grid.get(k,i,j-1)
        elif (i < 0):
            return self.grid.get(k,1,j)
        elif (j < 0):
            return self.grid.get(k,i,1)
        
        return self.grid.get(k,i,j)
    
    # compute next step in i, j
    def get_next(self, k, i, j):
        c = math.sqrt(9.81*self.d.get(0,i,j))
        term = pow(c * self.dt / self.h, 2)
        t1 = 2*self.grid.get(k,i,j) - self.grid.get(k-1,i,j)
        t2 = self.get_elmt(k,i+1,j) - 2*self.grid.get(k,i,j) + self.get_elmt(k,i-1,j)
        t3 = self.get_elmt(k,i,j+1) - 2*self.grid.get(k,i,j) + self.get_elmt(k,i,j-1)
        
        self.grid.set_v(k+1, i, j, t1 + term*t2 + term*t3)
    
    def solveEq(self):
        # for loop for time steps and updating grid
        k = 1
        self.grid.initialize_grid()

        for k in range (1, self.max): # some stopping condition
            # use next_grid to compute next step
            self.next_grid(k-1)
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
    
    def plot_result(self, name, vmax = 10, vmin = -10, step = 1):
        k = 0
        while k < self.max:
            self.grid.plot_grid(k,name+str(k),vmin,vmax)
            k += step
        self.grid.plot_grid(self.max-1, name+str(self.max-1), vmin, vmax)


# create grid
# parameters
#   grid: grid at time = 0
#   n: x dimension of grid
#   m: y dimension of grid
#   max_time: time up to which we are considering the model
#   dec: number of decimals after the point
size_x = 50
size_y = 50
t0 = time.time()
init_topo = np.array([np.full((size_x,size_y),5000)])
topography = Grid(grid = init_topo, n = size_x, m = size_y, max_time = 1, dec = 0)
#topography.set_unif_rect(0,[1,1],[1,10],[5,10],[5,1],v_max=2500,v_min=0,add=False)
topography.set_unif_rect(0,[5,5],[15,45],[35,40],[25,0],v_max=2500,v_min=0,add=False)
#topography.set_unif_rect(0,[300,300],[250,70],[230,70],[300,70],v_min=200,v_max=0,add=False)

#topography.set_unif_rect(grid_num=0,c_1=[0,0],c_2=[0,80],c_3=[100,0],c_4=[100,80],v_max=5000,v_min=0,add=False)
#topography.plot_grid(0,"topo1",0,5000,center=4800)

#topography.set_circ_unif(grid_num=0,c_i=55, c_j=120, radius=50,v_max=5000,v_min=0,c_min=30,add=False,inc=False)
#topography.set_rect(grid_num=0,i_init=0,j_init=0,x_size=45,y_size=300,value=5000)
#topography.set_circ_unif(grid_num=0,c_i=0,c_j=110,radius=40,v_max=5000,v_min=3000)
#topography.set_circ_unif(grid_num=0,c_i=0,c_j=160,radius=40,v_max=5000,v_min=3000)
#topography.set_rect_inc_dec(grid_num=0,i_init=0,j_init=110,x_s=40,y_s=50,v_max=5000,v_min=3000)
#topography.set_rect(grid_num=0,i_init=79,j_init=163,x_size=1,y_size=80,value=0)
#topography.set_rect(grid_num=0,i_init=78,j_init=164,x_size=1,y_size=88,value=0)
#topography.plot_grid(0,"topo3",0,5000,center=4800)
#topography.set_circ_unif(grid_num=0,c_i=0,c_j=290,radius=25,v_max=5000,v_min=1500)

#poly = Polygon(corners=[[0,0],[0,10],[5,10],[5,0]])
#print(poly.contains(7,10))
#print(poly.contains(7, 0))

topography.plot_grid(0,"topography",0,5000,center=4800)
exit(1)

init_grid = np.array([np.full((size_x,size_y),0)])
grid = Grid(grid = init_grid, n = size_x, m = size_y, max_time = 200, dec = 4)
grid.set_wave(grid_num=0,c_i=15,c_j=15,R=30,h=40,c=10)
grid.initialize_grid()
t1 = time.time()
print("Time to construct grid: ", (t1 - t0))

# create instance of modelling class
instance = Modelling(init_grid = grid, d = topography) # need to give the grid here like (grid)

# find minimum stability conditions
instance.min_stability()

# set the parameters
instance.set_param(h = 0.1, dt = 0.0002) # can add c = ?, h = ?, dt = ?

t2 = time.time()
print("Time to set up Modelling class: ", (t2 - t1))
# solve the wave equations numerically
instance.solveEq() # should take as parameter max time

t3 = time.time()
print("Time for solveEq(): ", (t3 - t2))
#instance.grid.print_grid()
instance.plot_result("test_set", vmax=8, vmin=-8, step = 1)
t4 = time.time()
print("Time to make plots: ", (t4 - t3))
print("Total time is: ", (t4 - t0))


print(instance.grid.what_is_max())