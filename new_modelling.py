import numpy as np
import math
from math import pow
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.animation import writers
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
        self.grid[k][i][j] = v #np.around(v,self.dec)
    
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
    def set_rect(self, grid_num, c, value, radius = -1, add = False, ignore = -1):
        rect = Polygon(c,3)
        for i in range(0,self.n):
            for j in range(0,self.m):
                if (rect.contains(i, j)):
                    self.grid[grid_num][i][j] = value
                else:
                    distance = rect.distance(i, j, ignore)[0]
                    if (distance <= radius):
                        mult = distance / radius
                        self.grid[grid_num][i][j] = mult*self.grid[grid_num][i][j] + (1 - mult)*value
    
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
    def set_unif_rect(self, grid_num, c_1, c_2, c_3, c_4, v_max, v_min, add = True):
        rect = Polygon([c_1,c_2,c_3,c_4],1,v_min,v_max)

        for i in range(0,self.n):
            for j in range(0,self.m):
                if (rect.contains(i,j)):
                    value  = rect.get_value(i,j)
                    if add:
                        self.grid[grid_num][i][j] += value #np.around(value,self.dec)
                    else:
                        self.grid[grid_num][i][j]  = value #np.around(value,self.dec)

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
    def set_rect_inc_dec(self, grid_num, c, side = 1,
                         v_max= 1, v_min = 0, add = False):
        rect = Polygon(c,2,v_min,v_max, side = side)

        for i in range(0,self.n):
            for j in range(0,self.m):
                if (rect.contains(i,j)):
                    value  = rect.get_value(i,j)
                    if add:
                        self.grid[grid_num][i][j] += value #np.around(value,self.dec)
                    else:
                        self.grid[grid_num][i][j]  = value #np.around(value,self.dec)
    
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
                            self.grid[grid_num][i][j] += value #np.around(value, self.dec)
                        elif radius <= radius:
                            self.grid[grid_num][i][j] = value #np.around(value, self.dec)
    
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
    def set_wave(self, grid_num, c_i, c_j, R, h, c=1, w=3, add = False):
        c = float(c)/6.283
        if c_i > self.n or c_j > self.m:
            print("error in provided parameters. Rectangle does not fit.")
        elif grid_num > self.max_time:
            print("error in time step. There is no grid at this time.")
        else:
            for i in range(self.n):
                for j in range(self.m):
                    dist = np.linalg.norm(np.subtract([i,j],[c_i,c_j]))#math.sqrt(math.pow(i - c_i, 2) + math.pow(j - c_j, 2))
                    if dist/c < R/c + (1/2) + (math.pi/2):
                        value = self.wave_function(dist/c, R/c, h, w)
                        if not add:
                            self.grid[grid_num][i][j] = value #np.around(value, self.dec)
                        else:
                            self.grid[grid_num][i][j] += value #np.around(value, self.dec)
    
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
                         square = True, xticklabels = False, yticklabels = False,
                         cbar = False)
        plt.savefig(name+".png")
        plt.clf()
    
    def wave_function(self, x, R, h, w):
        if x >= max(R - (w+0.5)*math.pi,0) and x < R:
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
        self.max_v = np.amax(corners)
    
    # returns whether the point is contained inside the polygon
    def contains(self, x, y, ex=0):
        extreme = [2*self.max_v, y+ex]
        count = 0
        n = len(self.corners)

        
        for i in self.corners:
            if i == [x,y]:
                return True
            
            if (self.orientation([x,y],extreme,i) == 0 and
                self.onSegment([x,y], i, extreme)):
                if ex == 0:
                    sign = 1
                else:
                    sign = ex/abs(ex)

                return self.contains(x,y,ex=-sign*(abs(ex)+10))

        for i in range(0,n):
            next = (i+1)%n
            if self.check_intersect(self.corners[i], self.corners[next], [x,y], extreme):
                if self.orientation(self.corners[i], [x,y], self.corners[next]) == 0:
                    return self.onSegment(self.corners[i], [x,y], self.corners[next])
                
                count += 1

        return count % 2 == 1
    
    # gets distance from nearst segment
    def distance(self, x, y, j = -1):
        dist = np.array([])
        n = len(self.corners)
        for i in range(n):
            dist = np.append(dist, [self.distance_to(self.corners[i], self.corners[(i+1)%n], [x,y])])
        a = min(dist)
        b = np.where(dist == a)[0][0]
        if (not j == -1) and (b == j or dist[j-1] == dist[j]):
            return [math.inf, j]
        return [a, b]
    
    # distance to segment c_1 -> c_2
    def distance_to(self, c_1, c_2, pt):
        v = np.subtract(c_1, c_2)
        a = np.dot(v, np.subtract(c_1, pt))
        v = np.subtract(c_2, c_1)
        b = np.dot(v, np.subtract(c_2, pt))
        if (a > 0 and b > 0):
            b = np.linalg.norm(v)
            w = np.subtract(c_1, pt)
            a = abs(v[0]*w[1] - v[1]*w[0])
            return a/b
        else:
            a = np.linalg.norm(np.subtract(c_1, pt))
            b = np.linalg.norm(np.subtract(c_2, pt))
            return min(a,b)
    
    # gets the value of the given point based on the chosen strategy (option)
    def get_value(self, x, y):
        if self.option == 1:
            dist = self.distance(x, y)
            step = self.get_step(dist[1])/2
            if step <= 0:
                print("Error, step size too small")
                exit(1)
            return self.v_min + (dist[0]/step)*(self.v_max - self.v_min)
        elif self.option == 2:
            dist = 0
            if self.side == len(self.corners)-1:
                dist = self.distance_to(self.corners[self.side], self.corners[0], [x,y])
            elif self.side < len(self.corners)-1:
                dist = self.distance_to(self.corners[self.side], self.corners[self.side+1], [x,y])
            else:
                print("Invalid side!")
                exit(1)
            step = self.get_step(self.side)
            if step <= 0:
                print("Error, step size too small")
                exit(1)
            return self.v_min*(1 - (dist/step)) + self.v_max*(dist/step)
        else:
            return self.v_max
        
    def get_step(self, i):
        vec = np.subtract(self.corners[i-1], self.corners[i])
        mag = math.sqrt(pow(vec[0],2) + pow(vec[1],2))
        return mag
    
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
    def __init__(self, grid_eta=Grid(), d=Grid(), poly=Polygon(), h=1, dt=1):#grid_w = Grid(), grid_v = Grid(), alpha = 1):
        self.d = d
        self.h = h
        self.dt = dt
        self.max = grid_eta.get_max()
        self.poly = poly
        self.matrix = self.setup()
        # add grid at time = 0
        self.grid = grid_eta
        #self.grid_w = grid_w
        #self.grid_v = grid_v
        #self.alpha = alpha
        self.wasted = 0
        self.count = 0

    def setup(self):
        c_1 = poly.corners[0]
        c_2 = poly.corners[1]
        c_3 = poly.corners[2]
        size_x = int(np.linalg.norm(np.subtract(c_1, c_2))) + 1
        size_y = int(np.linalg.norm(np.subtract(c_2, c_3))) + 1
        return np.zeros((size_y,size_x))
    
    def next_grid(self, k):
        # need to set up boundary conditions here
        if ((k+2)%50 == 0):
            print("Starting grid: ", k+2)
            if np.amax(self.matrix) > 0:
                ax = sns.heatmap(self.matrix)
                plt.savefig("matrix_"+str(k))
                plt.clf()
        
        mat = np.zeros(self.matrix.shape)
        for i in range(0, self.grid.n-1):
            for j in range(0, self.grid.m-1):
                # use the "update formula"
                self.get_open(k,i,j)
                t_init = time.time()
                if (i <= 80*s and j <= 142*s and self.grid.get(k+1,i,j) > 0 and poly.contains(i,j)):
                    c_1 = poly.corners[0]
                    c_2 = poly.corners[1]
                    c_3 = poly.corners[2]
                    l = int(poly.distance_to(c_1, c_2, [i,j]))
                    m = int(poly.distance_to(c_2, c_3, [i,j]))
                    mat[l][m] = self.grid.get(k+1,i,j)
                    self.matrix[l][m] = max(mat[l][m], self.matrix[l][m])
                self.wasted = self.wasted - t_init + time.time()

        if self.count < 5 and (np.amax(mat) > 0):
            self.count = self.count + 1
            ax = sns.heatmap(self.matrix)
            plt.savefig("matrix_num_"+str(self.count))
            plt.clf()
            ax = sns.heatmap(mat)
            plt.savefig("mat_num_"+str(self.count))
            plt.clf()
    
    # compute next step in i, j
    def get_open(self, k, i, j):
        c = math.sqrt(9.81*self.d.get(0,i,j))
        term = pow(c * self.dt / self.h, 2)
        t1 = 2*self.grid.get(k,i,j) - self.grid.get(k-1,i,j)
        t2 = self.get_elmt(k,i+1,j) - 2*self.grid.get(k,i,j) + self.get_elmt(k,i-1,j)
        t3 = self.get_elmt(k,i,j+1) - 2*self.grid.get(k,i,j) + self.get_elmt(k,i,j-1)
        
        self.grid.set_v(k+1, i, j, t1 + term*t2 + term*t3)

    def get_elmt(self, k, i, j):
        if (i > self.grid.n):
            return 0
        elif (j > self.grid.m):
            return 0
        elif (i < 0):
            return 0
        elif (j < 0):
            return 0
        
        return self.grid.get(k,i,j)

    '''
    def get_shallow(self, k, i, j):
        result = self.get_vec(k,i,j).copy()
        self.grid_eta.set_v(k+1,i,j, result[0])
        self.grid_w.set_v(k+1,i,j, result[1])
        self.grid_v.set_v(k+1,i,j, result[2])
    '''

    '''
    def get_vec(self, k, i, j):
        m1 = np.array([[self.alpha*self.grid_w.get(k,i,j), self.alpha*self.grid_eta.get(k,i,j)+self.d.get(0,i,j), 0],
                       [9.81, self.alpha*self.grid_w.get(k,i,j), 0],
                       [0, 0, self.alpha*self.grid_w.get(k,i,j)]])
        m2 = np.array([[self.alpha*self.grid_v.get(k,i,j), 0, self.alpha*self.grid_eta.get(k,i,j)+self.d.get(0,i,j)],
                       [0, self.alpha*self.grid_v.get(k,i,j), 0],
                       [9.81, 0, self.alpha*self.grid_v.get(k,i,j)]])
        m3 = np.array([[self.grid_w.get(k,i,j), self.grid_v.get(k,i,j), 0],
                       [0, 0, 0],
                       [0, 0, 0]])
        
        #if (self.d.n-1 < i+1):
        if (0 < i):
            ai_1 = self.grid_eta.get(k,i-1,j)
            bi_1 = self.grid_w.get(k,i-1,j)
            ci_1 = self.grid_v.get(k,i-1,j)
            di_1 = self.d.get(0,i-1,j)
        else:
            ai_1 = 0
            bi_1 = self.grid_w.get(k,i,j)
            ci_1 = self.grid_v.get(k,i,j)
            di_1 = 5000
        
        #if (self.d.m-1 < j+1):
        if (0 < j):
            aj_1 = self.grid_eta.get(k,i,j-1)
            bj_1 = self.grid_w.get(k,i,j-1)
            cj_1 = self.grid_v.get(k,i,j-1)
            dj_1 = self.d.get(0,i,j-1)
        else:
            aj_1 = 0
            bj_1 = self.grid_w.get(k,i,j)
            cj_1 = self.grid_v.get(k,i,j)
            dj_1 = 5000
       
        if (self.d.n-1 < i+1):
            ai_2 = self.grid_eta.get(k,i+1,j)
            bi_2 = self.grid_w.get(k,i+1,j)
            ci_2 = self.grid_v.get(k,i+1,j)
            di_2 = self.d.get(0,i+1,j)
        else:
            ai_2 = 0
            bi_2 = self.grid_w.get(k,i,j)
            ci_2 = self.grid_v.get(k,i,j)
            di_2 = 5000
        
        if (self.d.m-1 < j+1):
            aj_2 = self.grid_eta.get(k,i,j+1)
            bj_2 = self.grid_w.get(k,i,j+1)
            cj_2 = self.grid_v.get(k,i,j+1)
            dj_2 = self.d.get(0,i,j+1)
        else:
            aj_2 = 0
            bj_2 = self.grid_w.get(k,i,j)
            cj_2 = self.grid_v.get(k,i,j)
            dj_2 = 5000
        
        v1 = np.array([ai_2 - ai_1,#ai - self.grid_eta.get(k,i,j),
                       bi_2 - bi_1,#bi - self.grid_w.get(k,i,j),
                       ci_2 - ci_1])#ci - self.grid_v.get(k,i,j)])
        v2 = np.array([aj_2 - aj_1,#aj - self.grid_eta.get(k,i,j),
                       bj_2 - bj_1,#bj - self.grid_w.get(k,i,j),
                       cj_2 - cj_1])#cj - self.grid_v.get(k,i,j)])
        v3 = np.array([di_2 - di_1,#di - self.d.get(0,i,j),
                       dj_2 - dj_1,#dj - self.d.get(0,i,j),
                       0])
        v = np.array([self.grid_eta.get(k-1,i,j), self.grid_w.get(k-1,i,j), self.grid_v.get(k-1,i,j)])

        return np.subtract(v, (self.dt / self.h) * (np.add(np.add(m1 @ v1, m2 @ v2), m3 @ v3)))
    '''
    
    def solveEq(self):
        # for loop for time steps and updating grid
        # self.grid_eta.initialize_grid()

        for k in range(1, self.max-1): # some stopping condition
            # use next_grid to compute next step
            self.next_grid(k)
        #print("hello123")
        print(self.matrix)
        print("Time wasted: ", self.wasted)
    
    # set the initial parameters
    def set_param(self, c = 0, h = 0, dt = 0):
        if not c == 0:
            self.c = c
        if not h == 0:
            self.h = h
        if not dt == 0:
            self.dt = dt

def min_stability(h, c):
    # find minimal stable parameters, whatever that may mean
    return h / (c*math.sqrt(2))

# creates the Palma islands topography
def palmaTopography(size_x, size_y, s):
    init_topo = np.array([np.full((size_x,size_y),5000)])
    topography = Grid(grid = init_topo, n = size_x, m = size_y, max_time = 1, dec = 0)

    # decreasing polygons for the narrow channel
    topography.set_rect_inc_dec(0,[[40*s,110*s],[75*s,135*s],[160*s,135*s],[160*s,0],[100*s,0]],3,v_max=1000,v_min=50)
    topography.set_rect_inc_dec(0,[[40*s,110*s],[0,110*s],[0,0],[40*s,0]],2,v_max=1000,v_min=50)
    topography.set_rect(0,[[0,110*s],[75*s,135*s],[40*s,110*s]],1000)
    topography.set_rect_inc_dec(0,[[0,110*s],[75*s,135*s],[80*s,140*s],[80*s,170*s],
                                [0,170*s]],0,v_max=5000,v_min=1000)
    
    # Creates a polygon with 8 corners and a 70*s unit long shore (uniform decrease from 0 to 5000 depth)
    # Represents Spain / Portugal
    # First imput should always be 0
    # Second input is the corners of the polygon
    # Third input is the value inside the polygon
    # Fourth is how far the "shore" extends around the polygon, aka how far until 5000m depth
    topography.set_rect(0,[[160*s,280*s],[90*s,330*s],[75*s,340*s],[90*s,370*s],
                        [115*s,400*s],[130*s,425*s],[145*s,432*s],[160*s,460*s]],0,70*s)

    # France done in two Polygons due to different shores
    topography.set_rect(0,[[160*s,190*s],[140*s,160*s],[100*s,140*s],[80*s,140*s],
                        [160*s,135*s]],0,40*s,ignore=3)
    topography.set_rect(0,[[160*s,135*s],[80*s,140*s],[75*s,135*s],[80*s,130*s],[110*s,115*s],
                        [85*s,110*s],[120*s,80*s],[125*s,45*s],[160*s,20*s]],0,10*s)
    # The UK land mass
    topography.set_rect(0,[[0,20*s],[20*s,30*s],[20*s,50*s],
                        [40*s,40*s],[35*s,70*s],[40*s,110*s],
                        [70*s,60*s],[110*s,40*s],[120*s,25*s],
                        [100*s,30*s],[100*s,0],[0,0]], 0,10*s)

    # uniform values in a cirlce for the Palma and other islands, inputs are straightforward
    topography.set_circ_unif(0,160*s,600*s,36*s,5000,0,c_min=4*s)
    topography.set_circ_unif(0,0,580*s,50*s,5000,200,c_min=8*s)

    poly = Polygon([[45*s,103*s],[35*s,117*s],[70*s,142*s],[80*s,128*s]])
    
    return poly, topography

def otherTopography(size_x, size_y, s):
    init_topo = np.array([np.full((size_x, size_y), 5000)])
    topography = Grid(grid = init_topo, n = size_x, m = size_y, max_time = 1, dec = 0)
    
    return topography

def get_amplitude(size_x, size_y, s, t, h, r, c, w):
    init_grid = np.array([np.full((size_x,size_y),0)])
    grid = Grid(grid = init_grid, n = size_x, m = size_y, max_time = t, dec = 3)

    grid.set_wave(grid_num=0,c_i=160*s,c_j=600*s,R=r*s,h=h*1.2,c=c*s,w=w)
    grid.set_rect(grid_num=0,c=[[160*s-1,0],[160*s-1,600*s],[160*s,600*s],[160*s,0]],value=0)
    grid.set_rect(grid_num=0,c=[[160*s-1,600*s-1],[0,600*s-1],[0,600*s],[160*s-1,600*s]],value=0)

    grid.initialize_grid()
    grid.set_grid(0,init_grid)

    grid.set_wave(grid_num=0,c_i=160*s,c_j=600*s,R=r*s*0.99,h=h,c=(c-1)*s,w=w)
    grid.set_rect(grid_num=0,c=[[160*s-1,0],[160*s-1,600*s],[160*s,600*s],[160*s,0]],value=0)
    grid.set_rect(grid_num=0,c=[[160*s-1,600*s-1],[0,600*s-1],[0,600*s],[160*s-1,600*s]],value=0)

    return grid

'''
def get_velocity(size_x, size_y, s, topo, amp, v, t):
    init_grid = np.array([np.full((size_x,size_y),0)])
    grid_v = Grid(grid = init_grid, n = size_x, m = size_y, max_time = t, dec = 1)
    grid_w = Grid(grid = init_grid, n = size_x, m = size_y, max_time = t, dec = 1)

    for i in range(topo.n):
        for j in range(topo.m):
            #if not topo.get(0,i,j) == 0:
                #grid_w.set_v(0,i,j,200)
                #grid_v.set_v(0,i,j,200)
            
            if not amp.get(1,i,j) == 0 and (not topo.get(0,i,j) == 0):
                w = np.subtract(v, [i,j])
                x, y = get_dir(w)
                grid_v.set_v(0,i,j,amp.get(0,i,j)*x*math.sqrt(9.81/topo.get(0,i,j)))
                grid_w.set_v(0,i,j,amp.get(0,i,j)*y*math.sqrt(9.81/topo.get(0,i,j)))

    grid_v.initialize_grid()
    grid_w.initialize_grid()

    for i in range(topo.n):
        for j in range(topo.m):
            #if not topo.get(0,i,j) == 0:
                #grid_w.set_v(0,i,j,200)
                #grid_v.set_v(0,i,j,200)
            
            if not amp.get(0,i,j) == 0 and (not topo.get(0,i,j) == 0):
                w = np.subtract(v, [i,j])
                x, y = get_dir(w)
                grid_v.set_v(0,i,j,amp.get(0,i,j)*x*math.sqrt(9.81/topo.get(0,i,j)))
                grid_w.set_v(0,i,j,amp.get(0,i,j)*y*math.sqrt(9.81/topo.get(0,i,j)))
    
    return grid_v, grid_w
'''

'''
def get_dir(v):
    v = v/np.linalg.norm(v)
    return v[0], v[1]
'''

# create grid
# parameters
#   grid: grid at time = 0
#   n: x dimension of grid
#   m: y dimension of grid
#   max_time: time up to which we are considering the model
#   dec: number of decimals after the point
s = float(input("Enter the scale you want: "))
print()
print("s is now given by: ", s)
print("s is of type: ", type(s))

t = int(input("Enter the number of timesteps: "))
print()
print("t is now given by: ", t)
print("t is of type: ", type(t))

h = float(input("Enter the initial amplitude: "))
print()
print("h is now given by: ", h)
print("h is of type: ", type(h))

name = input("Enter a name: ")
print()
print("name is now given by: ", name)
print("name is of type: ", type(name))

# find minimum stability conditions
print("Given the parameters, the minimum delta t is: ", min_stability(10/(s*2), math.sqrt(9.81 * 5000)))

delta = float(input("Enter the timestep size: "))
print()
print("delta is now given by: ", delta)
print("delta is of type: ", type(delta))

radius = int(input("Enter the radius: "))
print()
print("radius is now given by: ", radius)
print("radius is of type: ", type(radius))

wavelen = int(input("Enter the wavelength: "))
print()
print("wavelen is now given by: ", wavelen)
print("wavelen is of type: ", type(wavelen))

waves = int(input("Enter the number of waves: "))
print()
print("waves is now given by: ", waves)
print("waves is of type: ", type(waves))

'''
alpha = float(input("Enter the value or alpha: "))
print()
print("alpha is now given by: ", alpha)
print("alpha is of type: ", type(alpha))
'''

size_x = int(160*s)
size_y = int(600*s)
t0 = time.time()

poly, topography = palmaTopography(size_x, size_y, s)
topography.plot_grid(0,"topography",center=3000,vmax=3000)
t1 = time.time()
print("Made topography in: ", (t1 - t0))

grid_eta = get_amplitude(size_x, size_y, s, t, h, radius, wavelen, waves)
grid_eta.plot_grid(0,"initial_wave "+name,vmin=-h*0.5,vmax=h*0.5,center=0)
grid_eta.plot_grid(1,"second_wave "+name, vmin=-h*0.5,vmax=h*0.5,center=0)
t2 = time.time()
print("Time to construct grid: ", (t2 - t1))

'''
grid_v, grid_w = get_velocity(size_x, size_y, s, topography, grid_eta, [160*s, 600*s], t)
grid_v.plot_grid(0, "initial x speed "+name,vmin=-h*0.1,vmax=h*0.1,center=0)
grid_w.plot_grid(0, "initial y speed "+name,vmin=-h*0.1,vmax=h*0.1,center=0)
t3 = time.time()
print("Time to construct velocity: ", (t3 - t2))
'''

# create instance of modelling class
instance = Modelling(grid_eta=grid_eta, d=topography, poly = poly)#grid_w=grid_w, grid_v=grid_v, alpha = alpha) # need to give the grid here like (grid)

# set the parameters
instance.set_param(h = 10/(s*2), dt = delta)#54) # can add c = ?, h = ?, dt = ?

t4 = time.time()
print("Time to set up Modelling class: ", (t4 - t2))
# solve the wave equations numerically
instance.solveEq() # should take as parameter max time

t5 = time.time()
print("Time for solveEq(): ", (t5 - t4))

#instance.plot_result("test_set", vmax=200, vmin=-200, step = 1, time = 200)
def animate(i):
    plt.clf()
    ax = sns.heatmap(grid[i], vmin = -h*0.5, vmax = h*0.5, center = 0,
                         square = True, xticklabels = False, yticklabels = False,
                         cbar = False)

def save_animation(grid, fig, name):
    anim = animation.FuncAnimation(fig, animate, interval = 20, frames = t)

    Writer = writers['ffmpeg']
    writer = Writer(fps=50, metadata={'artist': 'Me'}, bitrate=1000)

    anim.save('Tsunami simulation '+name+'.mp4', writer)

grid = instance.grid.grid
fig = plt.figure()
save_animation(grid, fig, ""+name)
t6 = time.time()
print("Time to make amplitude video: ", (t6 - t5))

'''
grid = instance.grid_w.grid
fig = plt.figure()
save_animation(grid, fig, "w "+name)
t7 = time.time()
print("Time to make x velocity video: ", (t7 - t6))

grid = instance.grid_v.grid
fig = plt.figure()
save_animation(grid, fig, "v "+name)
t8 = time.time()
print("Time to make y velocity video: ", (t8 - t7))
'''

#instance.grid_v.plot_grid(100, "test_v")
#instance.grid_w.plot_grid(100, "test_w")

print("Total time is: ", (t6 - t0))

#print(instance.grid.what_is_max())

