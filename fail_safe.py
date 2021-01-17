import numpy as np
import json

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

def save_to_json(name, case, m, k):
    data = list(m)
    for i in range(len(m)):
        data[i] = list(m[i])
    
    with open(name+"_"+case+"_data_py.json", "r") as read_file:
        a = json.load(read_file)
        a.append(data)
        data = a
    
    with open(name+"_"+case+"_data_py.json", "w") as write_file:
        json.dump(data, write_file)

    with open(name+"_"+case+"_data_time.json", "r") as read_file:
        data = json.load(read_file)
        data.append(list([k,np.amax(m)]))

    with open(name+"_"+case+"_data_time.json", "w") as write_file:
        json.dump(data, write_file, indent=2)

def init_json(name, case, mat):
    data = list(mat)
    for i in range(len(data)):
        data[i] = list(data[i])
    
    with open(name+"_"+case+"_data_py.json", "w") as write_file:
        json.dump(list([data]), write_file)

    with open(name+"_"+case+"_data_time.json", "w") as write_file:
        json.dump(list([list([0,0])]), write_file)

name = input("Enter the name: ")
s = float(input("Enter the scale: "))
case = int(input("Specify case 1 or 2: "))

data = 0
with open(name+"_grid.json", "r") as read_file:
	data = json.load(read_file)

data = np.array(data)

print("Gird has been loaded")

poly = Polygon()
if case == 1:
    poly = Polygon([[120*s,25*s],[90*s,70*s],[105*s,80*s],[135*s,35*s]])
else:
    poly = Polygon([[40*s,110*s],[100*s,30*s],[140*s,60*s],[80*s,140*s]])

#poly = Polygon([[100*s,0],[100*s,30*s],[120*s,30*s],[120*s,0]])
#poly = Polygon([[120*s,25*s],[90*s,70*s],[105*s,80*s],[135*s,35*s]])
#poly = Polygon([[40*s,110*s],[100*s,30*s],[140*s,60*s],[80*s,140*s]])
#poly = Polygon([[120*s,25*s],[100*s,55*s],[115*s,65*s],[135*s,35*s]])
#poly = Polygon([[45*s,103*s],[35*s,117*s],[70*s,142*s],[80*s,128*s]])

c_1 = poly.corners[0]
c_2 = poly.corners[1]
c_3 = poly.corners[2]
s_x = int(np.linalg.norm(np.subtract(c_1, c_2))) + 1
s_y = int(np.linalg.norm(np.subtract(c_2, c_3))) + 1
output = np.zeros((len(data),s_y,s_x))
init_json(name, str(case), output[0])

for k in range(len(data)):
    if (k+1)%50 == 0:
        print("Starting on gird: "+str(k+1))
    for i in range(len(data[k])):
        for j in range(int(200*s)):
            if (data[k][i][j] > 0 and poly.contains(i,j)):
                l = int(poly.distance_to(c_1, c_2, [i,j]))
                m = int(poly.distance_to(c_2, c_3, [i,j]))
                output[k][l][m] = data[k][i][j]
    if np.amax(output[k]) > 0:
        save_to_json(name, str(case), output[k], k)