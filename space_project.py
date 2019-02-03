
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation
from datetime import datetime


# Attaching 3D axis to the figure
fig = plt.figure()
ax = p3.Axes3D(fig)



earth = {
  'e': 0.016,
  'a': 1,
  'i': 0,
  'period': 365.24,
  'offset': 1.1519173063
}

ceres= {
  'e': 0.076, 
  'a': 2.77, 
  'i': 0.18483037, 
  'period': 1683.15, 
  'offset': 0
}



# Setting the axes properties

ax.set_xlabel('X')


ax.set_ylabel('Y')


ax.set_zlabel('Z')

ax.set_xlim(-3,3)
ax.set_ylim(-3,3)
ax.set_zlim(-3,3)
ax.set_title('3D Test')

# Creating the Animation object
#line_ani = animation.FuncAnimation(fig, update_lines, 25, fargs=(data, lines),
#                                   interval=50, blit=False)

#plt.show()

def focuspath(e,a,q,i,period, timediff, offset):
  """calculate the path of the comet based on inputs
  
  INPUTS: e = 
          a = semi major axis length
          i = inclination (cameron's phi)
          period = 
          timediff =  this parameter/period is the percentage along the orbit//this assumes t=0 is 30-Apr-2018 6:05
          initialPos = should be the percentage along orbit * period
  OUTPUTS: a 3-D point
  
  """
  b = a*np.sqrt(1-e**2)
  c=np.sqrt(a**2-b**2)
  theta=0 #need a real theta value 
  xcoord = a*np.cos(timediff)*np.cos(theta)-b*np.sin(timediff)*np.sin(theta)-c*np.cos(theta) #c term is to adjust for the focus
  ycoord = a*np.cos(timediff)*np.sin(theta)*np.cos(i)+b*np.sin(timediff)*np.cos(theta)*np.cos(i)-c*np.sin(theta)*np.cos(i)
  zcoord = a*np.cos(timediff)*np.sin(theta)*np.sin(i)+b*np.sin(timediff)*np.cos(theta)*np.sin(i)-c*np.sin(theta)*np.sin(i)
  
  
  return xcoord, ycoord, zcoord
def angle_helper(t,e):
  return t-(t-e*np.sin(t)-t)/(1-e*np.cos(t))-(t-(t-e*np.sin(t)-t)/(1-e*np.cos(t))-e*np.sin(t-(t-e*np.sin(t)-t)/(1-e*np.cos(t)))-t)/(1-e*np.cos(t-(t-e*np.sin(x)-t)/(1-e*np.cos(x))))-(t-(t-e*np.sin(t)-t)/(1-e*np.cos(t))-(t-(t-e*np.sin(t)-t)/(1-e*np.cos(t))-e*np.sin(t-(t-e*np.sin(t)-t)/(1-e*np.cos(t)))-t)/(1-e*np.cos(t-(t-e*np.sin(t)-t)/(1-e*np.cos(e))))-e*np.sin(t-(t-e*np.sin(t)-t)/(1-e*np.cos(t))-(t-(t-e*np.sin(t)-t)/(1-e*np.cos(t))-e*np.sin(t-(t-e*np.sin(t)-t)/(1-e*np.cos(t)))-t)/(1-e*np.cos(t-(t-e*np.sin(t)-t)/(1-e*np.cos(t)))))-t)/(1-e*np.cos(t-(t-e*np.sin(t)-t)/(1-e*np.cos(t))-(t-(t-e*np.sin(t)-t)/(1-e*np.cos(t))-e*np.sin(t-(t-e*np.sin(t)-t)/(1-e*np.cos(t)))-t)/(1-e*np.cos(t-(t-e*np.sin(t)-t)/(1-e*np.cos(t))))))
def path(e,a,q,i,period, timediff, offset):
  
  """calculate the path of the comet based on inputs
  
  INPUTS: e = eccentricity
          a = semi major axis length
          i = inclination (cameron's phi)
          period = 
          timediff =  //this assumes t=0 is 30-Apr-2018 6:05
          initialPos = 3D vector describing planet's position at t=0
  OUTPUTS: a 3-D point
  
  """
  b = a*np.sqrt(1-e**2)
  c=np.sqrt(a**2-b**2)
  #t = ((timediff*np.pi*b)/period)
  #t1 = ((timediff)*np.pi*b)/period
  #print(t1-t)
  #t1 = (((timediff*np.pi*b)/period) + e*np.sin(t))%(2*np.pi)
  #print(t)
  t1= angle_helper(((timediff % period)/period)*2*np.pi,e)
  theta=0 #need a real theta value 
  xcoord = a*np.cos(t1+offset)*np.cos(theta)-b*np.sin(t1+offset)*np.sin(theta)-c*np.cos(theta) #c term is to adjust for the focus
  ycoord = a*np.cos(t1+offset)*np.sin(theta)*np.cos(i)+b*np.sin(t1+offset)*np.cos(theta)*np.cos(i)-c*np.sin(theta)*np.cos(i)
  zcoord = a*np.cos(t1+offset)*np.sin(theta)*np.sin(i)+b*np.sin(t1+offset)*np.cos(theta)*np.sin(i)-c*np.sin(theta)*np.sin(i)
  
  
  return xcoord, ycoord, zcoord

import numpy as np


class Ceres:
  def __init__(self,planet):
    self.xcoords = []
    self.ycoords=[]
    self.zcoords=[]
    self.datax= []
    self.datay=[]
    self.dataz=[]
    self.frames = []
  def getPath(self):
    return [self.xcoords, self.ycoords, self.zcoords]
  def createFrames(self):
    for index in range(len(self.datax)):
      self.frames.append([self.datax[index],self.datay[index],self.dataz[index]])
    return self.frames

ceres = Ceres(ceres)
for t in range(1000):
    x, y, z = focuspath(0.076, 2.77, 0, 0.18483037, 0, t/10.0,0)
    ceres.xcoords.append(x)
    ceres.ycoords.append(y)
    ceres.zcoords.append(z)

#print(ceres.getPath())

data = ceres.getPath()
orbit1 = [ax.plot(data[0], data[1], data[2])[0]]
#line_ani = animation.FuncAnimation(fig, update_lines, 25, fargs=(data, lines),
##                                   interval=50, blit=True,)
earth = Ceres(earth)
for t in range(0,1000):
    x, y, z = focuspath(0.016, 1, 0, 0, 0, t/10.0,0)
    earth.xcoords.append(x)
    earth.ycoords.append(y)
    earth.zcoords.append(z)
data = earth.getPath()
orbit2 = [ax.plot(data[0], data[1], data[2])[0]]
for t in range(0,100000):
    x, y, z = path(0.016, 1,0,0,365.24,t, 1.1519173063)
    earth.datax.append(x)
    earth.datay.append(y)
    earth.dataz.append(z)
for t in range(0,100000):
    x, y, z = path(0.076, 2.77, 0, 0.18483037, 1683.15, t,0)
    ceres.datax.append(x)
    ceres.datay.append(y)
    ceres.dataz.append(z)
eframe = earth.createFrames()
cframe = ceres.createFrames()
escatter = ax.scatter(eframe[0][0], eframe[0][1],  eframe[0][2],c='b',marker = 'o')
cscatter = ax.scatter(cframe[0][0], cframe[0][1], cframe[0][2],c='b',marker = 'o')
sun = ax.scatter(0, 0, 0,c='r',marker = 'o')
textThing = ax.text(0,0,-3,0)
#escatter1 = ax.scatter(eframe[0][0], eframe[0][1],  eframe[0][2],c='b',marker = 'o')
#cscatter1 = ax.scatter(cframe[0][0], cframe[0][1], cframe[0][2],c='b',marker = 'o')
print(len(eframe))
print(len(cframe))
#print(eframe)
def update(frame):
    global escatter, cscatter, textThing
    escatter.remove()
    cscatter.remove()
   
    day = datetime.fromtimestamp((frame*24*60*60)+48*31557600+120*86400).strftime("%d, %Y, %B")
    textThing.set_text(day)
    escatter = ax.scatter(eframe[frame][0], eframe[frame][1], eframe[frame][2],c='b',marker = 'o')
    cscatter = ax.scatter(cframe[frame][0], cframe[frame][1], cframe[frame][2],c='b',marker = 'o')

    return  escatter, cscatter
ani = animation.FuncAnimation(fig, update, len(eframe),
                                   interval=100, blit=False)

# ani.save('kepler.mp4')
# for x in xrange(1,200):
#     plt.plot([eframe[x][0]], [eframe[x][1]], [eframe[x][2]])

plt.show()