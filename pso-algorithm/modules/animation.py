import matplotlib.pyplot as plt
import numpy as np

class Animation:
    def __init__(self,bounds,objective_function,x_positions,y_positions) -> None:
        self.bounds = bounds
        x=np.linspace(-self.bounds,self.bounds,100)
        y=np.linspace(-self.bounds,self.bounds,100)
        self.x,self.y = np.meshgrid(x,y)
        self.z = objective_function(self.x,self.y)
        self.x_positions = x_positions
        self.y_positions = y_positions
    def animateFunction(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(self.x,self.y,self.z)
        plt.show()
    def animateCountour(self,cmap='viridis'):
        plt.contour(self.x,self.y,self.z,levels=4*self.bounds,cmap=cmap)
    def animateScatter(self,size=2,color='black'):
        plt.scatter(self.x_positions,self.y_positions,s=size,c=color,zorder=10)
    def animate(self,velocity=5,cmap="viridis",size=2,color='black'):
        plt.xlim(-self.bounds,self.bounds)
        plt.ylim(-self.bounds,self.bounds)
        self.animateCountour(cmap=cmap)
        self.animateScatter(size=size,color=color)
        plt.pause(1/velocity)
    def clear(self):
        plt.clf()
    def show(self):
        plt.show()