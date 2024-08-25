import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from modules.particles import Particle


class Swarm:
    #initialize with some predefined values 
    def __init__(self , n_particles:int=50, particles:list=[],dim:int=2,w_max:float=0.9 ,w_min:float=0.4, c1:float=2, c2:float=2, bounds:float=200, iterations:int=100, objective_function:callable=lambda x,y: x**2+y**2,optimization_method:str="min"):
        self.objective_function = objective_function
        self.n_particles = n_particles
        self.dim = dim
        self.particles = particles
        self.w_max = w_max
        self.w_min = w_min
        self.w = w_max
        self.c1 = c1
        self.c2 = c2
        self.bounds = bounds
        self.global_best_position = np.repeat(None,dim)
        self.global_best_value = None
        self.iterations = iterations
        self.optmimization_method = optimization_method
        # self.fig, self.ax = plt.subplots()
        # self.data = []
        
    def lineal_reduction_inertia(self,iteration:int):
        self.w=(self.w_max-self.w_min)*(self.iterations-iteration)/self.iterations+self.w_min
        
    def create_particles(self):
        #initialize the particles
        for i in range(self.n_particles):
            self.particles.append(Particle(dim=self.dim,bounds=self.bounds))
    def find_global_best(self):
        #find the global best position of the particles
        for particle in self.particles:
            if self.global_best_value == None:
                self.global_best_value = particle.personal_best_value
                self.global_best_position = particle.personal_best_position        
            elif self.optmimization_method == "min":
                if self.global_best_value > particle.personal_best_value:
                    self.global_best_value = particle.personal_best_value
                    self.global_best_position = particle.personal_best_position
            elif self.optmimization_method == "max":
                if self.global_best_value < particle.personal_best_value:
                    self.global_best_value = particle.personal_best_value
                    self.global_best_position = particle.personal_best_position

            # elif self.global_best_value > particle.personal_best_value:
            #     self.global_best_value = particle.personal_best_value
            #     self.global_best_position = particle.personal_best_position

    def optimize(self):
        #optimize the particles
        #initialize the value, personal_best_position and personal_best_value of the particles
        #points=ax.plot([],[],'o')
        plt.x_label = "x"
        plt.y_label = "y"
        plt.title = "Particle Swarm Optimization"
        x_0 = np.linspace(-self.bounds,self.bounds,100)
        x_1 = np.linspace(-self.bounds,self.bounds,100)
        X_0,X_1 = np.meshgrid(x_0,x_1)
        Z = self.objective_function(X_0,X_1)
        plt.contour(X_0,X_1,Z,levels=35,cmap='RdGy')
        for particle in self.particles:
            particle.evaluate(self.objective_function)
            particle.update_personal_best(self.optmimization_method)
            print("Particle value: ",particle.value)
        for i in range(self.iterations):
            x_positions = []
            y_positions = []
            x_y_positions = []
            plt.clf()
            self.find_global_best()
            for particle in self.particles:
                #r1,r2 is a np.array with dimension dim with random values between 0 and 1
                r1 = np.random.uniform(0,1,self.dim)
                r2 = np.random.uniform(0,1,self.dim)
                particle.update_velocity(self.global_best_position,self.w,self.c1,self.c2,r1,r2)
                particle.update_position()
                particle.evaluate(self.objective_function)
                particle.update_personal_best()
                self.lineal_reduction_inertia(i)
                x_positions.append(particle.position[0])
                y_positions.append(particle.position[1])
                x_y_positions.append((x_positions,y_positions))
            if i%1 == 0:
                plt.text(30,57,"iteration: "+str(i))
                plt.xlim(-self.bounds,self.bounds)
                plt.ylim(-self.bounds,self.bounds)
                plt.contour(X_0,X_1,Z,levels=35,cmap='RdGy')
                plt.scatter(x_positions,y_positions,s=10,c='b')
                plt.pause(0.1)
            if i == self.iterations-1:
                plt.text(30,57,"iteration: "+str(i))
                plt.xlim(-self.bounds,self.bounds)
                plt.ylim(-self.bounds,self.bounds)
                plt.contour(X_0,X_1,Z,levels=35,cmap='RdGy')
                plt.scatter(x_positions,y_positions,s=10,c='b' )
                plt.pause(0.5)
        print("Global best value: ",self.global_best_value)
        print("Global best position: ",self.global_best_position)
        print("w: ",self.w)
        plt.show()


            # print("Iteration: ",i, "particles",[x.value for x in self.particles])
            # x_positions = [x.position[0] for x in self.particles]
            # y_positions = [x.position[1] for x in self.particles]
            #data.append((x_positions,y_positions))
            # plt.scatter(x_positions,y_positions)
            # plt.show()
