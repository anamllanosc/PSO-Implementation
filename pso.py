# is ready to run
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
#particle class
class Particle:
    #initialize with some predefined values
    def __init__(self, dim:int,bounds:float):
        #initialize the position of the particle with random values
        self.position = np.random.uniform(-bounds,bounds,dim)
        #limiting the position of the particle to the bounds
        #initialize the velocity of the particle with random values
        self.velocity = np.random.uniform(-bounds,bounds,dim)
        #initialize the personal best position of the particle with random values
        self.personal_best_position = np.zeros(dim)
        #initialize the personal best value of the particle with random values
        self.personal_best_value = None
        #initialize the value of the particle with random values
        self.value = None
        #initialize the bounds of the particle
        self.bounds = bounds
    def evaluate(self,objective_function):
        #evaluate the value of the particle
        self.value = objective_function(*self.position)

    def update_velocity(self, global_best_position:np.array,w:float,c1:float,c2:float,r1:np.array,r2:np.array):
        #update the velocity of the particle
        self.velocity = w*self.velocity + c1*r1*(self.personal_best_position-self.position) + c2*r2*(global_best_position-self.position)
    def update_position(self):
        #update the position of the particle
        self.position = np.clip(self.position + self.velocity,-self.bounds,self.bounds)
    def update_personal_best(self,optmimization_method:str="min"):
        #update the personal best position and value of the particle
        if self.personal_best_value == None:
            self.personal_best_value = self.value
            self.personal_best_position = self.position
        elif optmimization_method == "min":
            if self.personal_best_value > self.value:
                self.personal_best_value = self.value
                self.personal_best_position = self.position
        elif optmimization_method == "max":
            if self.personal_best_value < self.value:
                self.personal_best_value = self.value
                self.personal_best_position = self.position
        # elif self.personal_best_value > self.value:
        #     self.personal_best_value = self.value
        #     self.personal_best_position = self.position
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
        
#sphere function
def test_function_0(x,y):
    return x**2+y**2         
#function from https://cienciadedatos.net/documentos/py02_optimizacion_pso
def test_function_1(x_0,x_1):
    #Para la región acotada entre −10<=x_0<=0 y −6.5<=x_1<=0 la función tiene
    #múltiples mínimos locales y un único minimo global que se encuentra en
    #f(−3.1302468,−1.5821422) = −106.7645367
    f= np.sin(x_1)*np.exp(1-np.cos(x_0))**2 \
        + np.cos(x_0)*np.exp(1-np.sin(x_1))**2 \
        + (x_0-x_1)**2
    return(f)
#ackely function
def test_function_2(x_0,x_1):
    f = -20*np.exp(-0.2*np.sqrt(0.5*(x_0**2+x_1**2))) \
        - np.exp(0.5*(np.cos(2*np.pi*x_0)+np.cos(2*np.pi*x_1))) \
        + np.e + 20
    return(f)
#rastrigin function
def test_function_3(x_0,x_1):
    f = 20 + x_0**2 + x_1**2 - 10*(np.cos(2*np.pi*x_0)+np.cos(2*np.pi*x_1))
    return(f)
#rosenbrock function
def test_function_4(x_0,x_1):
    f = 100*(x_1-x_0**2)**2 + (1-x_0)**2
    return(f)
#holder table function
def test_function_5(x_0,x_1):
    f = -np.abs(np.sin(x_0)*np.cos(x_1)*np.exp(np.abs(1-np.sqrt(x_0**2+x_1**2)/np.pi)))
    return(f)

# here is just to change the function to test the code 
swarm1 = Swarm(objective_function=test_function_1,iterations=100,n_particles=200,bounds=10)
swarm1.create_particles()
swarm1.optimize()



#testing the functions
# import numpy as np
# def test_function_1(x_0,x_1):
#     #Para la región acotada entre −10<=x_0<=0 y −6.5<=x_1<=0 la función tiene
#     #múltiples mínimos locales y un único minimo global que se encuentra en
#     #f(−3.1302468,−1.5821422) = −106.7645367
#     f= np.sin(x_1)*np.exp(1-np.cos(x_0))**2 \
#         + np.cos(x_0)*np.exp(1-np.sin(x_1))**2 \
#         + (x_0-x_1)**2
#     return(f)

# print(test_function_1(820,821.5))
# import numpy as np
# def test_function_2(x_0,x_1):
#     f = -20*np.exp(-0.2*np.sqrt(0.5*(x_0**2+x_1**2))) \
#         - np.exp(0.5*(np.cos(2*np.pi*x_0)+np.cos(2*np.pi*x_1))) \
#         + np.e + 20
#     return(f)

# print(test_function_2(0,0))



