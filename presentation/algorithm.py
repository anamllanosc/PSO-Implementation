import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import tempfile

class Particle:
    def __init__(self, dim: int, bounds: float):
        self.position = np.random.uniform(-bounds, bounds, dim)
        self.velocity = np.random.uniform(-bounds, bounds, dim)
        self.personal_best_position = np.zeros(dim)
        self.personal_best_value = None
        self.value = None
        self.bounds = bounds

    def evaluate(self, objective_function):
        self.value = objective_function(*self.position)

    def update_velocity(self, global_best_position: np.array, w: float, c1: float, c2: float, r1: np.array, r2: np.array):
        self.velocity = w * self.velocity + c1 * r1 * (self.personal_best_position - self.position) + c2 * r2 * (global_best_position - self.position)

    def update_position(self):
        self.position = np.clip(self.position + self.velocity, -self.bounds, self.bounds)

    def update_personal_best(self, optimization_method: str = "min"):
        if self.personal_best_value is None or (optimization_method == "min" and self.value < self.personal_best_value) or (optimization_method == "max" and self.value > self.personal_best_value):
            self.personal_best_value = self.value
            self.personal_best_position = self.position

class Swarm:
    def __init__(self, n_particles: int = 50, particles: list = [], dim: int = 2, w_max: float = 0.9, w_min: float = 0.4, c1: float = 2, c2: float = 2, bounds: float = 200, iterations: int = 100, objective_function: callable = lambda x, y: x**2 + y**2, optimization_method: str = "min"):
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
        self.global_best_position = np.full(dim, np.nan)
        self.global_best_value = None
        self.iterations = iterations
        self.optimization_method = optimization_method

    def lineal_reduction_inertia(self, iteration: int):
        self.w = (self.w_max - self.w_min) * (self.iterations - iteration) / self.iterations + self.w_min

    def create_particles(self):
        for i in range(self.n_particles):
            self.particles.append(Particle(dim=self.dim, bounds=self.bounds))

    def find_global_best(self):
        for particle in self.particles:
            if self.global_best_value is None or (self.optimization_method == "min" and particle.personal_best_value < self.global_best_value) or (self.optimization_method == "max" and particle.personal_best_value > self.global_best_value):
                self.global_best_value = particle.personal_best_value
                self.global_best_position = particle.personal_best_position

    def optimize(self):
        x_0 = np.linspace(-self.bounds, self.bounds, 100)
        x_1 = np.linspace(-self.bounds, self.bounds, 100)
        X_0, X_1 = np.meshgrid(x_0, x_1)
        Z = self.objective_function(X_0, X_1)
        
        fig, ax = plt.subplots()
        ax.contour(X_0, X_1, Z, levels=35, cmap='RdGy')
        
        def update(frame):
            ax.clear()
            ax.contour(X_0, X_1, Z, levels=35, cmap='RdGy')
            x_positions = []
            y_positions = []
            self.find_global_best()
            
            for particle in self.particles:
                r1 = np.random.uniform(0, 1, self.dim)
                r2 = np.random.uniform(0, 1, self.dim)
                particle.update_velocity(self.global_best_position, self.w, self.c1, self.c2, r1, r2)
                particle.update_position()
                particle.evaluate(self.objective_function)
                particle.update_personal_best(self.optimization_method)
                x_positions.append(particle.position[0])
                y_positions.append(particle.position[1])
            
            ax.scatter(x_positions, y_positions, s=10, c='b')
            ax.set_xlim(-self.bounds, self.bounds)
            ax.set_ylim(-self.bounds, self.bounds)
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_title(f"Iteration: {frame}")
            
            self.lineal_reduction_inertia(frame)
            return ax.collections
        
        ani = animation.FuncAnimation(fig, update, frames=self.iterations, repeat=False)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
            ani.save(temp_file.name, writer='ffmpeg', fps=5)
            video_path = temp_file.name

        return video_path

class Test:
    def __init__(self, function, iterations: int = 100, particles: int = 200, bounds: int = 10) -> None:
        self.function = function
        self.iterations = iterations
        self.particles = particles
        self.bounds = bounds
    
    def run(self) -> None:
        swarm = Swarm(objective_function=self.function, iterations=self.iterations, n_particles=self.particles, bounds=self.bounds)
        swarm.create_particles()
        video_path = swarm.optimize()
        st.video(video_path)

#########################################################################################
#################################### Test Functions #####################################
#########################################################################################

### sphere function ###
def function_0(x,y):
    return x**2+y**2         
#function from https://cienciadedatos.net/documentos/py02_optimizacion_pso

### Function ###
def function_1(x_0,x_1):
    #Para la región acotada entre −10<=x_0<=0 y −6.5<=x_1<=0 la función tiene
    #múltiples mínimos locales y un único minimo global que se encuentra en
    #f(−3.1302468,−1.5821422) = −106.7645367
    f= np.sin(x_1)*np.exp(1-np.cos(x_0))**2 \
        + np.cos(x_0)*np.exp(1-np.sin(x_1))**2 \
        + (x_0-x_1)**2
    return(f)

#### ackely function ###
def function_2(x_0,x_1):
    f = -20*np.exp(-0.2*np.sqrt(0.5*(x_0**2+x_1**2))) \
        - np.exp(0.5*(np.cos(2*np.pi*x_0)+np.cos(2*np.pi*x_1))) \
        + np.e + 20
    return(f)

### rastrigin function ###
def function_3(x_0,x_1):
    f = 20 + x_0**2 + x_1**2 - 10*(np.cos(2*np.pi*x_0)+np.cos(2*np.pi*x_1))
    return(f)

### rosenbrock function ###
def function_4(x_0,x_1):
    f = 100*(x_1-x_0**2)**2 + (1-x_0)**2
    return(f)

### holder table function ###
def function_5(x_0,x_1):
    f = -np.abs(np.sin(x_0)*np.cos(x_1)*np.exp(np.abs(1-np.sqrt(x_0**2+x_1**2)/np.pi)))
    return(f)

#######################################################################################

functions = [function_0, function_1, function_2, function_3, function_4, function_5]
option = st.selectbox("Which function do you want to optimize?", (0, 1, 2, 3, 4, 5))

if st.button("Run Optimization"):
    Test(function=functions[option]).run()

