import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import tempfile

## To - Dos
# 1. Make personalized function work (Ana)
# 2. write a description for each function (Ana)
# 3. Make work the personalized function (Jorge)


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

    def optimize(self, progress_callback=None):
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
            
            # Update progress bar
            if progress_callback:
                progress_callback(frame / self.iterations)
            
            return ax.collections
        
        ani = animation.FuncAnimation(fig, update, frames=self.iterations, repeat=False)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
            ani.save(temp_file.name, writer='ffmpeg', fps=5)
            video_path = temp_file.name

        return video_path, self.global_best_position, self.global_best_value

class Test:
    def __init__(self, function, iterations: int = 100, particles: int = 200, bounds: int = 10) -> None:
        self.function = function
        self.iterations = iterations
        self.particles = particles
        self.bounds = bounds
    
    def run(self) -> tuple:
        swarm = Swarm(objective_function=self.function, iterations=self.iterations, n_particles=self.particles, bounds=self.bounds)
        swarm.create_particles()
        

        # Display progress bar
        progress_bar = st.progress(0, text="Optimizing...")
        
        def update_progress(progress):
            progress_bar.progress(int(progress * 100), text=f"Optimizing... {int(progress * 100)}%")
        
        video_path, best_position, best_value = swarm.optimize(progress_callback=update_progress)
        progress_bar.empty()
        return video_path, best_position, best_value
#########################################################################################
#################################### Test Functions #####################################
#########################################################################################

### sphere function ###
def function_0(x, y):
    return x**2 + y**2

### Function ###
def function_1(x_0, x_1):
    f = np.sin(x_1) * np.exp(1 - np.cos(x_0))**2 \
        + np.cos(x_0) * np.exp(1 - np.sin(x_1))**2 \
        + (x_0 - x_1)**2
    return f

#### ackley function ###
def function_2(x_0, x_1):
    f = -20 * np.exp(-0.2 * np.sqrt(0.5 * (x_0**2 + x_1**2))) \
        - np.exp(0.5 * (np.cos(2 * np.pi * x_0) + np.cos(2 * np.pi * x_1))) \
        + np.e + 20
    return f

### rastrigin function ###
def function_6(x_0, x_1):
    f = 20 + x_0**2 + x_1**2 - 10 * (np.cos(2 * np.pi * x_0) + np.cos(2 * np.pi * x_1))
    return f

### rosenbrock function ###
def function_4(x_0, x_1):
    f = 100 * (x_1 - x_0**2)**2 + (1 - x_0)**2
    return f

### holder table function ###
def function_5(x_0, x_1):
    f = -np.abs(np.sin(x_0) * np.cos(x_1) * np.exp(np.abs(1 - np.sqrt(x_0**2 + x_1**2) / np.pi)))
    return f

### Booth function ###
def function_3(x, y):
    f = (x + 2*y - 7)**2 + (2*x + y - 5)**2
    return f

## Personalized function ##
def function_7(x_0, x_1, sale_price, waited_demand, total_products, storage_cost_per_product):
    sale_gain = sale_price * np.minimum(x_0, waited_demand)
    storage_cost = waited_demand * storage_cost_per_product
    f = np.sum(sale_gain) - np.sum(storage_cost)
    return f
#######################################################################################

functions = [function_0, function_1, function_2, function_3, function_4, function_5, function_6, function_7]

st.markdown("## Particle Swarm Optimization functions")
tab1, tab2 = st.tabs(["Functions", "Run"])
with tab1:
    st.write("### Functions")
    col11, col21 = st.columns(2)
    with col11:
        st.write("#### 0. Sphere Function")
        st.write("#### 1. Function")
        st.write("#### 2. Ackley Function")
        st.write("#### 3. Booth Function")
    with col21: 
        st.write("#### 4. Rastrigin Function")
        st.write("#### 5. Rosenbrock Function")
        st.write("#### 6. Holder Table Function")
        st.write("#### 7. Personalized Function")
with tab2:
    values: dict = {
        "0 - Sphere": 0,
        "1 - Function": 1,
        "2 - Ackley": 2,
        "3 - Booth": 3,
        "4 - Rastrigin": 4,
        "5 - Rosenbrock": 5,
        "6 - Holder Table": 6,
        "7 - Personalized": 7
            }
    col12, col22 = st.columns(2)
    with col12:
        st.write("### Select Function")
        option = st.selectbox("Which function do you want to optimize?", 
                              ("0 - Sphere", "1 - Function", "2 - Ackley" , 
                               "3 - Booth", "4 -  Rastring", "5 - Rosenbrock", 
                               "6 - Holder Table", "7 - Personalized"))
        option = values[option]
    with col22:
        st.write("### Parameters")
        iterations = st.number_input("Iterations", value=100)
        particles = st.number_input("Particles", value=200)
        bounds = st.number_input("Bounds", value=10)
        
        if option == 7:  # For function_6, which has additional parameters
            with st.popover("Personalized Function"):
                sale_price = st.number_input("Sale Price", value=0)
                waited_demand = st.number_input("Waited Demand", value=0)
                total_products = st.number_input("Total Products", value=0)
                storage_cost_per_product = st.number_input("Storage Cost per Product", value=0)
        else:
            sale_price = waited_demand = total_products = storage_cost_per_product = None

    st.divider()

    if st.button("Run Optimization"):
        function_to_run = functions[option]
        if option == 7:
            function_to_run = lambda x, y: function_6(x, y, sale_price, waited_demand, total_products, storage_cost_per_product)
        video_path, best_position, best_value = Test(function=function_to_run, iterations=iterations, particles=particles, bounds=bounds).run()

    with st.expander("Results"):
        st.success(f"Function Optimized", icon="✅")
        st.video(video_path)
        st.write(f"Best value found: ", best_value)
        st.write(f"Best position found: ", best_position)
        
