import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import tempfile
from PIL import Image

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
            
            ax.scatter(x_positions, y_positions, s=10, c='b', zorder=10)
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
        
        # Save animation as GIF
        def save_animation_as_gif(ani, fps=5):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".gif") as temp_file:
                temp_file.close()
                ani.save(temp_file.name, writer='pillow', fps=fps)
                return temp_file.name

        video_path = save_animation_as_gif(ani)
        
        return video_path, self.global_best_position, self.global_best_value

class Test:
    def __init__(self, function, iterations: int = 100, particles: int = 200, bounds: int = 10, individual_inertia: int = 2, social_inertia: int = 2) -> None:
        self.function = function
        self.iterations = iterations
        self.particles = particles
        self.bounds = bounds
        self.social_inertia = social_inertia
        self.individual_inertia = individual_inertia
    
    def run(self) -> tuple:
        swarm = Swarm(objective_function=self.function, iterations=self.iterations, n_particles=self.particles, bounds=self.bounds, c1=self.individual_inertia, c2=self.social_inertia)
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
def function_7(x: int, height: int, weith: int):
    f = (height, 2*x)*(weith-2*x)*X
    return f

#########################################################################################
functions = [function_0, function_1, function_2, function_3, function_4, function_5, function_6, function_7]

st.markdown("## Particle Swarm Optimization functions")
tab1, tab2 = st.tabs(["Functions", "Run"])
with tab1:
    st.write("### Functions")
    col11, col21 = st.columns(2)
    with col11:
        st.write("#### 0. Sphere Function")
        """
                - Es una función cuadrática simple y convexa que se utiliza para probar algoritmos de optimización en espacios sin complejidad significativa.
                - Grafico:
        """
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/9/91/Sphere_contour.svg/200px-Sphere_contour.svg.png", width=400)
        """
                - Mínimo Global:
        """
        st.latex("f(x_{1}, x_{2}, \cdots, x_{n}) = f(0, \cdots, 0) = 0")
        st.write("#### 1. Function")
        st.write("#### 2. Ackley Function")
        """
                - Es una función no convexa que se utiliza para probar la capacidad de los algoritmos de escapar de mínimos locales. Tiene muchas oscilaciones, pero un único mínimo global en el origen. Es un desafío debido a su paisaje rugoso, que puede dificultar la convergencia de los algoritmos.
                - Grafico"""
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/9/9f/Ackley_2d.png/300px-Ackley_2d.png", width=400)
        """
                - Mínimo Global:
        """
        st.latex("f(0, 0) = 0")
        st.write("#### 3. Booth Function")
        """
                - Es una función bidimensional con un mínimo global en un punto específico, también usada para probar la precisión de los algoritmos en un entorno más simple.
                - Grafico:"""
        
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/5/54/Booth_contour.svg/200px-Booth_contour.svg.png", width=400)
        """
                - Mínimo Global:
        """
        st.latex("f(1,3)=0")
    with col21: 
        st.write("#### 4. Rastrigin Function")
        """
                - Es muy usada para probar la robustez de los algoritmos debido a sus múltiples mínimos locales distribuidos uniformemente.
                - Grafico:
        """
        st.image("https://upload.wikimedia.org/wikipedia/commons/8/8b/Rastrigin_function.png", width=400)
        """
                - Mínimo Global:
        """
        st.latex("f(0, \cdots, 0) = 0")
        st.write("#### 5. Rosenbrock Function")
        """
        - Conocida como la función del "valle", tiene un único mínimo global en el que los algoritmos deben encontrar un camino a lo largo de un valle estrecho y curvado. Es útil para probar la precisión de los algoritmos en problemas de optimización no lineales.
        - Grafico:
        """
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/1/11/Rosenbrock3.gif/300px-Rosenbrock3.gif", width=400)
        """
        """
        #st.latex()
        st.write("#### 6. Holder Table Function")
        """
        - Esta función es multimodal con varios mínimos globales. Es desafiante para los algoritmos porque tiene un paisaje complicado con muchos mínimos locales, además de que su función objetivo incluye términos exponenciales y trigonométricos.
        - Grafico:
        """
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/f/fc/Hoelder_table_contour.svg/1280px-Hoelder_table_contour.svg.png", width=400)
        st.write("#### 7. Personalized Function")
        """
            - La caja abierta se fabricará de 24 cm por 36 cm de cartón quitando un cuadrado de cada esquina de la caja y doblando las solapas de cada lado. ¿De qué tamaño es el cuadrado que hay que cortar en cada esquina para obtener una caja con el máximo volumen?
            - Volumen= longitud*ancho*alto
            - La altura de la caja es X cm
            - H= La longitud es de 36-2x cm (Altura del rectángulo menos dos veces la altura de la caja (el lado de la esquina que se quita) )
            - A= El ancho es de 24-2x cm ( Ancho del rectángulo menos dos veces la altura de la caja (el lado de la esquina que se quita)  )
            - Volumen de la caja
        """
        st.latex("V(x)=(36-2x)(24-2x)x=4x³-120x²+864x.")
        st.latex("V(x)=(H-2x)(A-2x)x= (HA)x-(2A + 2H)x²+4x³")
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
                               "3 - Booth", "4 - Rastrigin", "5 - Rosenbrock", 
                               "6 - Holder Table", "7 - Personalized"))
        option = values[option]
    with col22:
        st.write("### Parameters")
        iterations = st.number_input("Iterations", value=100)
        particles = st.number_input("Particles", value=200)
        bounds = st.number_input("Bounds", value=10)
        col1, col2 = st.columns(2)
        with col1:
            if option == 7:  # For function_7, which has additional parameters
                with st.expander("Personalized Function"):
                    waited_demand = st.number_input("Height", value=0)
                    total_products = st.number_input("width", value=0)
            else:
                st.empty()
                sale_price = waited_demand = total_products = storage_cost_per_product = None
        with col2:
            with st.expander("More configurations"):
                individual_inertia = st.number_input("Individual Inertia", value=2)
                social_inertia = st.number_input("Social Inertia", value=2)


    st.divider()

    if st.button("Run Optimization"):
        function_to_run = functions[option]
        if option == 7:
            function_to_run = lambda x, y: function_7(x, height, weith)
        video_path, best_position, best_value = Test(function=function_to_run, iterations=iterations, particles=particles, bounds=bounds).run()
        video_file = open(video_path, "rb")
        video_bytes = video_file.read()
        with st.expander("Results"):
            st.success(f"Function Optimized", icon="✅")
            st.image(video_bytes, use_column_width=True)
            st.write(f"Best value found: ", best_value)
            st.write(f"Best position found: ", best_position)
