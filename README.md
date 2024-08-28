<h1 align="center">Particle swarm optimization (PSO)</h1>
<div align="center">
    
###### POO = {a, j, d : (a $\in$ II) $\land$ (d, j $\in$ CC)}
[![Repo Page](https://img.shields.io/badge/GitHub-Page-blue?style=plastic&logo=github)](https://anamllanosc.github.io/PSO-Implementation/)
[![Repo Issues](https://img.shields.io/github/issues/anamllanosc/PSO-Implementation?style=plastic)](https://github.com/anamllanosc/PSO-Implementation/issues)
[![Repo Pull Requests](https://img.shields.io/github/issues-pr/anamllanosc/PSO-Implementation?style=plastic)](https://github.com/anamllanosc/PSO-Implementation/pulls)
</div>

----
## ℹ️ Table of Contents

- [About](#about)
- [Getting Started](#getting_started)
- [Authors](#authors)
- [Class Diagram](#diagram)

## 📂 About <a name = "about"></a>
This repo is for hosting the code of the Particle Swarm Optimization (PSO) algorithm. 
Made as the finall project for the OOP course given by @fegonzalez7 at the 
National University of Colombia.

## 👾 Getting Started <a name = "getting_started"></a>

### 😒 Prerequisites 
1. For the base package:
    - Numpy
    - Matplotlib

2. For the page deployment:
    - Streamlit

### 🎩 Installing 
1. Clone the repo:
```bash
git clone https://github.com/anamllanosc/PSO-Implementation.git
```
2. Install the requirements:
```bash
pip install -r requirements.txt
```
3. Run the main file:
```bash
python main.py
```

## 🤵 Authors <a name = "authors"></a>

- [@anamllanosc](https://github.com/anamllanosc)
- [@jorge9805](https://github.com/jorge9805)
- [@dmeloca](https://github.com/dmeloca)

### 📰 Class Diagram <a name="diagram"></a>
```mermaid

classDiagram
    Particle *--Swarm
    Animation *--Swarm
    Swarm *--Test
    class Particle{
        +int dim
        +float bound
        +position 
        +velocity
        +personal_best_position
        +personal_best_value
        +value
        -evaluate(self,objective_function)
        -update_velocity(self, global_best_position:np.array,w:float,c1:float,c2:float,r1:np.array,r2:np.array)
        -update_position(self)
        -update_personal_best(self,optmimization_method:str="min")
    }
    class Swarm {
        -particles: list
        -int n_particles = 50
        -list particles
        -int dim = 2
        -float w_max = 0.9
        -float w_min = 0.4
        -float c1 = 2
        -float c2 =2
        -float bounds = 200
        -global_best_position
        -global_best_value
        -int iterations = 100 
        -callable objective_function = lambda x,y: x**2+y**2
        -str optimization_method = "min"
        +lineal_reduction_inertia(self,iteration:int)
        +create_particles(self)
        +find_global_best(self)
        +optimize(self)
    }
    class Test {
        -function
        -int iterations
        -int particles
        -int bounds
        +run(self)
    }
    class Animation{
        -bounds
        -objetive_function
        -x_positions
        -y_positions
        +animateFunction(self)
        +animateCountour(self,cmap='RdYlGn')
        +animateScatter(self,size=2,color='b')
        +animate(self,velocity=5,cmap='RdYlGn',size=2,color='b')
        +clear(self)
        +show(self)
    }


```
--------
