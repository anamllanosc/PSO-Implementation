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
    Particle *--Enjambre
    class Particle{
        +vector n_variables
        +vector inf_limit
        +vector sup_limit
        +vector position
        +vector speed
        +float value
        +vector best_position
        +float best_value
        -set_value(self, value)
        -set_best_value()
        -set_best_position()
        -move_particle()
    }
    class Enjambre {
        -particles: list
        -n_particles: int
        -n_variables: int
        -lower_limits: list or ndarray
        -upper_limits: list or ndarray
        -best_particle: object Particle
        -best_value: float
        -best_position: ndarray
        -particles_history: list
        -best_position_history: list
        -best_value_history: list
        -absolute_difference: list
        -results_df: DataFrame
        -optimal_value: float
        -optimal_position: ndarray
        -optimized: bool
        -optimization_iterations: int
        -verbose: bool
        +__init__(n_particles, n_variables, lower_limits, upper_limits, verbose)
        +__repr__()
        +show_particles(n=None)
        +evaluate_swarm(objective_function, optimization, verbose=False)
        +move_swarm(inertia, cognitive_weight, social_weight, verbose=False)
        +optimize(objective_function, optimization, n_iterations, inertia, reduce_inertia=True, max_inertia=, min_inertia=, cognitive_weight, social_weight, early_stop=False, stop_rounds=None, stop_tolerance=None, verbose=False)
    }

```
--------
