<h1 align="center">Particle swarm optimization (PSO)</h1>
<div align="center">
    
###### POO = {a, j, d : (a $\in$ II) $\land$ (d, j $\in$ CC)}
[![Status](https://img.shields.io/badge/status-active-success.svg)]()
[![GitHub Issues](https://img.shields.io/github/issues/kylelobo/The-Documentation-Compendium.svg)](https://github.com/anamllanosc/PSO-Implementation/issues)
[![GitHub Pull Requests](https://img.shields.io/github/issues-pr/kylelobo/The-Documentation-Compendium.svg)](https://github.com/anamllanosc/PSO-Implementation/pulls)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](/LICENSE)
</div>

----
## ℹ️ Table of Contents

- [About](#about)
- [Getting Started](#getting_started)
- [Authors](#authors)
- [Class Diagram](#diagram)
- [Repo Page](https://anamllanosc.github.io/PSO-Implementation/)

## 📂 About <a name = "about"></a>

PSO implementation using OOP

## 👾 Getting Started <a name = "getting_started"></a>

### 😒 Prerequisites 

Soon you will see some prerequisites

### 🎩 Installing 

Here you will have the steps to install the project

## 🤵 Authors <a name = "authors"></a>

- [@anamllanosc](https://github.com/anamllanosc)
- [@jorge9805](https://github.com/jorge9805)
- [@estfloyd](https://github.com/estfloyd)

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
