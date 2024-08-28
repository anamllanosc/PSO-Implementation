import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

st.title("Optimización por Enjambre de Partículas (PSO)")

# Introducción
st.header("¿Qué es PSO?")
st.write("""
Es un algoritmo de optimización inspirado en el comportamiento de animales en grupo, como aves y peces. Está orientado a encontrar los máximos o mínimos globales en un espacio de búsqueda.

- **Cooperación de los agentes:** 
  El PSO utiliza un enjambre de partículas (soluciones) que se mueven en el espacio de búsqueda. Cada partícula ajusta su dirección y velocidad basándose en su propia experiencia y la de sus vecinos.

- **Objetivo:** Encontrar la mejor solución en el espacio de búsqueda.
""")

# Imagen de PSO
st.image("https://raw.githubusercontent.com/keurfonluu/stochopy/master/.github/sample.gif", caption='Ejemplo de movimiento de partículas en PSO', width= 1000)

# Pasos del algoritmo
st.header("Pasos del Algoritmo")
st.write("""
1. **Creación del enjambre inicial:**
   - **Posición:** Combinación de valores de las variables.
   - **Velocidad:** Indica el desplazamiento de la partícula.
   - **Registro:** Mejor posición obtenida por la partícula hasta el momento.

2. **Evaluación:** Calcula el valor de la función objetivo en la posición actual de cada partícula.

3. **Actualización:**
   - Actualiza la posición y velocidad según la mejor posición personal y global.
""")

# Fórmulas
st.latex(r"v_i(t+1) = wv_i(t) + c_1r_1[x^i(t) - x_i(t)] + c_2r_2[g(t) - x_i(t)]")
st.latex(r"x_i(t+1) = x_i(t) + v_i(t+1)")

# Historia
st.header("Historia")
st.write("""
El algoritmo de PSO fue introducido por James Kennedy y Russell Eberhart en 1995. Inspirado en el comportamiento de enjambres de animales, PSO simula el movimiento grupal para resolver problemas de optimización.

- **1995:** Publicación inicial en la IEEE International Conference on Neural Networks.
- **1997:** Expansión en el libro "Swarm Intelligence".
- **Finales de 1990 y 2000s:** Consolidación y variantes del algoritmo.
- **Última década:** Mejora continua y aplicaciones en problemas complejos.
""")
