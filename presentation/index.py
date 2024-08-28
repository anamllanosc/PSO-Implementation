import streamlit as st
""" ## Introduction"""
col1, col2, col3 = st.columns(3)

with col1:
    """
### What is?
Es un algoritmo de optimización inteligente inspirado en el comportamiento de animales en grupo. Esta orientado a encontrar máximos o mínimo globales.

- Cooperación de los medios.

La optimización por enjambre de partículas contiene una población de soluciones factibles llamadas enjambre, por lo que se tiene un enjambre de soluciones llamadas partículas. El movimiento de cada individuo (variación de dirección, velocidad y aceleración), se da a partir de las decisiones anteriores y comportamiento del resto.

El PSO busca la solución factible al problema de optimización que vamos a resolver. 

- Toda partícula tiene una posición en el espacio de búsqueda.
- El espacio es el conjunto de todas las posibles soluciones a la optimización.

**Objetivo** --> Encontrar la mejor solución de todo ese espacio.
"""
with col2:
    """### Steps
1. **Creación del enjambre inicial:** 
    
    - Posición: Determinada combinación de valores de las variables.
    
    - Velocidad: Indica como y hacia donde se desplaza la partícula
    
    - Registro: Registro de la mejor posición que ha tenido la partícula hasta el momento.
    

2. **Evaluar cada partícula con la función objetivo.**
    
    - Calcular el valor de la función objetivo en la posición que ocupa la partícula en ese momento.
    

3. **Actualización:**
    
    - Se actualizan los valores de la posición y la velocidad en función de la posición donde ha obtenido mejores resultados hasta el momento y la mejor posición encontrada por el enjambre hasta el momento.
"""
    st.latex("vi(t+1)=wvi(t)+c1r1[x^i(t)-xi(t)]+c2r2[g(t)-xi(t)]")
    st.latex("xi(t+1)=xi(t)+vi(t+1)")

with col3:
    """
### History

El algoritmo de Optimización por Enjambre de Partículas (PSO) fue propuesto por James Kennedy y Russell Eberhart en 1995. Inspirado en el comportamiento de enjambres de animales, como aves y peces, PSO simula el movimiento grupal para encontrar soluciones a problemas de optimización. La primera publicación sobre PSO, presentada en el IEEE International Conference on Neural Networks en 1995, introdujo el algoritmo y sus aplicaciones iniciales.

En 1997, Kennedy y Eberhart publicaron una versión extendida en su libro "Swarm Intelligence", profundizando en los detalles y mejorando el algoritmo. A finales de la década de 1990 y principios de 2000, PSO se consolidó con variantes que abordaron problemas de convergencia y exploración.

En la última década, PSO ha continuado evolucionando con mejoras en eficiencia y precisión, y se ha aplicado a problemas complejos en diversas áreas, incluyendo optimización multidimensional y ajuste de parámetros en redes neuronales. La investigación actual sigue explorando nuevas variantes y aplicaciones para abordar desafíos emergentes.

    """

