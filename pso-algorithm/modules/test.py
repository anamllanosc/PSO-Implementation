from modules.swarm import Swarm

class Test:
    """ Test object to run Pso Algorithm """

    def __init__(self, function, iterations: int = 100, particles: int = 200, bounds: int = 10) -> None:
        """
        Initialize a new instance of a test
        Arguments:
            function: function to optimize
            iterations(int): number of iterations to complete Algorithm
            Particles(int): numer of particles to create
            bounds(int): 
        """

        self.function = function
        self.iterations = iterations
        self.particles = particles
        self.bounds = bounds
    
    def run(self) -> None:
        """ Method to run the particle optimization """

        swarm = Swarm(objective_function=self.function, iterations=self.iterations, n_particles=self.particles, bounds=self.bounds)
        swarm.create_particles()
        swarm.optimize()

