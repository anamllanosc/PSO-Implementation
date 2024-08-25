import numpy as np
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
