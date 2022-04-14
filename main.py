import tensorflow
from tensorflow import keras as k
import pygad
import pygad.kerasga as kga

from DataPreprocessing import dataPreprocessing

# model creation
input_layer = k.layers.Input(19)
dense_layer = k.layers.Dense(11, activation="relu")
output_layer = k.layers.Dense(5, activation="softmax")

model = k.Sequential()
model.add(input_layer)
model.add(dense_layer)
model.add(output_layer)

# ga setup
num_individuals = 10
ga = kga.KerasGA(model = model, num_solutions=num_individuals)

data_inputs, data_outputs = dataPreprocessing(num_individuals)

def fitness_func(solution, sol_idx):
    global data_inputs, data_outputs, ga, model

    model_weights_matrix = kga.model_weights_as_matrix(model=model, weights_vector=solution)

    model.set_weights(weights=model_weights_matrix)

    predictions = model.predict(data_inputs)

    cce = k.losses.CategoricalCrossentropy()
    solution_fitness = 1.0 / cce(data_outputs, predictions).numpy()

    return solution_fitness

def callback_generation(ga_instance):
    print("Generation = {generation}".format(generation=ga_instance.generations_completed))
    print("Fitness    = {fitness}".format(fitness=ga_instance.best_solution()[1]))


num_generations = 100
num_parents_mating = 5
initial_population = ga.population_weights

ga_instance = pygad.GA(num_generations=num_generations,
                        num_parents_mating=num_parents_mating,
                        initial_population=initial_population,
                        fitness_func=fitness_func,
                        on_generation=callback_generation)

ga_instance.run()

ga_instance.plot_result(title="PyGAD & Keras - Iteration vs. Fitness", linewidth=4)

# Returning the details of the best solution.
solution, solution_fitness, solution_idx = ga_instance.best_solution()
print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))
print("Index of the best solution : {solution_idx}".format(solution_idx=solution_idx))

# Fetch the parameters of the best solution.
best_solution_weights = pygad.kerasga.model_weights_as_matrix(model=model,
                                                              weights_vector=solution)
model.set_weights(best_solution_weights)
predictions = model.predict(data_inputs)
# print("Predictions : \n", predictions)

# Calculate the categorical crossentropy for the trained model.
cce = tensorflow.keras.losses.CategoricalCrossentropy()
print("Categorical Crossentropy : ", cce(data_outputs, predictions).numpy())

# Calculate the classification accuracy for the trained model.
ca = tensorflow.keras.metrics.CategoricalAccuracy()
ca.update_state(data_outputs, predictions)
accuracy = ca.result().numpy()
print("Accuracy : ", accuracy)