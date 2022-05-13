# For the use of the GPU with tensorflow
import os
os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.6/bin")

import tensorflow as tf
from tensorflow import keras as k
import pygad
import pygad.kerasga as kga
from dataPreprocessing import dataPreprocessing

# creation of the model
input_layer = k.layers.Input(19)
dense_layer = k.layers.Dense(12, activation="relu")
output_layer = k.layers.Dense(5, activation="sigmoid")

model = k.Sequential()
model.add(input_layer)
model.add(dense_layer)
model.add(output_layer)

# ga setup
# Prepare train set 70%
train_size = 58000
x_train, y_train = dataPreprocessing(None, train_size)

ga = kga.KerasGA(model = model, num_solutions = 100)

# fitness function 
def fitness_func(solution, sol_idx):
    model_weights_matrix = kga.model_weights_as_matrix(model=model, weights_vector=solution)
    model.set_weights(weights=model_weights_matrix)
    predictions = model.predict(x_train)

    cce = k.losses.CategoricalCrossentropy()
    solution_fitness = 1.0 / cce(y_train, predictions).numpy()

    return solution_fitness

# callback function
def callback_generation(ga_instance):
    print("Generation = {generation}".format(generation=ga_instance.generations_completed))
    print("Best Fitness   = {fitness}".format(fitness=ga_instance.best_solution()[1]))


num_generations = 100
num_parents_mating = 10
initial_population = ga.population_weights

ga_instance = pygad.GA(num_generations=num_generations,
                        num_parents_mating=num_parents_mating,
                        initial_population=initial_population,
                        fitness_func=fitness_func,
                        on_generation=callback_generation,
                        parent_selection_type="rws",
                        crossover_type="uniform",
                        crossover_probability=0.5,
                        mutation_type="scramble",
                        mutation_probability=0.1,
                        save_best_solutions=True,
                        stop_criteria="saturate_7")

ga_instance.run()

ga_instance.plot_fitness(title="PyGAD & Keras - Iteration vs. Fitness", linewidth=4)

# Returning the details of the best solution.
solution, fitness, idx = ga_instance.best_solution()
print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=fitness))
print("Index of the best solution : {solution_idx}".format(solution_idx=idx))

# Fetch the parameters of the best solution.
best_solution_weights = pygad.kerasga.model_weights_as_matrix(model=model,
                                                              weights_vector=solution)
model.set_weights(best_solution_weights)
predictions = model.predict(x_train)
#print("Predictions : \n", predictions)

# Calculate the categorical crossentropy for the trained model.
cce = k.losses.CategoricalCrossentropy()
print("Categorical Crossentropy : ", cce(y_train, predictions).numpy())

# Calculate the classification accuracy for the trained model.
acc = tf.keras.metrics.BinaryAccuracy(threshold = 0.5)
acc.update_state(y_train, predictions)
accuracy = acc.result().numpy()
print("Accuracy : ", accuracy)

pre = tf.keras.metrics.Precision()
pre.update_state(y_train, predictions)
precision = pre.result().numpy()

rec = tf.keras.metrics.Recall()
rec.update_state(y_train, predictions)
recall = rec.result().numpy()

# Save model as json
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# Save weights as HDF5
model.save_weights("model.h5")
