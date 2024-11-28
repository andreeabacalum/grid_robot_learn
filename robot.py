

# Imports from external libraries
import numpy as np
import torch
import torch as nn
# Imports from this project
import constants
import configuration
from graphics import PathToDraw
import matplotlib.pyplot as plt
import torch.optim as optim
class BehavioralCloningModel(torch.nn.Module):

    # The class initialisation function.
    def __init__(self):
        # Call the initialisation function of the parent class.
        super(BehavioralCloningModel, self).__init__()
        # Define the network layers. This example network has two hidden layers, each with 10 units.
        self.layer_1 = torch.nn.Linear(in_features=2, out_features=50, dtype=torch.float32)
        self.layer_2 = torch.nn.Linear(in_features=50, out_features=50, dtype=torch.float32)
        self.layer_3 = torch.nn.Linear(in_features=50, out_features=50, dtype=torch.float32)
        self.output_layer = torch.nn.Linear(in_features=50, out_features=2, dtype=torch.float32)

    # Function which sends some input data through the network and returns the network's output.
    def forward(self, input):
        layer_1_output = torch.nn.functional.relu(self.layer_1(input))
        layer_2_output = torch.nn.functional.relu(self.layer_2(layer_1_output))
        layer_3_output = torch.nn.functional.relu(self.layer_3(layer_2_output))
        output = self.output_layer(layer_3_output)
        return output


class Robot:

    def __init__(self, goal_state):
        self.memory = []
        self.goal_state = goal_state
        self.paths_to_draw = []


        self.training_losses = []
        self.num_training_epochs = 0
        self.input_normalisation_factor = 0
        self.output_normalisation_factor = 0
        self.fig, self.ax = plt.subplots()
        self.ax.set(xlabel='Training Epochs', ylabel='Training Loss', title='Loss Curve for Policy Training')
        plt.yscale('log')
        self.model = BehavioralCloningModel()
        self.criterion = torch.nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters())
        self.demonstrations = []


    def get_next_action_type(self, state, money_remaining):
        # TODO: This informs robot-learning.py what type of operation to perform
        # It should return either 'demo', 'reset', or 'step'
        if True:
            return 'demo'
        if False:
            return 'reset'
        if False:
            return 'step'


    def get_next_action_training(self, state, money_remaining):
        # TODO: This returns an action to robot-learning.py, when get_next_action_type() returns 'step'
        # Currently just a random action is returned
        random_action = np.random.uniform([-constants.ROBOT_MAX_ACTION, constants.ROBOT_MAX_ACTION], 2)
        return random_action

    def get_next_action_testing(self, state):
        # TODO: This returns an action to robot-learning.py, when get_next_action_type() returns 'step'
        # Currently just a random action is returned
        random_action = np.random.uniform([-constants.ROBOT_MAX_ACTION, constants.ROBOT_MAX_ACTION], 2)
        return random_action

    # Function that processes a transition
    def process_transition(self, state, action, next_state, money_remaining):
        # TODO: This allows you to process or store a transition that the robot has experienced in the environment
        # Currently, nothing happens
        pass

    # Function that takes in the list of states and actions for a demonstration
    def process_demonstration(self, demonstration_states, demonstration_actions, money_remaining):
        # TODO: This allows you to process or store a demonstration that the robot has received
        # Currently, nothing happens
        # self.memory.append((demonstration_states, demonstration_actions, money_remaining))
        # Convert lists to PyTorch tensors
        states_tensor = torch.tensor(demonstration_states, dtype=torch.float32)
        actions_tensor = torch.tensor(demonstration_actions, dtype=torch.float32)
        # Store the processed demonstrations
        self.demonstrations.append((states_tensor, actions_tensor, money_remaining))

        # Function to train the model on the stored demonstrations

    # def train_on_demonstrations(self, num_epochs=50):
    #     # for epoch in range(num_epochs):
    #     #     total_loss = 0
    #     #     for states_tensor, actions_tensor, _ in self.demonstrations:
    #     #         self.model.train()
    #     #         self.optimizer.zero_grad()
    #     #         action_pred = self.model(states_tensor)
    #     #         loss = self.criterion(action_pred, actions_tensor)
    #     #         loss.backward()
    #     #         self.optimizer.step()
    #     #         total_loss += loss.item()
    #     #     print(f'Epoch {epoch}, Loss: {total_loss / len(self.demonstrations)}')
    #
    #     # Pad the demonstration data with zeros at the end, so that the robot learns to stop when it reaches the goal
    #     num_demos = len(self.demonstrations[0])
    #     num_pads = 50
    #     all_demonstration_states = np.zeros([0, 2], dtype=np.float32)
    #     all_demonstration_actions = np.zeros([0, 2], dtype=np.float32)
    #     for demo_num in range(num_demos):
    #         final_state = self.demonstrations[0][demo_num, -1]
    #         repeated_final_state = np.tile(final_state, (num_pads, 1))
    #         all_demonstration_states = np.concatenate((all_demonstration_states, self.demonstrations[0][demo_num]),
    #                                                   axis=0)
    #         all_demonstration_states = np.concatenate((all_demonstration_states, repeated_final_state), axis=0)
    #         final_action = np.array([0, 0], dtype=np.float32)
    #         repeated_final_action = np.tile(final_action, (num_pads, 1))
    #         all_demonstration_actions = np.concatenate((all_demonstration_actions, self.demonstrations[1][demo_num]),
    #                                                    axis=0)
    #         all_demonstration_actions = np.concatenate((all_demonstration_actions, repeated_final_action), axis=0)
    #     # Create the network data, by stacking the states and actions for each demonstration, and converting from numpy array to torch tensor
    #     network_input_data = torch.tensor(all_demonstration_states, dtype=torch.float32)
    #     network_label_data = torch.tensor(all_demonstration_actions, dtype=torch.float32)
    #     # Normalise the data
    #     self.input_normalisation_factor = torch.max(network_input_data)
    #     self.output_normalisation_factor = torch.max(network_label_data)
    #     network_input_data /= self.input_normalisation_factor
    #     network_label_data /= self.output_normalisation_factor
    #     # Create the optimiser
    #     optimiser = torch.optim.Adam(self.model.parameters(), lr=0.001)
    #     # Loop over epochs
    #     num_training_data = len(network_input_data)
    #     minibatch_size = 5
    #     num_minibatches = int(num_training_data / minibatch_size)
    #     loss_function = torch.nn.MSELoss()
    #     for epoch in range(25):
    #         # Set the learning rate depending on the number of epochs
    #         if epoch > 10:
    #             optimiser = torch.optim.Adam(self.model.parameters(), lr=0.001)
    #         elif epoch > 20:
    #             optimiser = torch.optim.Adam(self.model.parameters(), lr=0.0001)
    #         # Create a random permutation of the training indices
    #         permutation = torch.randperm(num_training_data)
    #         # Loop over minibatches
    #         training_epoch_losses = []
    #         for minibatch in range(num_minibatches):
    #             # Set all the gradients stored in the optimiser to zero.
    #             optimiser.zero_grad()
    #             # Get the indices for the training data based on the permutation
    #             training_indices = permutation[minibatch * minibatch_size: (minibatch + 1) * minibatch_size]
    #             minibatch_inputs = network_input_data[training_indices]
    #             minibatch_labels = network_label_data[training_indices]
    #             # Do a forward pass of the network using the inputs batch
    #             training_prediction = self.model.forward(minibatch_inputs)
    #             # Compute the loss based on the label's batch
    #             training_loss = loss_function(training_prediction, minibatch_labels)
    #             # Compute the gradients based on this loss
    #             training_loss.backward()
    #             # Take one gradient step to update the network
    #             optimiser.step()
    #             # Get the loss as a scalar value
    #             training_loss_value = training_loss.item()
    #             training_epoch_losses.append(training_loss_value)
    #         # Calculate the epoch loss
    #         training_epoch_loss = np.average(training_epoch_losses)
    #         # Store this loss in the list
    #         self.training_losses.append(training_epoch_loss)
    #         # Update the list of epochs
    #         self.num_training_epochs += 1
    #         training_epochs_list = range(self.num_training_epochs)
    #         # Plot and save the loss vs iterations graph
    #         self.ax.plot(training_epochs_list, self.training_losses, color='blue')
    #         plt.yscale('log')
    #         plt.show()


    def dynamics_model(self, state, action):
        # TODO: This is the learned dynamics model, which is currently called by graphics.py when visualising the model
        # Currently, it just predicts the next state according to a simple linear model, although the actual environment dynamics is much more complex
        next_state = state + action
        return next_state
    def compute_reward(self, path):
        reward = -np.linalg.norm(path[-1] - self.goal_state)
        return reward

    def cross_entropy_method(self, state):
        self.planning_actions = np.zeros(
            [constants.DEMOS_CEM_NUM_ITERATIONS, constants.DEMOS_CEM_NUM_PATHS, constants.DEMOS_CEM_PATH_LENGTH, 1, 2],
            dtype=np.float32)
        # planning_paths is the full set of paths (one path is a sequence of states) that are evaluated
        self.planning_paths = np.zeros(
            [constants.DEMOS_CEM_NUM_ITERATIONS, constants.DEMOS_CEM_NUM_PATHS, constants.DEMOS_CEM_PATH_LENGTH, 1, 2],
            dtype=np.float32)
        # planning_path_rewards is the full set of path rewards that are calculated
        self.planning_path_rewards = np.zeros([constants.DEMOS_CEM_NUM_ITERATIONS, constants.DEMOS_CEM_NUM_PATHS,])
        # planning_mean_actions is the full set of mean action sequences that are calculated at the end of each iteration (one sequence per iteration)
        self.planning_mean_actions = np.zeros([constants.DEMOS_CEM_NUM_ITERATIONS, constants.DEMOS_CEM_PATH_LENGTH, 1, 2],
                                              dtype=np.float32)
        # Loop over the iterations
        for iteration_num in range(configuration.CEM_NUM_ITERATIONS):
            for path_num in range(configuration.CEM_NUM_PATHS):
                planning_state = np.copy(state)
                for step_num in range(configuration.CEM_PATH_LENGTH):
                    if iteration_num == 0:
                        action = np.random.uniform(-constants.ROBOT_MAX_ACTION, constants.ROBOT_MAX_ACTION, [1, 2])
                    else:
                        action = np.random.normal(best_paths_action_mean[step_num], best_paths_action_std_dev[step_num])
                    self.planning_actions[iteration_num, path_num, step_num] = action
                    next_state = self.dynamics_model(planning_state, action)
                    self.planning_paths[iteration_num, path_num, step_num] = next_state
                    planning_state = next_state
                path_reward = self.compute_reward(self.planning_paths[iteration_num, path_num])
                self.planning_path_rewards[iteration_num, path_num] = path_reward
            sorted_path_rewards = self.planning_path_rewards[iteration_num].copy()
            sorted_path_costs = np.argsort(sorted_path_rewards)
            indices_best_paths = sorted_path_costs[-configuration.CEM_NUM_ELITES:]
            best_paths_action_mean = np.mean(self.planning_actions[iteration_num, indices_best_paths], axis=0)
            best_paths_action_std_dev = np.std(self.planning_actions[iteration_num, indices_best_paths], axis=0)
            self.planning_mean_actions[iteration_num] = best_paths_action_mean
        # Calculate the index of the best path
        index_best_path = np.argmax(self.planning_path_rewards[-1])
        # Set the planned path (i.e. the best path) to be the path whose index is index_best_path
        self.planned_path = self.planning_paths[-1, index_best_path]
        # Set the planned actions (i.e. the best action sequence) to be the action sequence whose index is index_best_path
        self.planned_actions = self.planning_actions[-1, index_best_path]


robot =Robot(goal_state = np.array([0.0, 0.0], dtype=np.float32))
print(robot.get_next_action_training(state=np.array([0.0, 0.0], dtype=np.float32), money_remaining=100))