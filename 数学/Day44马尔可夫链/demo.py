import numpy as np



transition_matrix = np.array([
    [0.7, 0.2, 0.1],
    [0.3, 0.4, 0.3],
    [0.2, 0.3, 0.5],
])

states = ['Sunny','Cloudy','Rainy']

initial_state_distribution =  np.array([1,0,0])


def calculate_stationary_distribution(transition_matrix):
    eigenvalues,eigenvectors=np.linalg.eig(transition_matrix.T)
    stationary_distribution=eigenvectors[:,np.isclose(eigenvalues,1)]
    stationary_distribution = stationary_distribution / stationary_distribution.sum()
    return stationary_distribution.real.flatten()
def simulate_markov_chain(transition_matrix,initial_state_distribution,steps):
    current_state = np.random.choice(len(states),p=initial_state_distribution)
    print("Day 0:",states[current_state])

    for day in range(1,steps+1):
        current_state = np.random.choice(len(states),
                                         p=transition_matrix[current_state])
        print(f"Day {day}:",states[current_state])


num_days = 10
simulate_markov_chain(transition_matrix,initial_state_distribution,num_days)


stationary_dist = calculate_stationary_distribution(transition_matrix)
print("平稳分布:",{state: prob for state,prob in zip(states,stationary_dist)})