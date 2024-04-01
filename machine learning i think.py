import numpy as np

# Define the graph as a dictionary
graph = {
    'A': {'B': 5, 'C': 2},
    'B': {'A': 5, 'C': 1, 'D': 3},
    'C': {'A': 2, 'B': 1, 'D': 1},
    'D': {'B': 3, 'C': 1, 'E': 2},
    'E': {'D': 2}
}

# Define the reinforcement learning agent
class Agent:
    def __init__(self, learning_rate, discount_factor, exploration_rate, revisit_penalty,length_penalty):
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.revisit_penalty = revisit_penalty
        self.q_table = {}
        self.mem_table={}
        self.mem_state=0
        self.visited_nodes = set()
        self.length_penalty = length_penalty
        self.age = 0
    
    def choose_action(self, state):
        # Choose an action based on the current state using an epsilon-greedy policy
        meh=np.random.uniform()
        smeh=self.exploration_rate

        if meh < smeh:
            # Choose a random action
            
            action = np.random.choice(list(graph[state].keys()))
           
            
        else:
            # Choose the action with the highest Q-value
            q_values = [self.q_table.get((state, a),0) for a in graph[state].keys()]
            self.mem_table = self.q_table
            self.mem_state = state


                
            max_q_value = np.max(q_values)
         
            best_actions = [a for a, q in zip(graph[state].keys(), q_values) if q == max_q_value]
            
                
            action = np.random.choice(best_actions)
            
        
           
            
        return action
    
    def update_q_table(self, state, action, reward, next_state):
        # Update the Q-value for the current state and action
        current_q = self.q_table.get((state, action), 0)
        
        next_max_q = np.max([self.q_table.get((next_state, a), 0) for a in graph[next_state].keys()])
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * next_max_q - current_q)

        self.q_table[(state, action)] = new_q
        print(self.q_table)
        
    def get_reward(self, state, next_state):
        reward = -graph[state][next_state]
        
        if next_state in self.visited_nodes:
            reward -= self.revisit_penalty + (self.length_penalty * self.age)
        self.age += 1
        return reward
    
# Define the reinforcement learning environment
class Environment:
    def __init__(self, graph):
        self.graph = graph
        self.current_state = None
        self.goal_state = None
        self.previous = None
    
    def reset(self, start_state, goal_state):
        self.current_state = start_state
        self.goal_state = goal_state
        self.previous = None
    def step(self, action):
        # Move to the next state and receive a reward
        self.previous = self.current_state
        next_state = action
        reward = agent.get_reward(self.current_state, next_state)
        self.current_state = next_state
        if next_state == self.goal_state:
            
            reward += 100
            
            agent.age=0
            agent.visited_nodes=set()
            
        if next_state != self.goal_state:
            agent.visited_nodes.add(next_state)
            if agent.age > 100:
                print(self.goal_state)
                print(self.previous,self.current_state)
                return
        
        return next_state, reward
    
# Train the reinforcement learning agent to find the shortest path
learning_rate = 0.6
discount_factor = 0.8
exploration_rate = 0.7
revisit_penalty = 0.5
num_episodes = 10
length_penalty = 1
agent = Agent(learning_rate, discount_factor, exploration_rate, revisit_penalty,length_penalty)
env = Environment(graph)

for episode in range(num_episodes):
    # Reset the environment for a new episode
    start_state = np.random.choice(list(graph.keys()))
    goal_state = np.random.choice(list(graph.keys()))
    
    while goal_state == start_state:
        goal_state = np.random.choice(list(graph.keys()))
    env.reset(start_state, goal_state)
    
    # Run the episode
    total_reward = 0
    while env.current_state != env.goal_state:
        # Choose an action and move to the next state
      

        
        # Update the Q-table
        
        action2= agent.choose_action(env.current_state)
       
        next_state, reward = env.step(action2)
        total_reward += reward
        
        agent.update_q_table(env.previous,action2 , reward, next_state)

    # Print the results of the episode

# Use the trained Q-table to find the shortest path between two nodes
start_node = 'A'
goal_node = 'E'

current_node = start_node
path = [current_node]
while current_node != goal_node:
    for q in graph[current_node].keys():
        qm_values =[agent.q_table.get((q, a),0) for a in graph[q].keys()]
        print("current state: ",current_node)
                
        print("following state: ",q,":",agent.q_table.get((current_node, q),0))
        #get highest from this v then add to the q values of current actions
        print("following keys: ",graph[q].keys())
        print("following qvalues: ",qm_values)
    # Choose the action with the highest Q-value
    q_values = [agent.q_table.get((current_node, a), 0) for a in graph[current_node].keys()]
    print(q_values)
    max_q_value = np.max(q_values)
    
    best_actions = [a for a, q in zip(graph[current_node].keys(), q_values) if q == max_q_value]
    

    if goal_node in [a for a, q in zip(graph[current_node].keys(), q_values)]:
        action = goal_node

    else:
        action = np.random.choice(best_actions)

    # Move to the next node
    current_node = action
    path.append(current_node)
    
# Print the shortest path
print(f"Shortest path between {start_node} and {goal_node}: {' -> '.join(path)}")
print(f"Episode {episode+1} completed with total reward {total_reward}")
print(agent.q_table)

