import numpy as np
import matplotlib.pyplot as plt

ACTIONS = ['R', 'D', 'L', 'U']
ACTION_VECTORS = {'U': (-1, 0), 'D': (1, 0), 'L': (0, -1), 'R': (0, 1)}

class MazeEnv:
    def __init__(self, n_rows, n_cols, obstacles, start, goal):
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.obstacles = set(map(tuple, obstacles))
        self.start = tuple(start)
        self.goal = tuple(goal)

    def step(self, current_position, action):
        delta_row, delta_col = ACTION_VECTORS[action]
        row, col = current_position
        next_row, next_col = row + delta_row, col + delta_col

        if (next_row < 0 or next_row >= self.n_rows or
            next_col < 0 or next_col >= self.n_cols or
            (next_row, next_col) in self.obstacles):
            return current_position, False
        else:
            return (next_row, next_col), True

class MDPAgent:
    def __init__(self, n_rows, n_cols, start, goal,
                 prior_block=0.3, reward_step=-1, reward_goal=100, gamma=0.95):
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.start = start
        self.goal = goal

        self.belief = np.ones((n_rows, n_cols)) * (1 - prior_block)
        self.belief[start] = 1.0
        self.belief[goal] = 1.0

        self.rewards = np.full((n_rows, n_cols), reward_step, dtype=float)
        self.rewards[goal] = reward_goal

        self.gamma = gamma

    def build_transition_model(self):
        num_states = self.n_rows * self.n_cols
        num_actions = len(ACTIONS)
        transition_prob = np.zeros((num_states, num_actions, num_states))

        for row in range(self.n_rows):
            for col in range(self.n_cols):
                state_index = row * self.n_cols + col

                for action_index, action in enumerate(ACTIONS):
                    delta_row, delta_col = ACTION_VECTORS[action]
                    next_row, next_col = row + delta_row, col + delta_col

                    if 0 <= next_row < self.n_rows and 0 <= next_col < self.n_cols:
                        next_state = next_row * self.n_cols + next_col
                        prob_free = self.belief[next_row, next_col]

                        transition_prob[state_index, action_index, next_state] += prob_free
                        transition_prob[state_index, action_index, state_index] += (1 - prob_free)
                    else:
                        transition_prob[state_index, action_index, state_index] += 1.0

        return transition_prob

    def policy_iteration(self):
        num_states = self.n_rows * self.n_cols
        num_actions = len(ACTIONS)

        transition_prob = self.build_transition_model()
        rewards_flat = self.rewards.flatten()

        policy = np.random.choice(num_actions, size=num_states)
        value_function = np.zeros(num_states)

        goal_index = self.goal[0] * self.n_cols + self.goal[1]
        is_goal_state = np.zeros(num_states, dtype=bool)
        is_goal_state[goal_index] = True

        while True:
            while True:
                new_values = np.zeros_like(value_function)
                for state in range(num_states):
                    action_index = policy[state]
                    new_values[state] = (
                        rewards_flat[state]
                        + self.gamma * np.dot(transition_prob[state, action_index], value_function)
                    )
                if np.max(np.abs(new_values - value_function)) < 1e-4:
                    value_function = new_values
                    break
                value_function = new_values

            policy_stable = True
            for state in range(num_states):
                if is_goal_state[state]:
                    continue

                q_values = [
                    rewards_flat[state] + self.gamma * np.dot(transition_prob[state, a], value_function)
                    for a in range(num_actions)
                ]
                best_action = int(np.argmax(q_values))

                if best_action != policy[state]:
                    policy[state] = best_action
                    policy_stable = False

            if policy_stable:
                break

        return policy

    def update_belief(self, current_position, action, move_successful):
        delta_row, delta_col = ACTION_VECTORS[action]
        row, col = current_position
        next_row, next_col = row + delta_row, col + delta_col

        if 0 <= next_row < self.n_rows and 0 <= next_col < self.n_cols:
            self.belief[next_row, next_col] = 1.0 if move_successful else 0.0

    def draw_maze(self, env, trajectory):
        plt.clf()

        for i in range(self.n_rows + 1):
            plt.plot([0, self.n_cols], [i, i], color='black')
        for j in range(self.n_cols + 1):
            plt.plot([j, j], [0, self.n_rows], color='black')

        for (r, c) in env.obstacles:
            plt.fill_between([c, c+1], r, r+1, color='black')

        gr, gc = env.goal
        plt.fill_between([gc, gc+1], gr, gr+1, color='green', alpha=0.5)

        ys = [r + 0.5 for r, c in trajectory]
        xs = [c + 0.5 for r, c in trajectory]
        plt.plot(xs, ys, marker='o')

        ar, ac = trajectory[-1]
        plt.plot(ac + 0.5, ar + 0.5, marker='s', markersize=12, color='red')

        plt.xlim(0, self.n_cols)
        plt.ylim(self.n_rows, 0)
        plt.gca().set_aspect('equal')
        plt.title(f'Maze Navigation - Step {len(trajectory)-1}')
        plt.pause(0.5)

    def plan_and_run(self, env, visualize=False):
        trajectory = [env.start]
        current_position = env.start

        if visualize:
            plt.figure(figsize=(6, 5))
            self.draw_maze(env, trajectory)

        while current_position != env.goal:
            policy = self.policy_iteration()
            state_index = current_position[0] * self.n_cols + current_position[1]
            action = ACTIONS[policy[state_index]]

            next_position, move_successful = env.step(current_position, action)
            self.update_belief(current_position, action, move_successful)

            if move_successful:
                current_position = next_position
                trajectory.append(current_position)
                if visualize:
                    self.draw_maze(env, trajectory)

        if visualize:
            plt.show()
        return trajectory

if __name__ == '__main__':
    rows, cols = 5, 6
    obstacles = [(0,1), (2,1), (3,1), (2,3), (3,4), (4,4)]
    start = (0, 0)
    goal  = (4, 5)

    env = MazeEnv(rows, cols, obstacles, start, goal)
    agent = MDPAgent(rows, cols, start, goal, prior_block=0.3)

    path = agent.plan_and_run(env, visualize=True)
    print("Final path:", path)
