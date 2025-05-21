import numpy as np
import matplotlib.pyplot as plt

ACTIONS = ['U', 'D', 'L', 'R']
ACTION_VECTORS = {'U': (-1, 0), 'D': (1, 0), 'L': (0, -1), 'R': (0, 1)}

class MazeEnv:
    def __init__(self, n_rows, n_cols, obstacles, start, goal):
        self.n, self.m = n_rows, n_cols
        self.obstacles = set(map(tuple, obstacles))
        self.start = tuple(start)
        self.goal = tuple(goal)

    def step(self, state, action):
        dr, dc = ACTION_VECTORS[action]
        r, c = state
        nr, nc = r + dr, c + dc
        if (nr < 0 or nr >= self.n or nc < 0 or nc >= self.m
            or (nr, nc) in self.obstacles):
            return state, False
        else:
            return (nr, nc), True

class MDPAgent:
    def __init__(self, n_rows, n_cols, start, goal,
                 prior_block=0.3, reward_step=-1, reward_goal=100, gamma=0.95):
        self.n, self.m = n_rows, n_cols
        self.start = start
        self.goal = goal
        self.belief = np.ones((n_rows, n_cols)) * (1 - prior_block)
        self.belief[start] = 1.0
        self.belief[goal] = 1.0
        self.R = np.full((n_rows, n_cols), reward_step, dtype=float)
        self.R[goal] = reward_goal
        self.gamma = gamma

    def transition_model(self):
        N = self.n * self.m
        T = np.zeros((N, len(ACTIONS), N))
        for r in range(self.n):
            for c in range(self.m):
                s = r * self.m + c
                for ai, a in enumerate(ACTIONS):
                    dr, dc = ACTION_VECTORS[a]
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < self.n and 0 <= nc < self.m:
                        s2 = nr * self.m + nc
                        p_free = self.belief[nr, nc]
                        T[s, ai, s2] += p_free
                        T[s, ai, s]  += (1 - p_free)
                    else:
                        T[s, ai, s] += 1.0
        return T

    def policy_iteration(self):
        N = self.n * self.m
        T = self.transition_model()
        R_flat = self.R.flatten()
        policy = np.random.choice(len(ACTIONS), size=N)
        V = np.zeros(N)
        is_goal = np.zeros(N, bool)
        is_goal[self.goal[0]*self.m + self.goal[1]] = True

        while True:
            while True:
                V_new = np.zeros_like(V)
                for s in range(N):
                    a = policy[s]
                    V_new[s] = R_flat[s] + self.gamma * np.dot(T[s, a], V)
                if np.max(np.abs(V_new - V)) < 1e-4:
                    V = V_new
                    break
                V = V_new

            changed = False
            for s in range(N):
                if is_goal[s]:
                    continue
                qsa = [R_flat[s] + self.gamma * np.dot(T[s, ai], V)
                       for ai in range(len(ACTIONS))]
                best_a = int(np.argmax(qsa))
                if best_a != policy[s]:
                    policy[s] = best_a
                    changed = True
            if not changed:
                break

        return policy

    def update_belief(self, state, action, success):
        dr, dc = ACTION_VECTORS[action]
        nr, nc = state[0] + dr, state[1] + dc
        if 0 <= nr < self.n and 0 <= nc < self.m:
            self.belief[nr, nc] = 1.0 if success else 0.0

    def draw_maze(self, env, trajectory):
        plt.clf()

        for i in range(self.n+1):
            plt.plot([0, self.m], [i, i], color='black')
        for j in range(self.m+1):
            plt.plot([j, j], [0, self.n], color='black')


        for (r, c) in env.obstacles:
            plt.fill_between([c, c+1], r, r+1, color='black')


        gr, gc = env.goal
        plt.fill_between([gc, gc+1], gr, gr+1, color='green', alpha=0.5)


        ys = [r + 0.5 for r, c in trajectory]
        xs = [c + 0.5 for r, c in trajectory]
        plt.plot(xs, ys, marker='o')


        ar, ac = trajectory[-1]
        plt.plot(ac + 0.5, ar + 0.5, marker='s', markersize=12, color='red')

        plt.xlim(0, self.m)
        plt.ylim(self.n, 0)
        plt.gca().set_aspect('equal')
        plt.title('Maze Navigation - Step {}'.format(len(trajectory)-1))
        plt.pause(0.5)

    def plan_and_run(self, env, visualize=False):
        trajectory = [env.start]
        state = env.start
        if visualize:
            plt.figure(figsize=(6, 5))
            self.draw_maze(env, trajectory)

        while state != env.goal:
            policy = self.policy_iteration()
            s_idx = state[0]*self.m + state[1]
            action = ACTIONS[policy[s_idx]]
            next_state, success = env.step(state, action)
            self.update_belief(state, action, success)

            if success:
                state = next_state
                trajectory.append(state)
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
