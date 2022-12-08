class AgentConfig:
    def __init__(self, epsilon=1.0, epsilon_decay=None, min_epsilon=0.01, alpha=0.1, gamma=0.9, lambdaa=0.9):
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon

        self.alpha = alpha
        self.gamma = gamma
        self.lambdaa = lambdaa
