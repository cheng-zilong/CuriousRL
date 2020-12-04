from __future__ import annotations
from CuriousRL.scenario.scen_wrapper import ScenarioWrapper

class BasicDQN():
    def __init__(self):
        pass

    def init(self, scenario:ScenarioWrapper):
        # initialize OpenAI Gym env and dqn agent
        env = gym.make(ENV_NAME)

    def solve(self):
        agent = DQNWrapper(env)
        for episode in xrange(EPISODE):
            # initialize task
            state = env.reset()
            # Train
            for step in xrange(STEP):
                action = agent.egreedy_action(state) # e-greedy action for train
                next_state,reward,done,_ = env.step(action)
                # Define reward for agent
                reward_agent = -1 if done else 0.1
                agent.perceive(state,action,reward,next_state,done)
                state = next_state
                if done:
                    break