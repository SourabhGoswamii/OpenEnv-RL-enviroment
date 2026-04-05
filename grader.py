# grader.py

def compute_score(total_reward, max_possible_reward):
    """
    Normalize score between 0 and 1
    """
    if max_possible_reward <= 0:
        return 0.0

    score = total_reward / max_possible_reward
    return max(0.0, min(1.0, score))

def evaluate_episode(rewards, max_per_step=15):
    total_reward = sum(rewards)
    max_possible = len(rewards) * max_per_step
    return compute_score(total_reward, max_possible)