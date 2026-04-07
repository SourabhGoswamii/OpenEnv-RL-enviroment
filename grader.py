# grader.py


def compute_score(total_reward, max_possible_reward):
    """
    Normalize score between 0 and 1
    """
    if max_possible_reward <= 0:
        return 0.001

    score = total_reward / max_possible_reward
    return max(0.001, min(0.999, score))


def evaluate_episode(rewards, max_per_step=15):
    total_reward = sum(rewards)
    max_possible = len(rewards) * max_per_step
    return compute_score(total_reward, max_possible)


def evaluate_task1(rewards):
    return evaluate_episode(rewards, 15)


def evaluate_task2(rewards):
    return evaluate_episode(rewards, 15)


def evaluate_task3(rewards):
    return evaluate_episode(rewards, 15)
