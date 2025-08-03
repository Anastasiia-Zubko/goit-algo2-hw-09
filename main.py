import random
import math
from typing import List, Tuple, Callable, Optional


# Визначення функції Сфери
def sphere_function(x):
  return sum(xi ** 2 for xi in x)


def _clip(val: float, low: float, high: float) -> float:
    return max(low, min(val, high))


def _random_point(bounds: List[Tuple[float, float]]) -> List[float]:
    return [random.uniform(low, high) for low, high in bounds]


# Hill Climbing

def hill_climbing(
    func: Callable[[List[float]], float],
    bounds: List[Tuple[float, float]],
    iterations: int = 1000,
    epsilon: float = 1e-6,
    step_size: float = 0.1,
    seed: Optional[int] = None,
) -> Tuple[List[float], float]:

    if seed is not None:
        random.seed(seed)

    current = _random_point(bounds)
    current_val = func(current)

    for _ in range(iterations):
        candidate = [
            _clip(x + random.uniform(-step_size, step_size), low, high)
            for x, (low, high) in zip(current, bounds)
        ]
        cand_val = func(candidate)

        if cand_val < current_val - epsilon:
            current, current_val = candidate, cand_val
        else:
            step_size *= 0.99

        if step_size < epsilon:
            break

    return current, current_val


# Random Local Search

def random_local_search(
    func: Callable[[List[float]], float],
    bounds: List[Tuple[float, float]],
    iterations: int = 1000,
    epsilon: float = 1e-6,
    radius: float = 0.5,
    seed: Optional[int] = None,
) -> Tuple[List[float], float]:

    if seed is not None:
        random.seed(seed)

    best = _random_point(bounds)
    best_val = func(best)

    for _ in range(iterations):
        candidate = [
            _clip(x + random.uniform(-radius, radius), low, high)
            for x, (low, high) in zip(best, bounds)
        ]
        cand_val = func(candidate)

        if cand_val < best_val - epsilon:
            best, best_val = candidate, cand_val

        radius *= 0.995
        if radius < epsilon:
            break

    return best, best_val


# Simulated Annealing

def simulated_annealing(
    func: Callable[[List[float]], float],
    bounds: List[Tuple[float, float]],
    iterations: int = 1000,
    temp: float = 1000.0,
    cooling_rate: float = 0.95,
    epsilon: float = 1e-6,
    seed: Optional[int] = None,
) -> Tuple[List[float], float]:

    if seed is not None:
        random.seed(seed)

    current = _random_point(bounds)
    current_val = func(current)
    best, best_val = current[:], current_val
    T = temp

    base_step = 0.1 * max(high - low for low, high in bounds)

    for _ in range(iterations):
        candidate = [
            _clip(x + random.uniform(-base_step, base_step), low, high)
            for x, (low, high) in zip(current, bounds)
        ]
        cand_val = func(candidate)
        delta = cand_val - current_val

        if delta < 0 or random.random() < math.exp(-delta / T):
            current, current_val = candidate, cand_val
            if current_val < best_val:
                best, best_val = current[:], current_val

        T *= cooling_rate
        if T < epsilon:
            break

    return best, best_val


if __name__ == "__main__":
  # Межі для функції
  bounds = [(-5, 5), (-5, 5)]

  # Виконання алгоритмів
  print("Hill Climbing:")
  hc_solution, hc_value = hill_climbing(sphere_function, bounds)
  print("Розв'язок:", hc_solution, "Значення:", hc_value)

  print("\nRandom Local Search:")
  rls_solution, rls_value = random_local_search(sphere_function, bounds)
  print("Розв'язок:", rls_solution, "Значення:", rls_value)

  print("\nSimulated Annealing:")
  sa_solution, sa_value = simulated_annealing(sphere_function, bounds)
  print("Розв'язок:", sa_solution, "Значення:", sa_value)