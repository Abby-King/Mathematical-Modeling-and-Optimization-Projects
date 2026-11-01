"""
Constrained Knapsack Problem

Problem:
Given n item types and a backpack of capacity W. Each item type i has a weight
w_i and value v_i, and can be selected unlimited times. 
Two types of logical constraints are introduced:
1. Dependency: If item u is selected, item k must be selected.
2. Conflict: Item a and item b cannot be selected at the same time.
The goal is to maximize the total value without exceeding capacity.

Method:
Formulated as a Mixed-Integer Linear Program (MILP). 
Used integer variables for item counts, boolean variables for selection status, 
and M constraints to encode the logical dependencies and conflicts.
"""

import cvxpy as cp
import numpy as np

def solve_constrained_knapsack(n, W, weights, values, deps, conflicts):
    # Maximum possible count for each item based on weight alone
    M = np.floor(W / weights).astype(int)
    
    # x[i] = number of items of type i
    x = cp.Variable(n, integer=True)
    
    # y[i] = 1 if item i is selected (x[i] >= 1)
    y = cp.Variable(n, boolean=True)
    
    # Objective: maximize total value
    objective = cp.Maximize(values @ x)
    
    constraints = [
        # 1. Capacity constraint
        weights @ x <= W,
        
        # 2. Non-negativity and upper bound
        x >= 0,
        x <= M,
        
        # 3. M link between x and y: 
        # x[i] <= M[i] * y[i] means y[i]=0 => x[i]=0
        # x[i] >= y[i] means y[i]=1 => x[i]>=1
        x <= cp.multiply(M, y),
        x >= y
    ]
    
    # 4. Dependency constraints: if u is selected, k must be selected
    for u, k in deps:
        constraints.append(y[u - 1] <= y[k - 1])
        
    # 5. Conflict constraints: a and b cannot both be selected
    for a, b in conflicts:
        constraints.append(y[a - 1] + y[b - 1] <= 1)
        
    prob = cp.Problem(objective, constraints)
    prob.solve(verbose=False)
    
    # Return both the objective value and the optimal item counts
    return int(prob.value), [int(val) for val in x.value]

if __name__ == "__main__":
    # Test case: 4 items, capacity W = 10
    # Item 1: weight 2, value 3;  Item 2: weight 3, value 4
    # Item 3: weight 4, value 5;  Item 4: weight 5, value 8
    n_items = 4
    W = 10
    weights = np.array([2, 3, 4, 5])
    values = np.array([3, 4, 5, 8])
    
    # Dependency: If item 2 is selected, item 1 must be selected
    deps = [(2, 1)]
    
    # Conflict: Item 3 and item 4 cannot be selected together
    conflicts = [(3, 4)]
    
    max_value, optimal_counts = solve_constrained_knapsack(n_items, W, weights, values, deps, conflicts)
    print(f"Maximum value: {max_value}")
    print(f"Optimal item counts: {optimal_counts}")
