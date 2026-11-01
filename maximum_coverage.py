"""
Maximum Coverage Facility Location Problem

Problem:
Given n communities and their adjacency relationships (representing nearby areas), 
each community has a certain population. We need to select exactly 2 communities 
to build sales agencies. Each agency can serve its own community and exactly one 
adjacent community. The goal is to maximize the total covered population.

Method:
Modeled as a Maximum Coverage Integer Linear Program (ILP). 
Used boolean variables and adjacency matrices to represent location selection 
and coverage assignments. Solved using CVXPY.
"""

import cvxpy as cp
import numpy as np

def solve_max_coverage(n, adj_matrix, population):
    # has_neighbor[i] = 1 if community i has at least one adjacent community
    has_neighbor = (adj_matrix.sum(axis=1) > 0).astype(int)
    
    # x[i] = 1 if we build an agency in community i
    x = cp.Variable(n, boolean=True)
    
    # y[i] = 1 if community i is covered
    y = cp.Variable(n, boolean=True)
    
    # z[i, j] = 1 if the agency at community i serves community j
    z = cp.Variable((n, n), boolean=True)
    
    # Objective: maximize the total covered population
    objective = cp.Maximize(population @ y)
    
    constraints = [
        # 1. Exactly 2 agencies must be built
        cp.sum(x) == 2,
        
        # 2. If an agency is built at i, it serves exactly 1 neighbor (if it has one)
        cp.sum(z, axis=1) == cp.multiply(has_neighbor, x),
        
        # 3. An agency can only serve itself or an adjacent community (z[i,j] = 0 if not adjacent)
        z <= adj_matrix,
        
        # 4. A community is covered if it has an agency (x=1) or is served by one (sum(z) > 0)
        y <= x + cp.sum(z, axis=0)
    ]
    
    prob = cp.Problem(objective, constraints)
    prob.solve(verbose=False)
    
    return int(prob.value)

if __name__ == "__main__":
    # Test case: 4 communities in a square shape
    # 1-2, 2-3, 3-4, 4-1 are adjacent
    n_communities = 4
    adj = np.array([
        [1, 1, 0, 1],
        [1, 1, 1, 0],
        [0, 1, 1, 1],
        [1, 0, 1, 1]
    ])
    pop = np.array([10, 20, 30, 40])
    
    max_pop = solve_max_coverage(n_communities, adj, pop)
    print(f"Maximum covered population: {max_pop}")
