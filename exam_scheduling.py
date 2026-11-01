"""
Exam Scheduling Optimization (Graph Coloring)

Problem:
Given n courses and their conflict relationships (students taking both courses), 
assign courses to time slots such that no conflicting courses share the same slot. 
The goal is to minimize the total number of required time slots.

Method:
Modeled as a Graph Coloring problem. Courses are vertices, conflicts are edges. 
Formulated as an Integer Linear Program (ILP) and solved using CVXPY.
"""

import cvxpy as cp

def solve_exam_scheduling(n, edges):
    # Worst case: every course needs its own slot
    K = n 
    
    # x[i, k] = 1 if course i is assigned to time slot k
    x = cp.Variable((n, K), boolean=True)
    
    # y[k] = 1 if time slot k is actually used
    y = cp.Variable(K, boolean=True)
    
    # Objective: minimize the total number of used time slots
    objective = cp.Minimize(cp.sum(y))
    
    constraints = []
    
    # 1. Each course must be scheduled in exactly one time slot
    constraints.append(cp.sum(x, axis=1) == 1)
    
    # 2. Conflicting courses cannot be scheduled in the same time slot
    for u, v in edges:
        for k in range(K):
            constraints.append(x[u-1, k] + x[v-1, k] <= 1)
            
    # 3. Link x and y: if any course is scheduled at slot k, y[k] must be 1
    for i in range(n):
        for k in range(K):
            constraints.append(x[i, k] <= y[k])
            
    # 4. Ensure slots are used in sequential order
    for k in range(K - 1):
        constraints.append(y[k] >= y[k + 1])
        
    prob = cp.Problem(objective, constraints)
    prob.solve(verbose=False)
    
    return int(prob.value)

if __name__ == "__main__":
    n_courses = 5
    conflict_edges = [(1, 2), (2, 3), (3, 4), (4, 5), (5, 1)]
    
    min_slots = solve_exam_scheduling(n_courses, conflict_edges)
    print(f"Minimum time slots required: {min_slots}")
