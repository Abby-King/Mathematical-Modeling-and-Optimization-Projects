"""
Multi-Stage Interview Scheduling

Given m students attending n stages of interviews, the order of students 
must remain consistent across all stages. The goal is to minimize the 
total makespan (completion time of the last student in the last stage).

Formulated as a Mixed-Integer Linear Program (MILP) and solved using CVXPY.
"""

import cvxpy as cp
import numpy as np

def solve_scheduling(m, n, t):
    # x[k, i] = 1 if student i is assigned to position k
    x = cp.Variable((m, m), boolean=True)
    
    # C[k, j] = completion time of student at position k, stage j
    C = cp.Variable((m, n))
    
    # Objective: minimize the completion time of the last student at the last stage
    obj = cp.Minimize(C[m - 1, n - 1])
    
    constraints = [
        # Each student is assigned to exactly one position (row sum = 1)
        cp.sum(x, axis=1) == 1,
        # Each position is assigned to exactly one student (col sum = 1)
        cp.sum(x, axis=0) == 1
    ]
    
    for k in range(m):
        for j in range(n):
            # Processing time of the student assigned to position k at stage j
            pt = cp.sum(x[k, :] * t[:, j])
            
            if k == 0 and j == 0:
                # First student, first stage
                constraints.append(C[0, 0] == pt)
            elif k == 0:
                # First student, subsequent stages
                constraints.append(C[0, j] == C[0, j - 1] + pt)
            elif j == 0:
                # Subsequent students, first stage
                constraints.append(C[k, 0] == C[k - 1, 0] + pt)
            else:
                # Must wait for both previous stage and previous student
                constraints.append(C[k, j] >= C[k - 1, j] + pt)
                constraints.append(C[k, j] >= C[k, j - 1] + pt)
    
    prob = cp.Problem(obj, constraints)
    prob.solve(verbose=False)
    return int(C[m - 1, n - 1].value)

if __name__ == "__main__":
    m = 3 # 3 students
    n = 2 # 2 stages
    t = np.array([
        [10, 20],
        [15, 10],
        [20, 15]
    ])
    result = solve_scheduling(m, n, t)
    print(f"Minimum makespan: {result} minutes")
