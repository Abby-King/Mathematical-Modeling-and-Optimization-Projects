# Mathematical Modeling and Optimization Projects

## Overview

This repository contains selected Python implementations from my mathematical modeling coursework.

These projects demonstrate my ability to formulate complex discrete optimization problems (e.g., graph coloring, scheduling, and resource allocation) into mathematical programming models.

## Project Highlights

### 1\. Multi-Stage Interview Scheduling (`interview_scheduling.py`)

* **Problem**: Scheduling multi-stage interviews for students where the sequence must remain consistent across all stages. The goal is to minimize the makespan (the completion time of the last student).
* **Method**: Formulated as a Mixed-Integer Linear Program (MILP). Used binary variables to determine the optimal student sequence and formulated completion time constraints to prevent overlaps.

### 2\. Exam Scheduling Optimization (`exam_scheduling.py`)

* **Problem**: Assigning courses to exam time slots to prevent student conflicts, aiming to minimize the total number of slots.
* **Method**: Formulated as an Integer Linear Program (ILP). Used binary variables to assign courses and ensured conflicting courses do not overlap. Also enforced the sequential use of time slots to eliminate redundant solutions and accelerate the solver.

### 3\. Maximum Coverage Facility Location (`maximum_coverage.py`)

* **Problem**: A facility location problem selecting two locations to maximize the total covered population under specific adjacency constraints.
* **Method**: Modeled as a maximum coverage Integer Linear Program (ILP) using boolean variables to represent location selection and coverage.

### 4\. Constrained Unbounded Knapsack (`constrained_knapsack.py`)

* **Problem**: Unbounded knapsack problem with complex logical constraints (Dependencies: selecting A requires B; Conflicts: C and D cannot be selected together).
* **Method**: Used integer variables for item counts and boolean indicator variables to linearize the logical dependencies and mutual exclusions.

## Tech Stack

* **Language**: Python
* **Optimization Libraries**: CVXPY
* **Data Processing**: NumPy
* **Methodologies**: Integer Linear Programming (ILP), Graph Theory, Combinatorial Optimization

