# TravelingSalesman
This program implements a solution to the Traveling Salesman Problem (TSP) using a Genetic Algorithm.

The driver program first opens a text file called input.txt and reads in integer values. The first line specifies the number of cities (N), followed by N lines of (x, y, z) coordinates representing the city locations in three-dimensional space.

The Genetic Algorithm begins by initializing a population of random paths and evaluates them using the total Euclidean distance traveled. It then iteratively evolves the population through parent selection, crossover, and mutation to find an optimal path. Roulette wheel selection is used to create a mating pool, ensuring that shorter paths have a higher probability of being selected for reproduction. The mutation process introduces diversity to prevent premature convergence.

To optimize computation time, the algorithm employs an early stopping mechanism, terminating execution if no improvement occurs for a predefined number of generations. The program outputs the optimal path and its total travel distance to both output.txt and stdout. 

<img width="780" alt="Image" src="https://github.com/user-attachments/assets/a8b6b761-b90a-42d3-b255-640a68a74e38" />
