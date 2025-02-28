# Programmer: Seung Kyu (Kevin) Kim
# Project: Traveling Salesman
import random
import math

def CreateInitialPopulation(size, cities):
    """
    Creates the initial population for the genetic algorithm.

    This function generates a list of random paths (permutations of cities) 
    to serve as the initial population for the genetic algorithm.

    Args:
        size (int): The number of paths (individuals) to generate in the population.
        cities (list): A list of cities/locations in 3D coordinates (x, y, z).

    Returns:
        list: A list of randomly shuffled paths.
    """
    initialPopulation = []
    for _ in range(size):
        # Make a copy for the list of cities.
        path = cities[:]
        # Randomly shuffle cities to create different paths.
        random.shuffle(path)
        initialPopulation.append(path)
    # Return a list of paths (a permutation of cities).
    return initialPopulation

def CreateMatingPool(population, RankList):
    """
    Creates a mating pool using roulette wheel-based selection.

    This function selects individuals from the population for the next generation 
    based on their fitness scores. Individuals with shorter paths (higher fitness) 
    have a higher probability of being selected.

    Args:
        population (list): A list of paths from which the mating pool is to be created.
        RankList (list): A list of tuples where each tuple contains 
                         (index, fitness score) for each path in the population.

    Returns:
        list: A list of selected paths for the mating pool.
    """
    matingPool = []
    # Inversely extract the fitness scores (total distance traveled) for
    # each path from RankList to have higher probabilty for shorter paths.
    fitnessScores = [1 / score for _, score in RankList]  
    totalFitness = sum(fitnessScores)
    # Assign probabilities for each fitness score.
    probabilities = [score / totalFitness for score in fitnessScores]
    
    # Use Roulette Wheel-based Selection.
    for _ in range(len(population)):
        # Randomly select a parent based on probability.
        parent = random.random()
        current = 0

        for i, prob in enumerate(probabilities):
            current += prob
            # Check to see if the cumulative sum exceeds parent.
            if current >= parent:
                # If so, add the selected parent to the mating pool.
                matingPool.append(population[i])
                break
    
    # Return a list of populations selected for mating.
    return matingPool

def Crossover(Parent1, Parent2, Start_Index, End_Index):
    """
    Performs a two-point crossover between two parent paths to create a child path.

    The function selects a segment from Parent1 and inserts it into the child. 
    The remaining positions are filled with the order of cities from Parent2 
    while preserving their relative order.

    Args:
        Parent1 (list): A random sequence of cities for the salesman to follow.
        Parent2 (list): A random sequence of cities for the salesman to follow.
        Start_Index (int): The start index of the subarray to be chosen from Parent1.
        End_Index (int): The start index of the subarray to be chosen from Parent2.

    Returns:
        list: A new child path generated from the crossover of the two parents.
    """
    size = len(Parent1)
    # NOTE: Child path will be a list containing a valid sequence of cities
    #       and it will initially be empty.
    child = [-1] * size
    # Copy the substring from Parent1.
    # NOTE: Increment the End_Index by one to ensure that 
    #       the End_Index is included for Python slicing.
    child[Start_Index:End_Index+1] = Parent1[Start_Index:End_Index+1]
    
    # Ensure that each city is only visited once by 
    # resolving the rest of the sequence from Parent2.
    index = 0
    for i in range(size):
        if child[i] == -1:
            while Parent2[index] in child:
                # NOTE: Each city is only visited once, so skip duplicates.
                index += 1
            child[i] = Parent2[index]
            index += 1 
    
    # Return the child path after performing the crossover.
    return child

def Mutate(path, mutationRate):
    """
    Performs mutation on a given path to modify the order of the locations and
    introduce randomness to the population by using swap and inversion mutations.

    Args:
        path (list): The current path (a sequence of cities).
        mutationRate (float): The probability of mutation occurring.

    Returns:
        list: A new mutated path.
    """
    length = len(path)
    
    for _ in range(length):  
        if random.random() < mutationRate:
            # Swap mutation
            s1, s2 = random.sample(range(length), 2)
            path[s1], path[s2] = path[s2], path[s1]

        elif random.random() < mutationRate / 2:
            # Reversal mutation (subpath reversal)
            s1, s2 = sorted(random.sample(range(length), 2))
            path[s1:s2+1] = reversed(path[s1:s2+1])
    
    return path

def CalculateEuclideanDistance(location1, location2):
    """
    Calculates the Euclidean distance between two locations.

    The Euclidean distance is computed using the formula:
        distance = sqrt((x2 - x1)^2 + (y2 - y1)^2 + (z2 - z1)^2)

    Args:
        location1 (tuple): A tuple representing the (x, y, z) coordinates of the first location.
        location2 (tuple): A tuple representing the (x, y, z) coordinates of the second location.

    Returns:
        float: The Euclidean distance between the two locations.
    """
    # Calculate the distance between two locations using the Euclidean formula.
    distance = math.sqrt((location2[0] - location1[0]) ** 2 +
                         (location2[1] - location1[1]) ** 2 +
                         (location2[2] - location1[2]) ** 2)

    return distance

def CalculateTotalDistance(path):
    """
    Calculates the total Euclidean distance traveled for a given path.

    The total distance is computed by summing the Euclidean distances between 
    consecutive locations in the path, including the return to the starting point 
    to satisfy the Traveling Salesman Problem (TSP) constraint.

    Args:
        path (list): A list of tuples representing the (x, y, z) coordinates of 
                     locations in the order they are visited.

    Returns:
        float: The total distance traveled for the given path.
    """
    n = len(path)
    totalDistance = 0.0

    # Compute the total distance traveled for a given path.
    for i in range(n):
        location1 = path[i]
        # NOTE: We will return to the starting city at the end
        #       to maintain the constraint for TSP.
        location2 = path[(i + 1) % n]

        d = CalculateEuclideanDistance(location1, location2)
        totalDistance += d

    return totalDistance

def GeneticAlgorithm(N, locations):
    """
    Performs the Genetic Algorithm to solve the Traveling Salesman Problem (TSP).

    This function initializes a population of potential solutions (paths), evaluates 
    them based on total travel distance, and iteratively evolves the population using 
    parent selection, crossover, and mutation to find an optimal solution.

    Args:
        N (int): The number of cities in the TSP problem.
        locations (list): A list of tuples representing the (x, y, z) coordinates of cities.

    Returns:
        tuple: A tuple containing:
            - float: The total distance of the resulting path.
            - list: The optimal path as a list of city indices, including the return 
                    to the starting city.
    """
    totalDistance = float('inf')
    prevBest = float('inf')
    optimalPath = None
    populationSize = 100
    numGenerations = 500
    mutationRate = 0.1
    stagnantGenerations = 0
    stopLimit = 50

    # Create initial population, which will be a list of paths 
    # (a permutation of cities).
    population = CreateInitialPopulation(populationSize, locations)

    for _ in range(numGenerations):
        # Rank each path based on total distance.
        rank = [(i, CalculateTotalDistance(path)) for i, path in enumerate(population)]
        # Create a list to rank the all the paths in the initial population
        # in descending order based on their fitness score.
        # NOTE: A high fitness score will be given to the shortest path,
        #       so the rank list will be sorted in descending order 
        #       in terms of the fitness score.
        rank.sort(key=lambda x: x[1])

        # Apply early stopping mechanism if no improvement is observed.
        if rank[0][1] < prevBest:
            prevBest = rank[0][1]
            # Reset counter for stagnant generations if evolution occurs.
            stagnantGenerations = 0
        else: 
            stagnantGenerations += 1

        if stagnantGenerations >= stopLimit:
            # Break out of for loop if no improvement is made.
            break

        # Keep track of the most optimal path.
        if rank[0][1] < totalDistance:
            optimalPath = population[rank[0][0]]
            totalDistance = rank[0][1]

        # Get a list of selected paths for mating.
        matingPool = CreateMatingPool(population, rank)

        # Generate next generation before performing crossover & mutation.
        newGeneration = []

        # Optimize the genetic algorithm by performing elitism and keeping 
        # the top 10% best individuals to maintain good solutions.
        # NOTE: Have at least two elite individuals survive for the next generation.
        elites = max(2, populationSize // 10)
        newGeneration.extend([population[i] for i, _ in rank[:elites]])

        # Perform crossover and mutation on the remaining population.
        while len(newGeneration) < populationSize:
            parent1, parent2 = random.sample(matingPool, 2)

            # Perform crossover on random points.
            start, end = sorted(random.sample(range(len(parent1)), 2))
            child = Crossover(parent1, parent2, start, end)

            # Perform mutation using controlled probability.
            if random.random() < mutationRate:
                child = Mutate(child, mutationRate)

            newGeneration.append(child)

        # Update population
        population = newGeneration

    # Append the starting city to the end of the path to maintain the TSP constraint.
    optimalPath.append(optimalPath[0]) 
    return totalDistance, optimalPath

# This is the driver program for Homework 1.
def main() :
    try:
        # Open input.txt for reading.
        with open('input.txt', 'r') as infile:
            # Read the first line to get the number of city locations (N) in the 3D space.
            try:
                nLine = infile.readline().strip()
                N = int(nLine)
                if N < 0:
                    raise ValueError("N must be a non-negative integer...")
            except ValueError as e:
                raise ValueError(f"Invalid value for N: {nLine}. N must be a non-negative integer...") from e

            # Initialize a list to store the city coordinates.
            locations = []

            # Read the next N lines to get the city coordinates.
            for i in range(N):
                # Read a line while stripping any leading/trailing whitespace.
                line = infile.readline().strip()
                if not line:
                    raise ValueError(f"Expected {N} lines of city coordinates, but found only {i} lines...")

                # Split the line into x, y, and z coordinates.
                coordinates = line.split()
                if len(coordinates) != 3:
                    raise ValueError(f"Invalid format in line {i + 2}: '{line}'...")

                # Map the x, y, and z coordinate of the current location.
                x, y, z = map(int, coordinates)

                # Append the coordinates as a tuple to the list of locations.
                locations.append((x, y, z))

            # Check if there are extra lines in the file
            extraLine = infile.readline()
            if extraLine.strip():
                raise ValueError(f"Extra data found in the file after reading {N} cities.")

        # Calculate the total distance traveled and find the optimal path
        # using the Genetic Algorithm.
        totalDistance, optimalPath = GeneticAlgorithm(N, locations) 

        # Open the output file for writing.
        with open('output.txt', 'w') as outfile:
            # Display the total distance of the path in the first line.
            outfile.write(f"{totalDistance:.3f}\n")
            print(f"{totalDistance:.3f}")

            # For next N+1 lines, display all the cities visited in order separated by one whitespace.
            # NOTE: coordinate[0] = x, coordinate[1] = y, and coordinate[2] = z for a city. 
            for coordinate in optimalPath:
                outfile.write(f"{coordinate[0]} {coordinate[1]} {coordinate[2]}\n")
                print(f"{coordinate[0]} {coordinate[1]} {coordinate[2]}")
            

    except FileNotFoundError:
        print("ERROR: \"input.txt\" file not found in the current directory...")
    except ValueError as e:
        print(f"ERROR: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    
if __name__ == "__main__":
    main()