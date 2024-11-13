import pandas as pd

def compare_csv(guesses, solutions):
    # load data 
    df_guesses = pd.read_csv(guesses, header=None, names=['Words'])
    df_solutions = pd.read_csv(solutions, header=None, names=['Words'])

    # convert to sets 
    set_guesses = set(df_guesses['Words'].str.strip().str.lower())
    set_solutions = set(df_solutions['Words'].str.strip().str.lower())

    # calc intersections and differences
    intersection = set_guesses & set_solutions
    unique_to_guesses = set_guesses - set_solutions
    unique_to_solutions = set_solutions - set_guesses

    # print the counts
    print(f"Total words in valid guesses: {len(set_guesses)}")
    print(f"Total words in valid solutions: {len(set_solutions)}")
    print(f"Words in both files: {len(intersection)}")
    print(f"Words unique to valid guesses: {len(unique_to_guesses)}")
    print(f"Words unique to valid solutions: {len(unique_to_solutions)}")



guesses = './kaggle_data/valid_guesses.csv'
solutions = './kaggle_data/valid_solutions.csv'
compare_csv(guesses, solutions)