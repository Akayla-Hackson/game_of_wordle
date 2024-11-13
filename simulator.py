import random
import pandas as pd

#####################################################################################################################################################################
# This class creates a simulation of the game of Wordle. 
# note: some code is commented out becasue the valid_guesses.csv and valid_solutions.csv do not have any overlap in words, 
#       so guesses are being made based off the valid_solutions.csv.
#
# Max # of guesses may be set to anything at the moment.
####################################################################################################################################################################
class WordleSimulator:
    def __init__(self, guesses_file, solutions_file):
        #### load words from CSV files ####
        # self.initial_valid_guesses = pd.read_csv(guesses_file, header=None).iloc[:,0].tolist()

        self.valid_solutions = pd.read_csv(solutions_file, header=None).iloc[:,0].tolist()
        self.max_guesses = 1000
        self.reset_game() 

    def reset_game(self):
        #### reset game to initial conditions ####
        self.secret_word = random.choice(self.valid_solutions)
        self.valid_guesses = self.valid_solutions.copy()
        # self.valid_guesses = self.initial_valid_guesses.copy()

    def guess_word(self, guess):

        #### invalid guess ####
        if len(guess) != len(self.secret_word) or guess not in self.valid_guesses:
            return None 
        
        # remove the guessed word from available guesses
        if guess in self.valid_guesses:
            self.valid_guesses.remove(guess)

        feedback = ['Gray'] * len(guess)
        secret_word_used = [False] * len(self.secret_word)
        guess_used = [False] * len(guess)

        # 1st pass for correct positions
        for i in range(len(guess)):
            if guess[i] == self.secret_word[i]:
                feedback[i] = 'Green'
                secret_word_used[i] = True
                guess_used[i] = True

        # 2nd pass for correct letters but wrong positions
        for i in range(len(guess)):
            if not guess_used[i]:
                for j in range(len(self.secret_word)):
                    if not secret_word_used[j] and guess[i] == self.secret_word[j]:
                        feedback[i] = 'Yellow'
                        secret_word_used[j] = True
                        guess_used[i] = True
                        break
        return feedback

def random_guessing_game(simulator):
    for attempt in range(simulator.max_guesses):
        if not simulator.valid_guesses:
            print("No more valid guesses available.")
            return False
        guess = random.choice(simulator.valid_guesses)
        feedback = simulator.guess_word(guess)
        print(f"Attempt {attempt + 1}: Guess = {guess}, Secret = {simulator.secret_word} Feedback = {feedback}")
        if guess == simulator.secret_word:
            print("The secret word was guessed correctly!")
            return True
    print("Failed to guess the secret word within the allowed attempts.")
    return False

guesses_file = './kaggle_data/valid_guesses.csv'
solutions_file = './kaggle_data/valid_solutions.csv'
game = WordleSimulator(guesses_file, solutions_file)

##### to play a game (based off of random guessing), use code below #####
random_guessing_game(game)


# reset game
game.reset_game()
