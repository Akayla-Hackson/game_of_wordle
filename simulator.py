import random
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

#####################################################################################################################################################################
# This class creates a simulation of the game of Wordle. 
# note: some code is commented out becasue the valid_guesses.csv and valid_solutions.csv do not have any overlap in words, 
#       so guesses are being made based off the valid_solutions.csv.
#
# guess_word() takes in a guess and returns the feedback (Green, yellow, or grey) corresponding to the placement of each letter
#
####### GAME STRATEGIES #########
# random_guessing() randomly guesses from the valid solutions dataset until it gets the word or it hits the max guesses limit. 
#                   This func doesn't take the feeback into account
#
#
#
#
####################################################################################################################################################################
class WordleSimulator:
    def __init__(self, solutions_file, max_guesses=6, num_games=1000, strategy='random'):
        #### load words from CSV files ####
        self.valid_solutions = pd.read_csv(solutions_file, header=None).iloc[:,0].tolist()
        self.max_guesses = int(max_guesses)
        self.num_games = int(num_games)
        self.strategy = strategy
        self.reset_game()

    def reset_game(self):
        #### reset game to initial conditions ####
        self.secret_word = random.choice(self.valid_solutions)
        self.valid_guesses = self.valid_solutions.copy()

    def play_game(self):
        if self.strategy == 'random':
            return self.random_guessing()
        
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

    def random_guessing(self):
        for attempt in range(1, self.max_guesses + 1):
            if not self.valid_guesses:
                print("No more valid guesses available.")
                return False, attempt
            guess = random.choice(self.valid_guesses)
            # feedback = self.guess_word(guess)
            # print(f"Attempt {attempt}: Guess = {guess}, Secret = {self.secret_word}, Feedback = {feedback}")
            if guess == self.secret_word:
                print("The secret word was guessed correctly!")
                return True, attempt
        print("Failed to guess the secret word within the allowed attempts.")
        return False, self.max_guesses


    def simulate_games(self):
        successes = 0
        attempts_distribution = []
        for _ in range(self.num_games):
            self.reset_game()
            success, attempts = self.play_game()
            if success:
                successes += 1
            attempts_distribution.append(attempts)
        success_rate = successes / self.num_games
        average_attempts = sum(attempts_distribution) / self.num_games
        print(f"Number of Successes: {successes}")
        print(f"Success Rate: {success_rate*100:.4f}%")
        print(f"Average Attempts Needed: {average_attempts:.4f}")
        self.plot_results(attempts_distribution)


    def plot_results(self, attempts):
        directory = f'plots_of_runs/{self.strategy}'
        os.makedirs(directory, exist_ok=True)
        plt.figure(figsize=(10, 5))
        bins = np.arange(1, self.max_guesses + 2) - 0.5  # shift bins to the left by 0.5
        counts, bins, patches = plt.hist(attempts, bins=bins, alpha=0.7, color='blue', edgecolor='black')
        for patch, label in zip(patches, counts):
            if label > 0:  # only add labels to non-zero bins
                plt.text(patch.get_x() + patch.get_width() / 2, patch.get_height(), f'{int(label)}', 
                        ha='center', va='bottom')

        plt.title('Distribution of Attempts to Guess the Secret Word')
        plt.xlabel('Number of Attempts')
        plt.ylabel('Frequency')
        plt.xticks(np.arange(1, self.max_guesses + 1))
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.savefig(f'plots_of_runs/{self.strategy}/{self.max_guesses}guesses_{self.num_games}_games.png')




guesses_file = './kaggle_data/valid_guesses.csv'
solutions_file = './kaggle_data/valid_solutions.csv'
game = WordleSimulator(solutions_file)

game.simulate_games()

