from collections import defaultdict
import random
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
from tqdm import tqdm
from colorama import Fore, Style, init
from collections import Counter
import math
import copy

init(autoreset=True)  # Auto-reset colorama colors

class WordleSimulator:
    def __init__(self, solutions_file, max_guesses, num_games, strategy):
        self.valid_solutions = pd.read_csv(solutions_file, header=None).iloc[:,0].tolist()
        self.valid_guesses = self.valid_solutions
        self.max_guesses = int(max_guesses)
        self.num_games = int(num_games)
        self.strategy = strategy
        self.reset_game()

    def reset_game(self):
        self.secret_word = random.choice(self.valid_solutions)
        print(f"The secret word is: {self.secret_word}")
        self.valid_guesses = self.valid_solutions.copy()

    def play_game(self):
        if self.strategy == 'random':
            return self.random_guessing()
        elif self.strategy == 'baseline':
            return self.baseline_guessing()
        elif self.strategy == 'entropy':
            return self.baseline_entropy()
        elif self.strategy == 'mcts':
            return self.mcts_guessing()
        else:
            raise ValueError(f"Invalid strategy: {self.strategy}")

    def guess_word(self, guess):
        if len(guess) != len(self.secret_word) or guess not in self.valid_guesses:
            print(f"Invalid guess: {guess}")
            return None 

        if guess in self.valid_guesses:
            self.valid_guesses.remove(guess)

        feedback = ['Gray'] * len(guess)
        secret_word_used = [False] * len(self.secret_word)
        guess_used = [False] * len(guess)

        for i in range(len(guess)):
            if guess[i] == self.secret_word[i]:
                feedback[i] = 'Green'
                secret_word_used[i] = True
                guess_used[i] = True

        for i in range(len(guess)):
            if not guess_used[i]:
                for j in range(len(self.secret_word)):
                    if not secret_word_used[j] and guess[i] == self.secret_word[j]:
                        feedback[i] = 'Yellow'
                        secret_word_used[j] = True
                        guess_used[i] = True
                        break
        return feedback

    def display_guess_feedback(self, guess, feedback):
        visual_feedback = ""
        for i, letter in enumerate(guess):
            if feedback[i] == 'Green':
                visual_feedback += f"{Fore.GREEN}{letter}{Style.RESET_ALL} "
            elif feedback[i] == 'Yellow':
                visual_feedback += f"{Fore.YELLOW}{letter}{Style.RESET_ALL} "
            else:
                visual_feedback += f"{Fore.LIGHTBLACK_EX}{letter}{Style.RESET_ALL} "
        print(visual_feedback)

    def random_guessing(self):
        remaining_attempts = self.max_guesses
        for attempt in range(1, remaining_attempts + 1):
            if not self.valid_guesses:
                print("No more valid guesses available.")
                return False, attempt
            
            guess = random.choice(self.valid_guesses)
            feedback = self.guess_word(guess)
            self.display_guess_feedback(guess, feedback)

            if guess == self.secret_word:
                print("The secret word was guessed correctly!")
                return True, attempt
        print("Failed to guess the secret word within the allowed attempts.")
        return False, remaining_attempts

    def baseline_guessing(self):
        remaining_attempts = self.max_guesses  # Initialize a local counter for guesses
        possible_words = self.valid_guesses.copy()
        
        for attempt in range(1, remaining_attempts + 1):
            if not possible_words:
                print("No more valid guesses available.")
                return False, attempt

            guess = random.choice(possible_words)
            possible_words.remove(guess)
            feedback = self.guess_word(guess)
            self.display_guess_feedback(guess, feedback)

            if guess == self.secret_word:
                print("The secret word was guessed correctly!")
                return True, attempt

            # Identify letters by their feedback type
            gray_letters = {guess[i] for i in range(len(feedback)) if feedback[i] == 'Gray'}
            yellow_positions = {i: guess[i] for i in range(len(feedback)) if feedback[i] == 'Yellow'}
            yellow_letters = {guess[i] for i in range(len(feedback)) if feedback[i] == 'Yellow'}
            green_positions = {i: guess[i] for i in range(len(feedback)) if feedback[i] == 'Green'}
            green_letters = {guess[i] for i in range(len(feedback)) if feedback[i] == 'Green'}

            def valid_word(word):
                if any(letter in word for letter in gray_letters):
                    if not (any(letter in green_letters for letter in gray_letters) or any(letter in yellow_letters for letter in gray_letters)):
                        return False
                if any(word[i] == letter for i, letter in yellow_positions.items()):
                    return False
                if not yellow_letters.issubset(set(word)):
                    return False
                if any(word[i] != letter for i, letter in green_positions.items()):
                    return False
                return True
            
            possible_words = [word for word in possible_words if valid_word(word)]
        print("Failed to guess the secret word within the allowed attempts.")
        return False, remaining_attempts

     #--- Entropy Method ---#
    
    def generate_feedback_entropy(self, guess,solution):
        """
        Generate feedback for a guess compared to a solution.
        'G' = Green, 'Y' = Yellow, 'B' = Gray. This function is developed for the entropy calculation only.
        """
        feedback = ['B'] * len(guess)
        solution_used = [False] * len(solution)
        guess_used = [False] * len(guess)

        # First pass for "Green" matches
        for i in range(len(guess)):
            if guess[i] == solution[i]:
                feedback[i] = 'G'
                solution_used[i] = True
                guess_used[i] = True

        # Second pass for "Yellow" matches
        for i in range(len(guess)):
            if not guess_used[i]:
                for j in range(len(solution)):
                    if not solution_used[j] and guess[i] == solution[j]:
                        feedback[i] = 'Y'
                        solution_used[j] = True
                        break

        return ''.join(feedback)
        
    def compute_entropy(self,guess):
        # this function is to calculate the entropy based on each word generating the same patterns
        pattern_counts = Counter(
            self.generate_feedback_entropy(guess, solution)
            for solution in self.valid_guesses
        )
        total = sum(pattern_counts.values())
        probabilities = [count / total for count in pattern_counts.values()]
        entropy = -sum(p * math.log2(p) for p in probabilities)
        return entropy
        
    def entropy_table(self):
        """
        Generate an entropy table for all valid guesses.
        """
        table = {guess: self.compute_entropy(guess) for guess in self.valid_guesses}
        return table
        
    def filter_solutions(self, guess, feedback):
        """
        Narrow down the possible solutions based on feedback.
        :param guess: The guessed word.
        :param feedback: The feedback string ('G', 'Y', 'B').
        """
        def matches_feedback(solution):
            return self.generate_feedback_entropy(guess, solution) == feedback

        tmp = [
            solution for solution in self.valid_guesses if matches_feedback(solution)
        ]
        self.valid_guesses = tmp # updating the self.guesses available
        
    def baseline_entropy(self):
        remaining_attempts = self.max_guesses  # Initialize a local counter for guesses
        for attempt in range(1, remaining_attempts + 1):
            if not self.valid_guesses:
                print("No more valid guesses available.")
                return False, attempt
            if attempt==1:
                entropy_table = self.entropy
                total_entropy = sum(entropy_table.values())
            else:
                if len(self.valid_guesses)>1:
                    entropy_table = self.entropy_table()
                    total_entropy = sum(entropy_table.values())
                elif len(self.valid_guesses)==0:
                    print("No informative guesses left! Terminating game.")
                    remaining_attempts=6-attempt+1
                    return False,remaining_attempts
                else:
                    feedback = self.generate_feedback_entropy(self.valid_guesses,self.secret_word)
                    self.display_guess_feedback(self.valid_guesses, feedback)
                    return True, attempt
            probabilities = [entropy / total_entropy for entropy in entropy_table.values()]
            words = list(entropy_table.keys())
            guess = random.choices(words, weights=probabilities, k=1)[0]
            feedback = self.generate_feedback_entropy(guess,self.secret_word)
            self.display_guess_feedback(guess, feedback)
            self.filter_solutions(guess, feedback)
            if feedback == "G" * len(self.secret_word):
                return True, attempt
        return False, remaining_attempts
    
    def mcts_guessing(self):
        # This does not work yet, I just started writing some code here.
        remaining_attempts = self.max_guesses
        
        for attempt in range(1, remaining_attempts + 1):
            if not self.valid_guesses:
                print("No more valid guesses available.")
                return False, attempt

            scores = defaultdict(float)
            simulations = 100  # Number of simulations per word

            for word in self.valid_guesses:
                total_score = 0.0

                for _ in range(simulations):
                    temp_game = WordleSimulator('./kaggle_data/valid_solutions.csv', max_guesses=self.max_guesses, num_games=10, strategy="random")
                    temp_game.secret_word = self.secret_word
                    temp_game.valid_guesses = [word] + [w for w in self.valid_guesses if w != word]

                    feedback = temp_game.guess_word(word)
                    green_score = feedback.count('Green') * 2
                    yellow_score = feedback.count('Yellow') * 1
                    total_score += green_score + yellow_score

                scores[word] = total_score / simulations

            best_guess = max(scores, key=scores.get)
            feedback = self.guess_word(best_guess)
            self.display_guess_feedback(best_guess, feedback)

            if best_guess == self.secret_word:
                print("The secret word was guessed correctly!")
                return True, attempt

            gray_letters = {best_guess[i] for i in range(len(feedback)) if feedback[i] == 'Gray'}
            yellow_positions = {i: best_guess[i] for i in range(len(feedback)) if feedback[i] == 'Yellow'}
            green_positions = {i: best_guess[i] for i in range(len(feedback)) if feedback[i] == 'Green'}

            def valid_word(word):
                if any(letter in word for letter in gray_letters):
                    return False
                if any(word[i] == letter for i, letter in yellow_positions.items()):
                    return False
                if any(word[i] != letter for i, letter in green_positions.items()):
                    return False
                return True

            self.valid_guesses = [word for word in self.valid_guesses if valid_word(word)]

        print("Failed to guess the secret word within the allowed attempts.")
        return False, remaining_attempts
    
    def simulate_games(self):
        successes = 0
        attempts_distribution = []
        print(f"Simulating {self.num_games} games with strategy: {self.strategy}\n")
        self.entropy = self.entropy_table() # calculating the entropy of each word
        for _ in tqdm(range(self.num_games), desc="Simulating Games", unit="game"):
            self.reset_game()
            success, attempts = self.play_game()
            if success:
                successes += 1
            attempts_distribution.append(attempts)
        
        success_rate = successes / self.num_games
        average_attempts = sum(attempts_distribution) / self.num_games
        print(f"\nNumber of Successes: {successes}")
        print(f"Success Rate: {success_rate*100:.4f}%")
        print(f"Average Guesses Needed: {average_attempts:.4f}")
        self.plot_results(attempts_distribution)

    def plot_results(self, attempts):
        directory = f'plots_of_runs/{self.strategy}'
        os.makedirs(directory, exist_ok=True)
        plt.figure(figsize=(10, 5))
        bins = np.arange(1, self.max_guesses + 2) - 0.5
        counts, bins, patches = plt.hist(attempts, bins=bins, alpha=0.7, edgecolor='black')
        for patch, label in zip(patches, counts):
            if label > 0:
                plt.text(patch.get_x() + patch.get_width() / 2, patch.get_height(), f'{int(label)}', 
                        ha='center', va='bottom')

        plt.title('Distribution of Attempts to Guess the Secret Word')
        plt.xlabel('Number of Attempts')
        plt.ylabel('Frequency')
        plt.xticks(np.arange(1, self.max_guesses + 1))
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.savefig(f'{directory}/{self.max_guesses}guesses_{self.num_games}_games.png')

def main(args):
        game = WordleSimulator('./kaggle_data/valid_solutions.csv', max_guesses=args.max_guesses, num_games=args.num_games, strategy=args.strategy)
        game.simulate_games()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run the Wordle simulator with different strategies.")
    parser.add_argument('--strategy', type=str, default='random', help='The strategy to use for simulating the game.')
    parser.add_argument('--max_guesses', type=int, default=6, help='The maximum number of guesses allowed per game (6 is the normal amount for Wordle).')
    parser.add_argument('--num_games', type=int, default=1000, help='The number of games to simulate.')
    args = parser.parse_args()
    main(args)
