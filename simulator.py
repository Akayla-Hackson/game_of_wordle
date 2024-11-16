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

init(autoreset=True)  # Auto-reset colorama colors

class WordleSimulator:
    def __init__(self, solutions_file, max_guesses, num_games, strategy):
        self.valid_solutions = pd.read_csv(solutions_file, header=None).iloc[:,0].tolist()
        self.max_guesses = int(max_guesses)
        self.num_games = int(num_games)
        self.strategy = strategy
        self.reset_game()

    def reset_game(self):
        self.secret_word = random.choice(self.valid_solutions)
        self.valid_guesses = self.valid_solutions.copy()

        print(f"The secret word is: {self.secret_word}")

    def play_game(self):
        if self.strategy == 'random':
            return self.random_guessing()
        elif self.strategy == "random_feedback":
            return self.random_guessing_with_feedback()
        elif self.strategy == 'baseline':
            return self.baseline_guessing()
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
    


    def random_guessing_with_feedback(self):
        remaining_attempts = self.max_guesses
        # possible_words = self.valid_guesses.copy()
        print("Initial possible words:", self.valid_guesses)

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
            else:
                if guess in self.valid_guesses:
                    self.valid_guesses.remove(guess)

            # update possible_words based on feedback
            updated_possible_words = []
            for word in self.valid_guesses:
                if self.is_valid_guess(word, guess, feedback):
                    updated_possible_words.append(word)
                    
            self.valid_guesses = updated_possible_words
            print("Updated possible words after feedback:", self.valid_guesses)
        print("Failed to guess the secret word within the allowed attempts.")
        return False, remaining_attempts

    def is_valid_guess(self, word, guess, feedback):
        # map each letter in guess to its feedback
        word_counts = Counter(word)
        guess_feedback = defaultdict(list)
        for index, (char, fb) in enumerate(zip(guess, feedback)):
            guess_feedback[fb].append(char)
        
        # green feedback: exact matches
        for index, (g_char, fb) in enumerate(zip(guess, feedback)):
            if fb == 'Green' and word[index] != g_char:
                # print(f"Removing '{word}' because at index {index}, expected '{g_char}' but found '{word[index]}'.")
                return False

        # yellow feedback: right letter, wrong position
        for index, (g_char, fb) in enumerate(zip(guess, feedback)):
            if fb == 'Yellow':
                if word[index] == g_char:
                    # print(f"Removing '{word}' because '{g_char}' should not be at index {index}.")
                    return False
                if word_counts[g_char] <= 0:
                    # print(f"Removing '{word}' because there are not enough '{g_char}' to match feedback.")
                    return False
                word_counts[g_char] -= 1

        # gray feedback: letter shouldn't be present
        for g_char in guess_feedback['Gray']:
            if word_counts[g_char] > 0 and g_char not in guess_feedback['Green'] and g_char not in guess_feedback['Yellow']:
                # print(f"Removing '{word}' because it contains '{g_char}' which should not be present at all based on feedback.")
                return False
            word_counts[g_char] -= 1

        return True


    def simulate_games(self):
        successes = 0
        attempts_distribution = []
        print(f"Simulating {self.num_games} games with strategy: {self.strategy}\n")
        
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

    def test_guess_word(self):
        game = WordleSimulator('./kaggle_data/valid_solutions.csv', max_guesses=args.max_guesses, num_games=args.num_games, strategy=args.strategy)
        for guess in self.valid_guesses:
            feedback = game.guess_word(guess)
            print(f"Guess: {guess} -> Feedback: {feedback}")

    
def main(args):
        game = WordleSimulator('./kaggle_data/valid_solutions.csv', max_guesses=args.max_guesses, num_games=args.num_games, strategy=args.strategy)
        game.simulate_games()
        # game.test_guess_word()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run the Wordle simulator with different strategies.")
    parser.add_argument('--strategy', type=str, default='random', help='The strategy to use for simulating the game.')
    parser.add_argument('--max_guesses', type=int, default=6, help='The maximum number of guesses allowed per game (6 is the normal amount for Wordle).')
    parser.add_argument('--num_games', type=int, default=1000, help='The number of games to simulate.')
    args = parser.parse_args()
    main(args)



    # python simulator.py --strategy random --max_guesses 6 --num_games 1000
