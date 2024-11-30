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
    def __init__(self, solutions_file, max_guesses, num_games, strategy, prioritize_success=False):
        self.prioritize_success = prioritize_success
        self.valid_solutions = pd.read_csv(solutions_file, header=None).iloc[:,0].tolist()
        self.max_guesses = int(max_guesses)
        self.num_games = int(num_games)
        self.strategy = strategy

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

        feedback = self.get_feedback(guess, self.secret_word)
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
            self.valid_guesses.remove(guess)
            self.display_guess_feedback(guess, feedback)

            if guess == self.secret_word:
                print("The secret word was guessed correctly!")
                return True, attempt
        print("Failed to guess the secret word within the allowed attempts.")
        return False, remaining_attempts

    def baseline_guessing(self):
        remaining_attempts = self.max_guesses
        possible_words = self.valid_guesses.copy()
        history = []

        for attempt in range(1, remaining_attempts + 1):
            if not possible_words:
                print("No more valid guesses available.")
                return False, attempt

            guess = random.choice(possible_words)
            possible_words.remove(guess)
            feedback = self.guess_word(guess)
            self.valid_guesses.remove(guess)
            self.display_guess_feedback(guess, feedback)
            history.append((guess, feedback))

            if guess == self.secret_word:
                print("The secret word was guessed correctly!")
                return True, attempt

            possible_words = self.filter_possible_words_real(possible_words, guess, feedback)
        print("Failed to guess the secret word within the allowed attempts.")
        return False, remaining_attempts

    def mcts_guessing(self):
        # Initialize the game state
        remaining_attempts = self.max_guesses
        possible_words = self.valid_guesses.copy()
        history = []
        
        # Step 1: Make the first guess randomly from TALES, CRANE, or SALET
        first_guess_choices = ["crane"]
        first_guess = random.choice(first_guess_choices)
        
        print(f"First guess: {first_guess}")
        feedback = self.guess_word(first_guess)
        self.display_guess_feedback(first_guess, feedback)
        self.valid_guesses.remove(first_guess)
        history.append((first_guess, feedback))
        
        # Check if the first guess is correct
        if first_guess == self.secret_word:
            print("The secret word was guessed correctly on the first attempt!")
            return True, 1
        
        # Filter possible words based on the first guess feedback
        possible_words = self.filter_possible_words_real(possible_words, first_guess, feedback)
        
        # Print updated game state after the first guess
        print(f"Updated game state after the first guess:")
        print(f"Remaining attempts: {remaining_attempts - 1}")
        print(f"Possible words: {len(possible_words)}")
        print(f"History: {history}")
        
        # Step 2: Proceed with MCTS for the remaining attempts
        for attempt in range(2, remaining_attempts + 1):
            # Create the root node of the MCTS tree
            root = self.Node(
                parent=None,
                state={'history': history.copy(), 'possible_words': possible_words.copy()},
                untried_actions=possible_words.copy()
            )
            
            # Run the MCTS algorithm for a fixed number of iterations
            num_iterations = 1000  # Adjusted for performance
            for _ in range(num_iterations):
                # Select a node to expand
                node = root
                while node.untried_actions == [] and node.children != []:
                    node = node.select_child()
                
                # Expand the selected node
                if node.untried_actions != []:
                    action = self.select_next_guess(node.untried_actions, node.state)
                    node.untried_actions.remove(action)
                    new_state = self.simulate_action(node.state, action)
                    untried_actions = new_state['possible_words']
                    child_node = node.add_child(action, new_state, untried_actions)
                    node = child_node
                
                # Simulate a random playout from the expanded node
                reward = self.simulate_heuristic_playout(node.state, remaining_attempts - attempt)
                
                # Backpropagate the reward to the root node
                while node is not None:
                    node.visit_count += 1
                    node.total_reward += reward
                    node = node.parent
            
            # Choose the best action from the root node
            if root.children:
                best_child = max(root.children, key=lambda c: c.visit_count)
                guess = best_child.action_taken
            else:
                guess = random.choice(possible_words)
            
            # Get feedback from the actual game
            feedback = self.guess_word(guess)
            self.display_guess_feedback(guess, feedback)
            self.valid_guesses.remove(guess)
            
            # Check if the guess is correct
            if guess == self.secret_word:
                print("The secret word was guessed correctly!")
                return True, attempt
            
            # Update the game state
            history.append((guess, feedback))
            possible_words = self.filter_possible_words(possible_words, guess, feedback)
            
            # Print updated game state
            print(f"Updated game state:")
            print(f"Remaining attempts: {remaining_attempts - attempt}")
            print(f"Possible words: {len(possible_words)}")
            print(f"History: {history}")
        
        print("Failed to guess the secret word within the allowed attempts.")
        return False, remaining_attempts

    def select_next_guess(self, possible_guesses, state):
        # Prioritize guesses with the highest weights
        weights = self.calculate_word_weights(possible_guesses, state)
        max_weight = max(weights.values())
        best_guesses = [word for word, weight in weights.items() if weight == max_weight]
        return random.choice(best_guesses)

    def calculate_word_weights(self, words, state):
        # Calculate weights based on green and yellow letters
        history = state['history']
        green_positions = {}
        yellow_letters = set()

        for guess, feedback in history:
            for i, (letter, fb) in enumerate(zip(guess, feedback)):
                if fb == 'Green':
                    green_positions[i] = letter
                elif fb == 'Yellow':
                    yellow_letters.add(letter)

        weights = {}
        for word in words:
            weight = 0
            for i, letter in enumerate(word):
                if i in green_positions and letter == green_positions[i]:
                    weight += 30
                elif letter in yellow_letters:
                    weight += 10
            weights[word] = weight
        return weights

    def simulate_heuristic_playout(self, state, remaining_attempts):
        possible_words = state['possible_words']
        if not possible_words:
            return 0  # No possible words left
        secret_word = random.choice(possible_words)
        current_possible_words = possible_words.copy()
        history = state['history'].copy()
        attempts = len(history)
        while attempts < remaining_attempts:
            if not current_possible_words:
                return 0
            # Select guess based on heuristics
            if self.
            guess = self.select_next_guess(current_possible_words, state)
            feedback = self.get_feedback(guess, secret_word)
            history.append((guess, feedback))
            if guess == secret_word:
                # Reward inversely proportional to the number of attempts
                return 10 + (remaining_attempts - attempts) * 1
            current_possible_words = self.filter_possible_words(current_possible_words, guess, feedback)
            attempts += 1
        return 0  # Failed to guess the secret word

    def simulate_action(self, state, action):
        possible_words = state['possible_words']
        if not possible_words:
            return state  # No possible words left
        # Simulate feedback as if the action was made
        secret_word = random.choice(possible_words)
        feedback = self.get_feedback(action, secret_word)
        new_possible_words = self.filter_possible_words(possible_words, action, feedback)
        new_history = state['history'] + [(action, feedback)]
        return {'history': new_history, 'possible_words': new_possible_words}

    def get_feedback(self, guess, secret_word):
        feedback = ['Gray'] * len(guess)
        secret_word_used = [False] * len(secret_word)
        guess_used = [False] * len(guess)

        for i in range(len(guess)):
            if guess[i] == secret_word[i]:
                feedback[i] = 'Green'
                secret_word_used[i] = True
                guess_used[i] = True

        for i in range(len(guess)):
            if not guess_used[i]:
                for j in range(len(secret_word)):
                    if not secret_word_used[j] and guess[i] == secret_word[j]:
                        feedback[i] = 'Yellow'
                        secret_word_used[j] = True
                        guess_used[i] = True
                        break
        return feedback

    def filter_possible_words(self, possible_words, guess, feedback):
        gray_letters = {guess[i] for i in range(len(feedback)) if feedback[i] == 'Gray'}
        yellow_positions = {i: guess[i] for i in range(len(feedback)) if feedback[i] == 'Yellow'}
        yellow_letters = {guess[i] for i in range(len(feedback)) if feedback[i] == 'Yellow'}
        green_positions = {i: guess[i] for i in range(len(feedback)) if feedback[i] == 'Green'}
        green_letters = {guess[i] for i in range(len(feedback)) if feedback[i] == 'Green'}

        def valid_word(word):
            if word==guess:
                return False
            if any(word[i] != letter for i, letter in green_positions.items()):
                return False
            if any(word[i] == letter for i, letter in yellow_positions.items()):
                return False
            if not yellow_letters.issubset(set(word)):
                return False
            if any(letter in word for letter in gray_letters):
                if not (any(letter in green_letters for letter in gray_letters) or any(letter in yellow_letters for letter in gray_letters)):
                    return False
            return True

        return [word for word in possible_words if valid_word(word)]
        
    def filter_possible_words_real(self, possible_words, guess, feedback):
        gray_letters = {guess[i] for i in range(len(feedback)) if feedback[i] == 'Gray'}
        yellow_positions = {i: guess[i] for i in range(len(feedback)) if feedback[i] == 'Yellow'}
        yellow_letters = {guess[i] for i in range(len(feedback)) if feedback[i] == 'Yellow'}
        green_positions = {i: guess[i] for i in range(len(feedback)) if feedback[i] == 'Green'}
        green_letters = {guess[i] for i in range(len(feedback)) if feedback[i] == 'Green'}

        def valid_word(word):
            if word==guess:
                return False
            if any(word[i] != letter for i, letter in green_positions.items()):
                return False
            if any(word[i] == letter for i, letter in yellow_positions.items()):
                return False
            if not yellow_letters.issubset(set(word)):
                return False
            if any(letter in word for letter in gray_letters):
                if not (any(letter in green_letters for letter in gray_letters) or any(letter in yellow_letters for letter in gray_letters)):
                    return False
            return True

        return [word for word in possible_words if valid_word(word)]

    class Node:
        def __init__(self, parent, state, untried_actions):
            self.parent = parent
            self.children = []
            self.state = state  # Contains 'history' and 'possible_words'
            self.untried_actions = untried_actions  # Possible guesses from this state
            self.visit_count = 0
            self.total_reward = 0
            self.action_taken = None  # The action (guess) that led to this node

        def select_child(self):
            # UCT formula
            C = 0.5 # Exploration parameter
            total_visits = self.visit_count
            log_total_visits = np.log(total_visits) if total_visits > 0 else 0

            def uct_value(child):
                if child.visit_count == 0:
                    return float('inf')  # Encourage exploration of unvisited nodes
                return (child.total_reward / child.visit_count) + C * np.sqrt(log_total_visits / child.visit_count)

            print(f"Selecting child from {len(self.children)} children")
            selected_child = max(self.children, key=uct_value)

            return selected_child

        def add_child(self, action, state, untried_actions):
            child_node = WordleSimulator.Node(parent=self, state=state, untried_actions=untried_actions)
            child_node.action_taken = action
            self.children.append(child_node)
            print(f"Child node added to {self.state["history"]} with possible words {len(self.state["possible_words"])}: {child_node.state["history"]} with possible words {len(child_node.state["possible_words"])}")
            return child_node

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
        plt.close()

def main(args):
    game = WordleSimulator('./kaggle_data/valid_solutions.csv', max_guesses=args.max_guesses, num_games=args.num_games, strategy=args.strategy)
    game.simulate_games()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run the Wordle simulator with different strategies.")
    parser.add_argument('--strategy', type=str, default='random', help='The strategy to use for simulating the game.')
    parser.add_argument('--max_guesses', type=int, default=6, help='The maximum number of guesses allowed per game (6 is the normal amount for Wordle).')
    parser.add_argument('--prioritize_success', type=bool, default=False, help='Prioritize success over attempts.')
    parser.add_argument('--num_games', type=int, default=100, help='The number of games to simulate.')
    args = parser.parse_args()
    main(args)
