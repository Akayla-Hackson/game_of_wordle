import math
from collections import Counter

class WordleEntropy:
    def __init__(self, valid_guesses, valid_solutions):
        self.valid_guesses = valid_guesses
        self.valid_solutions = valid_solutions

    def generate_feedback(self, guess, solution):
        """
        Generate feedback for a guess compared to a solution.
        'G' = Green, 'Y' = Yellow, 'B' = Gray (Black).
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

    def compute_entropy(self, guess):
        """
        Compute the entropy of a word guess over all possible solutions.
        """
        pattern_counts = Counter(
            self.generate_feedback(guess, solution)
            for solution in self.valid_solutions
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

class WordleGame:
    def __init__(self, valid_guesses, valid_solutions,entropy_table):
        self.valid_guesses = valid_guesses
        self.valid_solutions = valid_solutions
        self.secret_word = None  # Will be set during play
        self.entropy = entropy_table

    def generate_feedback(self, guess, solution):
        """
        Generate feedback for a guess compared to a solution.
        'G' = Green, 'Y' = Yellow, 'B' = Gray (Black).
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

    def compute_entropy(self, guess):
        """
        Compute the entropy of a word guess over all possible solutions.
        """
        pattern_counts = Counter(
            self.generate_feedback(guess, solution)
            for solution in self.valid_solutions
        )

        total = sum(pattern_counts.values())
        probabilities = [count / total for count in pattern_counts.values()]
        entropy = -sum(p * math.log2(p) for p in probabilities)
        return entropy

    def filter_solutions(self, guess, feedback):
        """
        Narrow down the possible solutions based on feedback.
        :param guess: The guessed word.
        :param feedback: The feedback string ('G', 'Y', 'B').
        """
        def matches_feedback(solution):
            return self.generate_feedback(guess, solution) == feedback

        self.valid_solutions = [
            solution for solution in self.valid_solutions if matches_feedback(solution)
        ]
        self.valid_guesses = self.valid_solutions
        # print(self.valid_solutions)

    def entropy_table(self):
        """
        Generate an entropy table for all valid guesses.
        """
        return {guess: self.compute_entropy(guess) for guess in self.valid_guesses}

    def play(self, secret_word, max_attempts=6):
        """
        Play the game by always choosing the word with the highest entropy until the secret word is found or attempts are exhausted.
        :param secret_word: The word to be guessed.
        :param max_attempts: Maximum number of allowed attempts.
        :return: Number of attempts used and success status (True if guessed, False otherwise).
        """
        self.secret_word = secret_word
        attempts = 0
    
        while attempts < max_attempts:
            attempts += 1
    
            # Calculate entropy table
            if attempts == 1:
                entropy_table = self.entropy
                total_entropy = sum(entropy_table.values())
            else:
                if len(self.valid_guesses)>1:
                    entropy_table = self.entropy_table()
                    total_entropy = sum(entropy_table.values())
                elif len(self.valid_guesses)==0:
                    print("No informative guesses left! Terminating game.")
                    attempts=6
                    return attempts, False
                else:
                    return attempts+1, True

            probabilities = [entropy / total_entropy for entropy in entropy_table.values()]

        # Choose a word randomly based on probabilities
            words = list(entropy_table.keys())
            best_guess = random.choices(words, weights=probabilities, k=1)[0]
    
            # Generate feedback for the guess
            feedback = self.generate_feedback(best_guess, self.secret_word)
            print(f"Attempt {attempts}: Guess = {best_guess}, Feedback = {feedback}")
            # Remove the guessed word from valid guesses
            
    
            # Update possible solutions based on feedback
            self.filter_solutions(best_guess, feedback)
            # self.valid_guesses.remove(best_guess)
            
    
            # Check if the guess matches the secret word
            if feedback == "G" * len(self.secret_word):
                return attempts, True
    
        return attempts, False

  def simulate_games(vg, vs,entropy_table, num_games=10):
    results = []
    for _ in range(num_games):
        # Choose a random secret word
        secret_word = random.choice(vg)
        print(secret_word)
        game = WordleGame(vg, vg,entropy_table)  # Pass copies to avoid modifying original lists
        attempts, won = game.play(secret_word)
        game=[]
        results.append((attempts, won))
    return results

def main(args):
        valid_guesses = pd.read_csv('valid_solutions.csv', header=None).iloc[:, 0].tolist()
        valid_solutions = pd.read_csv('valid_solutions.csv', header=None).iloc[:, 0].tolist()
        solver = WordleEntropy(valid_guesses, valid_guesses)
        entropy_table = solver.entropy_table()
        
        num_games = 1000
        results = simulate_games(valid_guesses,valid_guesses,entropy_table, num_games)
        
        # Print results
        total_attempts = sum(r[0] for r in results)
        wins = sum(1 for r in results if r[1])
        average_attempts = total_attempts / len(results)
        
        print(f"Simulated {num_games} games.")
        print(f"Wins: {wins}/{num_games}")
        print(f"Average Attempts: {average_attempts:.2f}")
        print(f"Details: {results}")

if __name__ == '__main__':
  main()
