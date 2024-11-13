import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


data_path = './kaggle_data/valid_solutions.csv'
df = pd.read_csv(data_path, header=None, names=['word'])

# Init empty dict for the frequency data
letter_position_frequency = {i: {} for i in range(5)}

# populate the dict with frequencies of each letter in each position
for word in df['word']:
    for position, letter in enumerate(word.strip().lower()):  # ensure lowercase and no surrounding whitespace
        if letter in letter_position_frequency[position]:
            letter_position_frequency[position][letter] += 1
        else:
            letter_position_frequency[position][letter] = 1

# convert the dict to a DataFrame (easier plotting)
position_data = []
for position, freq_dict in letter_position_frequency.items():
    for letter, count in freq_dict.items():
        position_data.append({'Position': position, 'Letter': letter, 'Frequency': count})

df_position = pd.DataFrame(position_data)

# create a matrix where each row is a letter and each column is a position
heatmap_data = df_position.pivot(index='Letter', columns='Position', values='Frequency')
total_count = df['word'].count()
heatmap_data = heatmap_data.fillna(0).astype(int)

# plot the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(heatmap_data, annot=True, fmt="d", cmap='viridis')
plt.title(f'Letter Position Frequency Heatmap\nTotal Count of Words: {total_count}')
plt.xlabel('Position in Word')
plt.ylabel('Letter')
plt.savefig('kaggle_data_letter_position_frequency_heatmap.png')

