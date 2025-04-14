# %% [markdown]
# * Libraries

# %%
# 🧠 Core Libraries
import random
import numpy as np
import copy
import torch.optim as optim



# %% [markdown]
# * Environment setup

# %%
import random

class TicTacToeEnv:
    def __init__(self):
        self.players = ['X', 'O', '△']
        self.board_size = 5
        self.win_length = 4
        self.reset()

    def reset(self):
        self.board = [[' ' for _ in range(self.board_size)] for _ in range(self.board_size)]
        self.done = False
        self.winner = None
        self.current_player_index = 0
        self.turns = 0

        # Random blocked cells
        all_cells = [(i, j) for i in range(self.board_size) for j in range(self.board_size)]
        self.blocked_cells = random.sample(all_cells, 3)

        # Bonus and trap cells
        remaining_cells = list(set(all_cells) - set(self.blocked_cells))
        self.bonus_cells = random.sample(remaining_cells, 3)
        remaining_cells = list(set(remaining_cells) - set(self.bonus_cells))
        self.trap_cells = random.sample(remaining_cells, 2)

        # Track player state
        self.player_info = {
            'X': {'power_used': False, 'swap_used': False, 'last_move': None, 'bonus_cells_collected': [], 'trap_cells_hit': []},
            'O': {'power_used': False, 'swap_used': False, 'last_move': None, 'bonus_cells_collected': [], 'trap_cells_hit': []},
            '△': {'power_used': False, 'swap_used': False, 'last_move': None, 'bonus_cells_collected': [], 'trap_cells_hit': []}
        }

        # VISUAL MARKING: Set '*' on blocked cells
        for (r, c) in self.blocked_cells:
            self.board[r][c] = '*'

        return self.get_state()

    def get_state(self):
        return [row.copy() for row in self.board]  # return a deep copy of the board

    def check_winner(self, row, col, player):
        directions = [(-1, 0), (1, 0),   # vertical
                      (0, -1), (0, 1),   # horizontal
                      (-1, -1), (1, 1),  # diagonal ↘
                      (-1, 1), (1, -1)]  # diagonal ↙

        for dr, dc in directions:
            count = 1
            # Check one direction
            r, c = row + dr, col + dc
            while 0 <= r < self.board_size and 0 <= c < self.board_size and self.board[r][c] == player:
                count += 1
                r += dr
                c += dc

            # Check opposite direction
            r, c = row - dr, col - dc
            while 0 <= r < self.board_size and 0 <= c < self.board_size and self.board[r][c] == player:
                count += 1
                r -= dr
                c -= dc

            if count >= self.win_length:
                return True
        return False

    def step(self, action, use_power=False, use_swap=False):
        if self.done:
            return self.get_state(), 0, True, {"reason": "Game is over"}

        row, col = action
        player = self.players[self.current_player_index]

        # Check if cell is valid
        if (row, col) in self.blocked_cells or not (0 <= row < self.board_size) or not (0 <= col < self.board_size):
            return self.get_state(), -10, False, {"reason": "Invalid or blocked cell"}

        # Swap logic
        if use_swap:
            if self.player_info[player]['swap_used'] or self.player_info[player]['last_move'] is None:
                return self.get_state(), -5, False, {"reason": "Swap already used or no move to swap"}
            last_row, last_col = self.player_info[player]['last_move']
            self.board[last_row][last_col] = ' '
            self.player_info[player]['swap_used'] = True
            return self.get_state(), 1, False, {"action": "Swapped own move"}

        # Check if cell already filled
        if self.board[row][col] != ' ':
            return self.get_state(), -5, False, {"reason": "Cell already filled"}

        # Place mark
        self.board[row][col] = player
        self.player_info[player]['last_move'] = (row, col)

        # Bonus/trap cell logic
        reward = 0
        if (row, col) in self.bonus_cells:
            reward += 5
            self.player_info[player]['bonus_cells_collected'].append((row, col))
        elif (row, col) in self.trap_cells:
            reward -= 3
            self.player_info[player]['trap_cells_hit'].append((row, col))

        # Power move logic
        if use_power:
            if self.player_info[player]['power_used']:
                return self.get_state(), -5, False, {"reason": "Power move already used"}
            self.player_info[player]['power_used'] = True
            return self.get_state(), reward + 1, False, {"action": "Used power move: Place another mark"}

        # Check for win
        if self.check_winner(row, col, player):
            self.done = True
            self.winner = player
            return self.get_state(), reward + 10, True, {"winner": player}

        # Count turns (only if move was successful)
        self.turns += 1

        # Draw condition based on turn limit
        if self.turns >= 25:
            self.done = True
            return self.get_state(), reward, True, {"winner": None}

        # ✅ New: Fallback - Check if no empty cells remain
        empty_cells = sum(row.count(' ') for row in self.board)
        if empty_cells == 0:
            self.done = True
            return self.get_state(), reward, True, {"winner": None}

        # Continue to next player
        self.current_player_index = (self.current_player_index + 1) % len(self.players)
        return self.get_state(), reward, False, {"action": "Standard move"}



# %%
env = TicTacToeEnv()
env.reset()

# Pre-fill 3 marks for 'X' on the board
env.board[0][0] = 'X'
env.board[0][1] = 'X'
env.board[0][2] = 'X'

# Set the current player index to match 'X'
env.current_player_index = 0

# Now make the winning move
state, reward, done, info = env.step((0, 3))

# Output results
for row in state:
    print(row)
print("Reward:", reward)
print("Done:", done)
print("Info:", info)


# %%
env = TicTacToeEnv()
env.reset()

# Try placing a move
state, reward, done, info = env.step((1, 2))
print(f"Reward: {reward}, Done: {done}, Info: {info}")

# Try a blocked move
state, reward, done, info = env.step(env.blocked_cells[0])
print(f"Reward: {reward}, Done: {done}, Info: {info}")


# %%
env = TicTacToeEnv()
env.reset()
env.step((1, 1))


# %% [markdown]
# ---

# %% [markdown]
# * ## Building the DQN Agent

# %% [markdown]
# * 4.1 Code: State Encoder

# %%
import numpy as np

def encode_state(board):
    encoding = {
        ' ': 0,
        'X': 1,
        'O': 2,
        '△': 3,
        '*': -1
    }
    flat_encoded = [encoding[cell] for row in board for cell in row]
    return np.array(flat_encoded, dtype=np.float32)


# %%
env = TicTacToeEnv()
state = env.reset()

encoded = encode_state(state)
print("Encoded shape:", encoded.shape)
print(encoded.reshape(5, 5))


# %% [markdown]
# * 4.2: Build the DQN Model (PyTorch)

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self, input_dim=25, hidden_dim=128, output_dim=25):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x)  # Q-values for each action


# %%
model = DQN()
dummy_input = torch.tensor(encode_state(env.reset())).unsqueeze(0)  # shape: [1, 25]
output = model(dummy_input)
print("Output shape:", output.shape)
print(output)


# %% [markdown]
# * 4.3 – Build the Replay Buffer

# %%
from collections import deque
import random
import numpy as np

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        # Save as tuples
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        # Randomly sample a batch
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.stack(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.stack(next_states),
            np.array(dones, dtype=np.uint8)
        )

    def __len__(self):
        return len(self.buffer)


# %%
rb = ReplayBuffer(capacity=1000)

# Push a few dummy transitions
s1 = encode_state(env.reset())
s2 = encode_state(env.reset())
rb.push(s1, 3, 1.0, s2, False)

print("Buffer size:", len(rb))
sampled = rb.sample(1)
print("Sample shape:", [x.shape for x in sampled])


# %% [markdown]
# * 4.4: Training Loop (Core DQN Engine - epsilon-greedy, optimizer, target update)

# %%
def train_dqn(env, model, buffer, episodes=500, batch_size=64, gamma=0.99, 
              epsilon_start=1.0, epsilon_end=0.1, epsilon_decay=0.995, learning_rate=1e-3):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    epsilon = epsilon_start
    all_rewards = []
    all_outcomes = []  # List to store "win" or "draw" for each episode
    
    for episode in range(episodes):
        state = encode_state(env.reset())
        total_reward = 0
        done = False
        outcome = None  # Will be set to "win" or "draw" at the end
        
        while not done:
            # Get list of valid actions: only empty cells (' ')
            valid_actions = [i for i in range(25) if env.board[i // 5][i % 5] == ' ']
            if not valid_actions:
                break  # No available move

            # ε-greedy action selection from valid actions only
            if random.random() < epsilon:
                action = random.choice(valid_actions)
            else:
                with torch.no_grad():
                    q_values = model(torch.tensor(state).unsqueeze(0))
                    action = max(valid_actions, key=lambda a: q_values[0][a].item())
                    
            # Convert action number into board coordinates
            row, col = divmod(action, env.board_size)
            
            next_state_raw, reward, done, info = env.step((row, col))
            next_state = encode_state(next_state_raw)
            
            buffer.push(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            # Train the model once we have enough samples in the replay buffer
            if len(buffer) >= batch_size:
                states, actions, rewards, next_states, dones = buffer.sample(batch_size)

                states = torch.tensor(states)
                actions = torch.tensor(actions, dtype=torch.long)
                rewards = torch.tensor(rewards)
                next_states = torch.tensor(next_states)
                dones = torch.tensor(dones)

                q_values = model(states)
                next_q_values = model(next_states).detach()

                target_q = rewards + gamma * next_q_values.max(1)[0] * (1 - dones)
                predicted_q = q_values.gather(1, actions.unsqueeze(1)).squeeze()

                loss = F.mse_loss(predicted_q, target_q)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
        # Determine outcome based on info in the last step
        if "winner" in info and info["winner"]:
            outcome = "win"
        else:
            outcome = "draw"

        all_rewards.append(total_reward)
        all_outcomes.append(outcome)
        epsilon = max(epsilon_end, epsilon * epsilon_decay)

        if (episode + 1) % 50 == 0:
            print(f"Episode {episode+1}/{episodes} | Reward: {total_reward:.2f} | Epsilon: {epsilon:.3f} | Outcome: {outcome}")

    return all_rewards, all_outcomes


# %%
env = TicTacToeEnv()
model = DQN()
buffer = ReplayBuffer()

all_rewards, all_outcomes = train_dqn(env, model, buffer, episodes=1500)


# %% [markdown]
# ---

# %% [markdown]
# * Plotting Win & Draw Percentages

# %%
import numpy as np
import matplotlib.pyplot as plt

def plot_win_draw(outcomes):
    episodes = np.arange(1, len(outcomes)+1)
    # Convert outcomes to binary arrays: win -> 1, draw -> 0
    wins = np.array([1 if outcome == "win" else 0 for outcome in outcomes])
    draws = np.array([1 if outcome == "draw" else 0 for outcome in outcomes])
    
    # Compute cumulative percentage averages
    cum_wins = np.cumsum(wins) / episodes * 100
    cum_draws = np.cumsum(draws) / episodes * 100
    
    plt.figure(figsize=(10, 5))
    plt.plot(episodes, cum_wins, label="Win %", color='green')
    plt.plot(episodes, cum_draws, label="Draw %", color='blue')
    plt.xlabel("Episode")
    plt.ylabel("Percentage (%)")
    plt.title("Cumulative Win and Draw Percentages During Training")
    plt.legend()
    plt.grid(True)
    plt.show()


# %%
plot_win_draw(all_outcomes)


# %% [markdown]
# * Plot the Reward Curve

# %%
def plot_rewards(rewards):
    plt.figure(figsize=(10, 5))
    plt.plot(rewards, label='Total reward per episode', color='orange')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('DQN Training Rewards')
    plt.grid(True)
    plt.legend()
    plt.show()


# %%
plot_rewards(all_rewards)


# %% [markdown]
# * Saving the trained model

# %%
import torch

torch.save(model.state_dict(), 'dqn_tictactoe_model.pt')
print("Model saved as dqn_tictactoe_model.pt")


# %% [markdown]
# ---

# %% [markdown]
# * Trained Agent Play a Game

# %%
def agent_play(model, env):
    def print_board(state):
        for row in state:
            print(' | '.join(row))
            print('-' * 17)

    state = env.reset()
    model.eval()
    done = False
    print("=== Game Start ===\n")
    print_board(state)

    while not done:
        encoded = torch.tensor(encode_state(state)).unsqueeze(0)
        with torch.no_grad():
            q_values = model(encoded)

        # Only choose valid actions
        valid_actions = [i for i in range(25) if env.board[i // 5][i % 5] == ' ']
        if not valid_actions:
            break

        action = max(valid_actions, key=lambda a: q_values[0][a].item())
        row, col = divmod(action, env.board_size)

        state, reward, done, info = env.step((row, col))

        print(f"\nAgent plays: ({row}, {col})")
        print_board(state)

        if done:
            print("\n🎯 Game Over")
            if "winner" in info and info["winner"]:
                print("🏆 Winner:", info["winner"])
            else:
                print("🤝 It's a draw")
        
        



# %%
agent_play(model, TicTacToeEnv())


# %% [markdown]
# ---

# %% [markdown]
# * play_human_vs_agent_game()

# %%
def play_human_vs_agent_game(model):
    import torch

    def print_board(state):
        for row in state:
            print(' | '.join(row))
            print('-' * 17)

    env = TicTacToeEnv()
    model.eval()
    state = env.reset()
    print("=== Game Start ===")
    print_board(state)

    while not env.done:
        player = env.players[env.current_player_index]
        print(f"\n👉 {player}'s turn")

        if player == 'X':  # Human
            use_power = False
            use_swap = False

            # Swap?
            if not env.player_info['X']['swap_used'] and env.player_info['X']['last_move']:
                swap_choice = input("Do you want to use your SWAP move? (y/n): ").lower()
                if swap_choice == 'y':
                    use_swap = True

            # Power move?
            if not env.player_info['X']['power_used']:
                power_choice = input("Do you want to use your POWER move (2 moves)? (y/n): ").lower()
                if power_choice == 'y':
                    use_power = True

            try:
                move_input = input("Enter your move as row,col (0-indexed): ")
                row, col = map(int, move_input.strip().split(','))
                state, reward, done, info = env.step((row, col), use_power=use_power, use_swap=use_swap)
                print_board(state)
                if done:
                    break

                # If power move, allow second move immediately
                if use_power and not done:
                    move_input = input("Enter your second move (power move): ")
                    row, col = map(int, move_input.strip().split(','))
                    state, reward, done, info = env.step((row, col))
                    print_board(state)

            except Exception as e:
                print("Invalid input. Try again.")
                continue

        elif player == 'O':  # Agent
            encoded = torch.tensor(encode_state(state)).unsqueeze(0)
            with torch.no_grad():
                q_values = model(encoded)

            valid_actions = [i for i in range(25) if env.board[i // 5][i % 5] == ' ']
            if not valid_actions:
                break
            action = max(valid_actions, key=lambda a: q_values[0][a].item())
            row, col = divmod(action, env.board_size)

            use_power = not env.player_info['O']['power_used']
            use_swap = not env.player_info['O']['swap_used'] and env.player_info['O']['last_move'] is not None

            print(f"🤖 Agent plays: ({row}, {col}) | Power: {use_power} | Swap: {use_swap}")
            state, reward, done, info = env.step((row, col), use_power=use_power, use_swap=use_swap)
            print_board(state)

            if use_power and not done:
                # Choose second best move
                valid_actions = [i for i in range(25) if env.board[i // 5][i % 5] == ' ']
                if valid_actions:
                    action = max(valid_actions, key=lambda a: q_values[0][a].item())
                    row, col = divmod(action, env.board_size)
                    print(f"🤖 Agent plays POWER move 2nd: ({row}, {col})")
                    state, reward, done, info = env.step((row, col))
                    print_board(state)

        elif player == '△':  # Random bot
            valid_actions = [(i, j) for i in range(5) for j in range(5) if env.board[i][j] == ' ']
            if valid_actions:
                row, col = random.choice(valid_actions)
                print(f"🤖 Random Bot plays: ({row}, {col})")
                state, reward, done, info = env.step((row, col))
                print_board(state)

        if env.done:
            break

    print("\n🎯 Game Over")
    if env.winner:
        print(f"🏆 Winner: {env.winner}")
    else:
        print("🤝 It's a draw")


# %% [markdown]
# ---

# %% [markdown]
# * Testing the setup

# %%
env = TicTacToeEnv()
state = env.reset()

# Print the board to check
for row in state:
    print(row)

print("\nBlocked cells:", env.blocked_cells)
print("Bonus cells:", env.bonus_cells)
print("Trap cells:", env.trap_cells)

def plot_combined(rewards, outcomes):
    episodes = np.arange(1, len(rewards)+1)
    wins = np.array([1 if outcome == "win" else 0 for outcome in outcomes])
    draws = np.array([1 if outcome == "draw" else 0 for outcome in outcomes])
    cum_wins = np.cumsum(wins) / episodes * 100
    cum_draws = np.cumsum(draws) / episodes * 100

    fig, axs = plt.subplots(1, 2, figsize=(14, 5))

    axs[0].plot(rewards, color='orange')
    axs[0].set_title("Total Reward per Episode")
    axs[0].set_xlabel("Episode")
    axs[0].set_ylabel("Total Reward")
    axs[0].grid(True)

    axs[1].plot(cum_wins, label="Win %", color='green')
    axs[1].plot(cum_draws, label="Draw %", color='blue')
    axs[1].set_title("Cumulative Win & Draw Percentages")
    axs[1].set_xlabel("Episode")
    axs[1].set_ylabel("Percentage (%)")
    axs[1].legend()
    axs[1].grid(True)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Optional: Only if you want to train interactively again
    pass

    

