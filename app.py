import streamlit as st
import numpy as np
import torch
import random
from gamelogic import TicTacToeEnv, DQN, encode_state

# --- Initialize ---
if "env" not in st.session_state:
    st.session_state.env = TicTacToeEnv()
    st.session_state.model = DQN()
    st.session_state.model.load_state_dict(torch.load("dqn_tictactoe_model.pt"))
    st.session_state.model.eval()
    st.session_state.done = False
    st.session_state.message = "Game Start!"
    st.session_state.last_action = ""
    st.session_state.turn_count = 0
    st.session_state.power_used = {'X': False, 'O': False, '△': False}
    st.session_state.swap_used = {'X': False, 'O': False, '△': False}
    st.session_state.awaiting_second_power = False
    st.session_state.await_agent_moves = False
    st.session_state.use_power = False
    st.session_state.use_swap = False
    st.session_state.power_pending_move = None

env = st.session_state.env
model = st.session_state.model
players = ['X', 'O', '△']

# --- Helper ---
def get_cell_display(cell):
    colors = {'X': 'blue', 'O': 'orange', '△': 'purple', '*': 'gray', '+': 'green', '-': 'red', ' ': 'black'}
    return f"<span style='color:{colors.get(cell, 'black')}; font-weight:bold'>{cell}</span>"

def dqn_move():
    encoded = torch.tensor(encode_state(env.get_state()), dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        q_values = model(encoded)
    valid_actions = [i for i in range(25) if env.board[i // 5][i % 5] == ' ']
    if not valid_actions:
        return None
    action = max(valid_actions, key=lambda a: q_values[0][a].item())
    return divmod(action, 5)

# --- Gameplay handlers ---
def play_human_move(row, col):
    if env.done or st.session_state.done:
        return

    player = 'X'

    if st.session_state.awaiting_second_power:
        if env.board[row][col] != ' ':
            st.session_state.message = "Invalid move!"
            return
        state, reward, done, info = env.step((row, col))
        st.session_state.turn_count += 1
        st.session_state.last_action = f"{player} used second POWER move at ({row}, {col})"
        st.session_state.awaiting_second_power = False
        st.session_state.power_pending_move = None
        st.session_state.use_power = False
        st.session_state.use_swap = False
        if done:
            st.session_state.done = True
            st.session_state.message = f"🏆 Winner: {info['winner']}" if info.get("winner") else "🤝 It's a draw."
        else:
            env.current_player_index = (env.current_player_index + 1) % 3
            st.session_state.await_agent_moves = True
        return

    if env.board[row][col] != ' ':
        st.session_state.message = "Invalid move!"
        return

    use_power = st.session_state.use_power and not st.session_state.power_used[player]
    use_swap = st.session_state.use_swap and not st.session_state.swap_used[player]

    if use_swap and env.player_info[player]['last_move']:
        r, c = env.player_info[player]['last_move']
        env.board[r][c] = ' '
        st.session_state.swap_used[player] = True
        st.session_state.use_swap = False
        st.session_state.last_action = f"{player} used SWAP at ({r}, {c})"
        return

    state, reward, done, info = env.step((row, col))
    st.session_state.turn_count += 1
    st.session_state.last_action = f"{player} moved to ({row}, {col})"

    if use_power:
        st.session_state.awaiting_second_power = True
        st.session_state.power_used[player] = True
        st.session_state.power_pending_move = (row, col)
        st.session_state.use_power = False
        return

    if done:
        st.session_state.done = True
        st.session_state.message = f"🏆 Winner: {info['winner']}" if info.get("winner") else "🤝 It's a draw."
        return

    env.current_player_index = (env.current_player_index + 1) % 3
    st.session_state.await_agent_moves = True

# --- Agent Handler ---
def play_agent_turn():
    while not env.done and players[env.current_player_index] != 'X':
        player = players[env.current_player_index]
        move = dqn_move()
        if move is None:
            break
        row, col = move
        state, reward, done, info = env.step((row, col))
        st.session_state.turn_count += 1
        st.session_state.last_action = f"{player} moved to ({row}, {col})"

        if not st.session_state.power_used[player]:
            st.session_state.power_used[player] = True
            move2 = dqn_move()
            if move2:
                row2, col2 = move2
                state, reward, done, info = env.step((row2, col2))
                st.session_state.turn_count += 1
                st.session_state.last_action += f" and used POWER at ({row2}, {col2})"

        if done:
            st.session_state.done = True
            st.session_state.message = f"🏆 Winner: {info['winner']}" if info.get("winner") else "🤝 It's a draw."
            return

        env.current_player_index = (env.current_player_index + 1) % 3

# Trigger agent moves after human turn
if st.session_state.await_agent_moves and not st.session_state.done:
    st.session_state.await_agent_moves = False
    play_agent_turn()
    st.rerun()

# --- UI ---
st.title("🎮 Human vs 2 Agents - Tic Tac Toe with Powers")

if players[env.current_player_index] == 'X' and not st.session_state.awaiting_second_power:
    st.session_state.use_power = st.checkbox("Use Power Move (play twice)", value=False, disabled=st.session_state.power_used['X'])
    st.session_state.use_swap = st.checkbox("Use Swap Move (undo your last move)", value=False, disabled=st.session_state.swap_used['X'])

st.markdown("### Game Board")
for r in range(5):
    cols = st.columns(5)
    for c in range(5):
        key = f"{r}-{c}-{st.session_state.turn_count}"
        if env.board[r][c] == ' ' and players[env.current_player_index] == 'X':
            if cols[c].button(" ", key=key):
                play_human_move(r, c)
                st.rerun()
        else:
            cols[c].markdown(get_cell_display(env.board[r][c]), unsafe_allow_html=True)

# --- Status ---
st.markdown("---")
st.markdown(f"**{st.session_state.last_action}**")
if st.session_state.done:
    st.success(st.session_state.message)
else:
    st.info(st.session_state.message)
    st.markdown(f"**Next Player: {players[env.current_player_index]}**")
