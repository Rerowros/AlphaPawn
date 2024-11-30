import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import random
import chess
import chess.engine
import numpy as np
import concurrent.futures
import json
from collections import deque
import math
from copy import deepcopy
from tqdm import tqdm


class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
    
    def push(self, experience, priority):
        self.memory.append(experience)
        self.priorities.append(priority)
    
    def sample(self, batch_size):
        priorities = np.array(self.priorities, dtype=np.float32)
        probabilities = priorities / priorities.sum()
        indices = np.random.choice(len(self.memory), batch_size, p=probabilities)
        batch = [self.memory[idx] for idx in indices]
        return batch, indices
    
    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority
    
    def __len__(self):
        return len(self.memory)
    
memory_capacity = 50000  # Размер буфера памяти
replay_memory = ReplayMemory(memory_capacity)

def flip_board(fen):
    board = chess.Board(fen)
    board.apply_mirror()
    return board.fen()

def fen_to_tensor(fen, flip_probability=0.4):
    # Случайно переворачиваем доску и меняем стороны местами
    if random.random() < flip_probability:
        fen = flip_board(fen)
    
    board = chess.Board(fen)
    tensor = np.zeros((12, 8, 8), dtype=np.float32)
    piece_map = board.piece_map()

    for square, piece in piece_map.items():
        x, y = divmod(square, 8)
        channel = piece_to_channel(piece)
        tensor[channel, x, y] = 1
    return tensor


# Преобразование типа фигуры в канал
def piece_to_channel(piece):
    piece_types = {'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5}
    base_channel = piece_types[piece.symbol().upper()]
    return base_channel + (6 if piece.color == chess.BLACK else 0)

class ChessPositionEvaluator(nn.Module):
    def __init__(self):
        super(ChessPositionEvaluator, self).__init__()
        
        # Свёрточные блоки с BatchNorm и Attention
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(12, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        
        self.conv_block4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        
        # Attention layers
        self.attention1 = nn.MultiheadAttention(64, num_heads=4, batch_first=True)
        self.attention2 = nn.MultiheadAttention(128, num_heads=4, batch_first=True)
        self.attention3 = nn.MultiheadAttention(256, num_heads=4, batch_first=True)
        self.attention4 = nn.MultiheadAttention(512, num_heads=4, batch_first=True)
        
        # Полносвязные слои
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 8 * 8, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

    def apply_attention(self, x, attention_layer):
        b, c, h, w = x.size()
        # Преобразуем для attention (batch, seq_len, features)
        x_flat = x.view(b, c, -1).permute(0, 2, 1)
        x_att, _ = attention_layer(x_flat, x_flat, x_flat)
        # Возвращаем к исходной форме
        return x_att.permute(0, 2, 1).view(b, c, h, w)

    def forward(self, x):
        # Применяем блоки с attention
        x = self.conv_block1(x)
        x = self.apply_attention(x, self.attention1)
        
        x = self.conv_block2(x)
        x = self.apply_attention(x, self.attention2)
        
        x = self.conv_block3(x)
        x = self.apply_attention(x, self.attention3)
        
        x = self.conv_block4(x)
        x = self.apply_attention(x, self.attention4)
        
        x = self.fc_layers(x)
        return x

num_epochs = 100
max_moves = 200
gamma = 0.9 # Коэффициент дисконтирования


def train_from_replay(model, optimizer, replay_memory, batch_size=64):
    if len(replay_memory) < batch_size:
        return

    batch, indices = replay_memory.sample(batch_size)
    states, actions, rewards, next_states, dones = zip(*batch)

    l2_lambda = 0.01
    l2_norm = sum(p.pow(2.0).sum() for p in model.parameters())

    states = torch.cat(states).cuda()
    next_states = torch.cat(next_states).cuda()
    rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).cuda()

    model.train()
    predictions = model(states)
    with torch.no_grad():
        next_predictions = model(next_states)
        targets = rewards + gamma * next_predictions * (1 - torch.tensor(dones, dtype=torch.float32).unsqueeze(1).cuda())

    losses = nn.MSELoss(reduction='none')(predictions, targets)
    loss = losses.mean() + l2_lambda * l2_norm
    
    # Обновляем приоритеты
    td_errors = losses.detach().cpu().numpy().flatten()
    new_priorities = np.abs(td_errors) + 1e-6
    replay_memory.update_priorities(indices, new_priorities)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    
    return loss.item()


engine_path = "stockfish\\stockfish-windows-x86-64-avx2.exe"  

def get_stockfish_evaluation(board, engine_path, time_limit=1.5):
    with chess.engine.SimpleEngine.popen_uci(engine_path) as engine:
        info = engine.analyse(board, chess.engine.Limit(time=time_limit))
        score = info["score"].relative
        if score.is_mate():
            return f"Мат в {score.mate()}"
        else:
            return score.score(mate_score=10000)

class MCTSNode:
    def __init__(self, board, parent=None, move=None, prior_p = 0):
        self.board = deepcopy(board)
        self.parent = parent
        self.move = move
        self.children = []
        self.Q = 0  # Ожидаемая ценность
        self.N = 0  # Количество посещений
        self.P = prior_p  # Априорная вероятность из политики
        self.untried_moves = list(board.legal_moves)
        
    def puct_value(self, c_param=5.0):
            if self.N == 0:
                return float('inf')
            
            # Формула PUCT
            Q_value = self.Q / self.N if self.N > 0 else 0
            exploration = c_param * self.P * math.sqrt(self.parent.N) / (1 + self.N)
            return Q_value + exploration

    def select_child(self):
        return max(self.children, key=lambda x: x.puct_value())
        #print(f"Выбран дочерний узел: ход {selected.move} с UCT значением {selected.uct_value()}")
        #return selected
    
    def expand(self):
        if not self.untried_moves:
            return None
        
        move = self.untried_moves.pop()
        new_board = deepcopy(self.board)
        new_board.push(move)
        
        # Получаем априорную вероятность из модели
        tensor = torch.tensor(fen_to_tensor(new_board.fen()), dtype=torch.float32).unsqueeze(0).cuda()
        with torch.no_grad():
            prior_p = torch.sigmoid(model(tensor)).item()
        
        child = MCTSNode(new_board, parent=self, move=move, prior_p=prior_p)
        self.children.append(child)
        return child

    
    def update(self, result):
        self.N += 1
        self.Q += (result - self.Q) / self.N  # Инкрементальное обновление среднего


def mcts_search(board, model, num_simulations=10, max_depth=50, current_move=0, engine_path="stockfish\\stockfish-windows-x86-64-avx2.exe"):
    root = MCTSNode(board)
    
    def run_simulation(_):
        node = root
        
        # Selection
        while node.untried_moves == [] and node.children != []:
            node = node.select_child()
        
        # Expansion
        if node.untried_moves != []:
            node = node.expand()
            if node is None:
                return
        
        # Simulation
        sim_board = deepcopy(node.board)
        positions = []
        current_max_depth = max_depth
        if len(list(sim_board.piece_map())) < 10:  # если осталось мало фигур
            current_max_depth = 100
            if isinstance(stockfish_score, str) or stockfish_score < -100:  
                current_max_depth = 150

        while not sim_board.is_game_over() and len(sim_board.move_stack) < current_max_depth:
            positions.append(fen_to_tensor(sim_board.fen()))
            legal_moves = list(sim_board.legal_moves)
            next_positions = []
            for move in legal_moves:
                sim_board.push(move)
                next_positions.append(fen_to_tensor(sim_board.fen()))
                sim_board.pop()
            
            # Батчевое оценивание позиций
            next_positions_array = np.array(next_positions)
            tensor_batch = torch.tensor(next_positions_array, dtype=torch.float32).cuda()
            model.eval()
            with torch.no_grad():
                scores = model(tensor_batch).cpu().numpy()
            
            # Добавляем небольшое случайное значение для исследования
            scores += np.random.normal(0, 0.3, size=scores.shape)
            best_move_index = np.argmax(scores)
            best_move = legal_moves[best_move_index]
            sim_board.push(best_move)
        
        # Оценка конечной позиции
        if sim_board.is_game_over():
            result = 1 if sim_board.result() == "1-0" else 0
        else:
            eval_tensor = torch.tensor(fen_to_tensor(sim_board.fen()), dtype=torch.float32).unsqueeze(0).cuda()
            with torch.no_grad():
                result = (model(eval_tensor).item() + 1) / 2  # нормализация к [0,1]
                
        # Backpropagation
        local_node = node
        while local_node:
            local_node.update(result if local_node.board.turn == chess.WHITE else 1 - result)
            local_node = local_node.parent
            
    # Увеличиваем количество симуляций в эндшпиле
    if len(list(board.piece_map())) < 10:  # если осталось мало фигур
        num_simulations *= 3

    # Получаем оценку от Stockfish
    stockfish_score = get_stockfish_evaluation(board, engine_path)
    if isinstance(stockfish_score, str) or stockfish_score < -1:  # Порог для увеличения симуляций и глубины
        num_simulations *= 3
        max_depth += 40  
        if isinstance(stockfish_score, str) or stockfish_score < -500:  # Порог для увеличения симуляций и глубины
            num_simulations *= 5
            max_depth += 70

    # Уменьшаем количество симуляций для первых трех ходов
    if current_move < 3:
        num_simulations = max(5, num_simulations // 2)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        list(tqdm(executor.map(run_simulation, range(num_simulations)), total=num_simulations, desc="Симуляции"))
    
    # Проверка наличия дочерних узлов перед выбором лучшего хода
    if not root.children:
        raise ValueError("Корневой узел не имеет дочерних узлов после симуляций")
    
    # Выбираем ход с наибольшим числом посещений
    best_move = max(root.children, key=lambda x: x.N).move
    print(f"Лучший ход выбран: {best_move} с посещениями {max(root.children, key=lambda x: x.N).N}")

    print(f"Оценка Stockfish: {stockfish_score}")
    current_move += 1
    return best_move

def simulate_game(model, replay_memory, max_moves=max_moves):
    board = chess.Board()
    engine = chess.engine.SimpleEngine.popen_uci(engine_path)
    game_data = []
    total_rewards = 0

    for _ in range(max_moves):
        if board.is_game_over():
            break

        # Текущее состояние
        state = torch.tensor(fen_to_tensor(board.fen()), dtype=torch.float32).unsqueeze(0).cuda()
        
        model.eval()

        best_move = mcts_search(board, model)

        # Применение хода
        board.push(best_move)

        piece_values = {
            'p': 1, 'n': 3, 'b': 3, 'r': 5, 'q': 8, 'k': 0.1
        }
        # Коэффициенты штрафа за подставление фигур (множитель к базовой ценности)
        exposure_penalties = {
            'p': 0.2, 'n': 1.2, 'b': 1.2, 'r': 1.2, 'q': 1.2, 'k': 10.0
        }

        # Обновленная логика обработки взятий и угроз
        if board.is_capture(best_move):
            # Обработка взятий
            captured_piece = board.piece_at(best_move.to_square)
            if captured_piece:
                reward = piece_values[captured_piece.symbol().lower()]
            else:
                reward = 0
        else:
            reward = 0

        # Проверка угроз после хода
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece and piece.color == board.turn:
                attackers = board.attackers(not piece.color, square)
                if attackers:
                    reward -= piece_values[piece.symbol().lower()] * exposure_penalties[piece.symbol().lower()]

        # Награда за шах
        if board.is_check():
            reward += 10

        # Награда за мат
        if board.is_checkmate():
            reward += 100

        game_data.append({
            'fen': board.fen(),
            'move': best_move.uci(),
            'evaluation': reward
        })

        # Следующее состояние
        next_state = torch.tensor(fen_to_tensor(board.fen()), dtype=torch.float32).unsqueeze(0).cuda()

        # Сохранение перехода в память с приоритетом
        experience = (state, best_move, reward, next_state, board.is_game_over())
        priority = abs(reward) + 0.01  # Приоритет на основе абсолютного значения награды
        replay_memory.push(experience, priority)
        total_rewards += reward

        if board.is_game_over():
            break

    engine.quit()
    save_game_data('game_analysis.json', game_data, total_rewards)
    return board.result()

def save_game_data(filename, game_data, total_rewards=None):
    try:
        with open(filename, 'r') as f:
            all_data = json.load(f)
    except FileNotFoundError:
        all_data = []
    
    # Преобразуем данные в нужный формат
    game_info = {
        'moves': game_data if isinstance(game_data, list) else game_data.get('moves', []),
        'total_rewards': total_rewards if total_rewards is not None else game_data.get('total_rewards', 0)
    }
    
    all_data.append(game_info)

    with open(filename, 'w') as f:
        json.dump(all_data, f, indent=4)

def load_game_data(filename):
    try:
        with open(filename, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return []


def save_model(model, optimizer, filename):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, filename)

def load_model(model, optimizer, filename):
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


if __name__ == "__main__":
    # Инициализация модели и оптимизатора
    model = ChessPositionEvaluator().cuda()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)  # Увеличение weight decay
    
    # Загрузка модели, если она существует
    try:
        load_model(model, optimizer, "mtcs.pth")
        print("Модель успешно загружена.")
    except FileNotFoundError:
        print("Модель не найдена, начинаем с нуля.")

    try:
        game_data = load_game_data('game_analysis.json')
        print("Данные игр успешно загружены.")
    except FileNotFoundError:
        game_data = []
        print("Данные игр не найдены, начинаем с нуля.")

    
    num_games = 100
    batch_size = 128
    for game in range(num_games):
        result = simulate_game(model, replay_memory)
        train_from_replay(model, optimizer, replay_memory, batch_size)
        print(f"Игра {game+1}/{num_games}, Результат: {result}")

        # Сохранение данных игры
        game_data = {
            'game_number': game + 1,
            'result': result,
            'replay_memory': replay_memory.memory
        }
        save_game_data('game_analysis.json', game_data)

            #Смести вправо Сохранение модели каждые 10 игр
            #if (game + 1) % 10 == 0:
        save_model(model, optimizer, "mtcs.pth")
        print("Модель сохранена.")