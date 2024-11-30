import chess
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import chess.engine
import tkinter as tk
from tkinter import messagebox
from tkinter import simpledialog, messagebox

engine_path = "stockfish\stockfish-windows-x86-64-avx2.exe"  
def fen_to_tensor(fen):
    board = chess.Board(fen)
    tensor = np.zeros((12,8,8), dtype=np.float32)
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

    
def load_model(model, optimizer, filename):
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

# Загрузка обученной модели
model = ChessPositionEvaluator().cuda()
optimizer = optim.Adam(model.parameters(), lr=0.001)
load_model(model, optimizer, "mtcs.pth")

class ChessGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("By Rerowros")
        self.board = chess.Board()
        self.canvas = tk.Canvas(root, width=400, height=400)
        self.canvas.pack()
        self.draw_board()
        self.canvas.bind("<Button-1>", self.on_click)
        self.selected_square = None
        self.engine = chess.engine.SimpleEngine.popen_uci(engine_path)

    def draw_board(self):
        self.canvas.delete("all")
        colors = ["#f0d9b5", "#b58863"]
        for row in range(8):
            for col in range(8):
                color = colors[(row + col) % 2]
                x1 = col * 50
                y1 = row * 50
                x2 = x1 + 50
                y2 = y1 + 50
                self.canvas.create_rectangle(x1, y1, x2, y2, fill=color)
                piece = self.board.piece_at(chess.square(col, 7 - row))
                if piece:
                    self.canvas.create_text(x1 + 25, y1 + 25, text=piece.symbol(), font=("Arial", 24))

    def on_click(self, event):
        col = event.x // 50
        row = 7 - (event.y // 50)
        square = chess.square(col, row)
        print(f"Clicked on: col={col}, row={row}, square={square}")

        if self.selected_square is None:
            # Проверяем, что выбрана своя фигура
            piece = self.board.piece_at(square)
            if piece and piece.color == self.board.turn:
                self.selected_square = square
        else:
            try:
                # Создаем базовый ход
                move = chess.Move(self.selected_square, square)
                
                # Проверяем, является ли это ходом пешки на последнюю линию
                if (self.board.piece_at(self.selected_square).piece_type == chess.PAWN and 
                    (chess.square_rank(square) == 0 or chess.square_rank(square) == 7)):
                    # Добавляем превращение в ферзя к ходу
                    move = chess.Move(self.selected_square, square, chess.QUEEN)

                # Проверяем легальность хода
                if move in self.board.legal_moves:
                    self.board.push(move)
                    self.draw_board()
                    self.selected_square = None
                    self.root.after(1000, self.ai_move)
                else:
                    self.selected_square = None
            except Exception as e:
                print(f"Error: {e}")
                self.selected_square = None
        

    def choose_promotion(self):
        pieces = {'Queen': chess.QUEEN, 'Rook': chess.ROOK, 'Bishop': chess.BISHOP, 'Knight': chess.KNIGHT}
        choice = tk.simpledialog.askstring("Promotion", "Choose promotion piece (Queen, Rook, Bishop, Knight):")
        return pieces.get(choice, chess.QUEEN)  # По умолчанию выбираем ферзя, если ввод некорректен

    def ai_move(self):
        state = torch.tensor(fen_to_tensor(self.board.fen()), dtype=torch.float32).unsqueeze(0).cuda()
        model.eval()
        with torch.no_grad():
            move_scores = []
            legal_moves = list(self.board.legal_moves)
            for move in legal_moves:
                self.board.push(move)
                next_state = torch.tensor(fen_to_tensor(self.board.fen()), dtype=torch.float32).unsqueeze(0).cuda()
                score = model(next_state).item()
                move_scores.append((score, move))
                self.board.pop()
            best_move = max(move_scores, key=lambda x: x[0])[1]
            self.board.push(best_move)
            self.draw_board()
            if self.board.is_game_over():
                messagebox.showinfo("Game Over", self.board.result())

if __name__ == "__main__":
    root = tk.Tk()
    gui = ChessGUI(root)
    root.mainloop()