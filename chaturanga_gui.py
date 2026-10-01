import os
import random
import threading
import tkinter as tk
import chess
import chess.pgn
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image, ImageTk, ImageDraw, ImageFont

from tokenizers import pre_tokenizers


class ChessLLMGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Chaturanga Nano")
        self.root.geometry("1000x700")
        self.root.minsize(840, 620)
        self.root.configure(bg="#1e1e1e")

        self.board = chess.Board()
        self.player_color = chess.WHITE
        self.selected_square = None
        self.legal_destinations = []
        self.last_move = None
        self.flipped = False
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.is_model_thinking = False
        self.is_model_loaded = False

        self.board_size = 560
        self.sq_size = self.board_size // 8

        self.color_light_sq = "#eeeed2"
        self.color_dark_sq = "#769656"
        self.color_selected = "#baca44"
        self.color_legal = "#f7f769"
        self.color_last_move = "#cdd26a"

        self.piece_symbols_text = {
            'P': ('♙', '#FFFFFF', '#111111'),
            'N': ('♘', '#FFFFFF', '#111111'),
            'B': ('♗', '#FFFFFF', '#111111'),
            'R': ('♖', '#FFFFFF', '#111111'),
            'Q': ('♕', '#FFFFFF', '#111111'),
            'K': ('♔', '#FFFFFF', '#111111'),
            'p': ('♟', '#1a1a1a', '#e6c687'),
            'n': ('♞', '#1a1a1a', '#e6c687'),
            'b': ('♝', '#1a1a1a', '#e6c687'),
            'r': ('♜', '#1a1a1a', '#e6c687'),
            'q': ('♛', '#1a1a1a', '#e6c687'),
            'k': ('♚', '#1a1a1a', '#e6c687'),
        }

        self.has_valid_pil_fonts = False
        self.piece_images = self.create_piece_images(self.sq_size)

        self.setup_ui()
        self.draw_board()

        self.update_status("Loading board... Initializing Chaturanga-Nano model in background...")
        threading.Thread(target=self.load_model, daemon=True).start()

    def create_piece_images(self, sq_size):
        images = {}
        font_size = int(sq_size * 0.75)
        font_paths = [
            "C:\\Windows\\Fonts\\seguisym.ttf",
            "C:\\Windows\\Fonts\\arial.ttf",
            "C:\\Windows\\Fonts\\segoeui.ttf",
        ]
        font_large = None
        scale = 2
        w, h = sq_size * scale, sq_size * scale
        f_size = int(font_size * scale)

        for path in font_paths:
            if os.path.exists(path):
                font_large = ImageFont.truetype(path, f_size)
                self.has_valid_pil_fonts = True
                break

        if not font_large:
            return {}

        pieces = {
            'K': ('♔', '#FFFFFF', '#2b2b2b'),
            'Q': ('♕', '#FFFFFF', '#2b2b2b'),
            'R': ('♖', '#FFFFFF', '#2b2b2b'),
            'B': ('♗', '#FFFFFF', '#2b2b2b'),
            'N': ('♘', '#FFFFFF', '#2b2b2b'),
            'P': ('♙', '#FFFFFF', '#2b2b2b'),
            'k': ('♚', '#1c1c1c', '#e6c687'),
            'q': ('♛', '#1c1c1c', '#e6c687'),
            'r': ('♜', '#1c1c1c', '#e6c687'),
            'b': ('♝', '#1c1c1c', '#e6c687'),
            'n': ('♞', '#1c1c1c', '#e6c687'),
            'p': ('♟', '#1c1c1c', '#e6c687'),
        }

        for code, (char, fill_col, stroke_col) in pieces.items():
            img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
            draw = ImageDraw.Draw(img)

            bbox = draw.textbbox((0, 0), char, font=font_large)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
            x = (w - text_w) // 2 - bbox[0]
            y = (h - text_h) // 2 - bbox[1]

            draw.text((x, y), char, font=font_large, fill=fill_col, stroke_width=scale * 2, stroke_fill=stroke_col)
            images[code] = ImageTk.PhotoImage(img.resize((sq_size, sq_size), Image.LANCZOS))

        return images

    def setup_ui(self):
        main_frame = tk.Frame(self.root, bg="#1e1e1e")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)

        board_frame = tk.Frame(main_frame, bg="#2d2d2d", bd=2, relief=tk.RAISED)
        board_frame.pack(side=tk.LEFT, padx=(0, 15), pady=0)

        self.canvas = tk.Canvas(
            board_frame,
            width=self.board_size,
            height=self.board_size,
            bg="#2d2d2d",
            highlightthickness=0
        )
        self.canvas.pack(padx=5, pady=5)
        self.canvas.bind("<Button-1>", self.on_square_click)

        right_frame = tk.Frame(main_frame, bg="#2d2d2d", bd=2, relief=tk.RAISED)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        title_label = tk.Label(
            right_frame,
            text="Chaturanga Nano",
            font=("Helvetica", 18, "bold"),
            bg="#2d2d2d",
            fg="#ffffff"
        )
        title_label.pack(pady=(15, 2))

        self.model_status_label = tk.Label(
            right_frame,
            text="Model: Loading...",
            font=("Helvetica", 10, "italic"),
            bg="#2d2d2d",
            fg="#ffcc00"
        )
        self.model_status_label.pack(pady=(0, 8))

        color_frame = tk.LabelFrame(
            right_frame,
            text=" Play As ",
            font=("Helvetica", 10, "bold"),
            bg="#2d2d2d",
            fg="#cccccc",
            bd=1,
            relief=tk.SOLID
        )
        color_frame.pack(fill=tk.X, padx=15, pady=(0, 10))

        self.btn_play_white = tk.Button(
            color_frame,
            text="♔ White (Player)",
            font=("Helvetica", 9, "bold"),
            bg="#4682b4",
            fg="#ffffff",
            activebackground="#5a9bd4",
            activeforeground="#ffffff",
            bd=0,
            padx=10,
            pady=5,
            command=lambda: self.set_player_color(chess.WHITE)
        )
        self.btn_play_white.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(8, 4), pady=8)

        self.btn_play_black = tk.Button(
            color_frame,
            text="♚ Black (Player)",
            font=("Helvetica", 9, "bold"),
            bg="#3a3a3a",
            fg="#aaaaaa",
            activebackground="#4a4a4a",
            activeforeground="#ffffff",
            bd=0,
            padx=10,
            pady=5,
            command=lambda: self.set_player_color(chess.BLACK)
        )
        self.btn_play_black.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(4, 8), pady=8)

        history_title = tk.Label(
            right_frame,
            text="Move History (PGN)",
            font=("Helvetica", 11, "bold"),
            bg="#2d2d2d",
            fg="#cccccc"
        )
        history_title.pack(anchor="w", padx=15, pady=(5, 2))

        history_frame = tk.Frame(right_frame, bg="#2d2d2d")
        history_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=(0, 10))

        self.history_text = tk.Text(
            history_frame,
            wrap=tk.WORD,
            font=("Consolas", 10),
            bg="#181818",
            fg="#00ffcc",
            bd=1,
            relief=tk.SUNKEN,
            height=10
        )
        scrollbar = tk.Scrollbar(history_frame, command=self.history_text.yview)
        self.history_text.configure(yscrollcommand=scrollbar.set)

        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.history_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.status_label = tk.Label(
            right_frame,
            text="White to move.",
            font=("Helvetica", 10, "bold"),
            bg="#3a3a3a",
            fg="#ffffff",
            anchor="w",
            padx=10,
            pady=8,
            wraplength=320,
            justify=tk.LEFT
        )
        self.status_label.pack(fill=tk.X, padx=15, pady=(0, 10))

        btn_frame = tk.Frame(right_frame, bg="#2d2d2d")
        btn_frame.pack(fill=tk.X, padx=15, pady=(0, 15))

        tk.Button(
            btn_frame,
            text="New Game",
            font=("Helvetica", 10, "bold"),
            bg="#4682b4",
            fg="#ffffff",
            activebackground="#5a9bd4",
            activeforeground="#ffffff",
            bd=0,
            padx=10,
            pady=6,
            command=self.reset_game
        ).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 5))

        tk.Button(
            btn_frame,
            text="Undo",
            font=("Helvetica", 10, "bold"),
            bg="#555555",
            fg="#ffffff",
            activebackground="#666666",
            activeforeground="#ffffff",
            bd=0,
            padx=10,
            pady=6,
            command=self.undo_move
        ).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(5, 5))

        tk.Button(
            btn_frame,
            text="Flip Board",
            font=("Helvetica", 10, "bold"),
            bg="#555555",
            fg="#ffffff",
            activebackground="#666666",
            activeforeground="#ffffff",
            bd=0,
            padx=10,
            pady=6,
            command=self.flip_board
        ).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(5, 0))

    def set_player_color(self, color):
        if self.is_model_thinking:
            return
        self.player_color = color
        if color == chess.WHITE:
            self.btn_play_white.config(bg="#4682b4", fg="#ffffff")
            self.btn_play_black.config(bg="#3a3a3a", fg="#aaaaaa")
            self.flipped = False
        else:
            self.btn_play_black.config(bg="#4682b4", fg="#ffffff")
            self.btn_play_white.config(bg="#3a3a3a", fg="#aaaaaa")
            self.flipped = True

        self.reset_game()

    def update_model_status(self, success, message):
        color = "#00ff88" if success else "#ff5555"
        self.model_status_label.config(text=f"Model: {message}", fg=color)

    def update_status(self, text):
        self.status_label.config(text=text)

    def load_model(self):
        # Load Chaturanga-Nano model from HuggingFace with local fallback
        model_id = "DireDreadlord/Chaturanga-Nano"
        if not os.path.exists(model_id) and os.path.exists("./suprachess-mini"):
            model_id = "./suprachess-mini"

        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        if pre_tokenizers is not None and hasattr(self.tokenizer, "_tokenizer") and hasattr(self.tokenizer._tokenizer, "pre_tokenizer"):
            self.tokenizer._tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

        dtype = torch.bfloat16 if torch.cuda.is_available() and self.device == "cuda" else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=dtype,
            device_map="auto" if torch.cuda.is_available() and self.device == "cuda" else None,
            trust_remote_code=True
        )
        if self.device == "cuda" and not hasattr(self.model, "device"):
            self.model.to("cuda")
        self.model.eval()

        self.is_model_loaded = True
        self.root.after(0, lambda: self.update_model_status(True, f"Loaded on {self.device.upper()}"))

        if self.board.turn != self.player_color:
            self.root.after(0, lambda: self.make_model_move())
        else:
            self.root.after(0, lambda: self.update_status(f"Board and Model loaded. Your turn ({'White' if self.player_color == chess.WHITE else 'Black'})."))

    def draw_board(self):
        # Render 8x8 chessboard and pieces
        self.canvas.delete("all")

        for rank in range(8):
            for file in range(8):
                display_file = 7 - file if self.flipped else file
                display_rank = rank if self.flipped else 7 - rank

                sq = chess.square(display_file, display_rank)

                x1, y1 = file * self.sq_size, rank * self.sq_size
                x2, y2 = x1 + self.sq_size, y1 + self.sq_size

                is_light = (display_file + display_rank) % 2 != 0
                color = self.color_light_sq if is_light else self.color_dark_sq

                if self.last_move and (sq == self.last_move.from_square or sq == self.last_move.to_square):
                    color = self.color_last_move

                if sq == self.selected_square:
                    color = self.color_selected

                self.canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline="")

                if sq in self.legal_destinations:
                    r = 8
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    if self.board.piece_at(sq):
                        self.canvas.create_oval(cx - 24, cy - 24, cx + 24, cy + 24, outline="#ff4444", width=3)
                    else:
                        self.canvas.create_oval(cx - r, cy - r, cx + r, cy + r, fill="#3a86ff", outline="")

                piece = self.board.piece_at(sq)
                if piece:
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    img_photo = self.piece_images.get(piece.symbol())

                    if img_photo and self.has_valid_pil_fonts:
                        self.canvas.create_image(x1, y1, anchor="nw", image=img_photo)
                    else:
                        symbol, fill_col, outline_col = self.piece_symbols_text.get(piece.symbol(), ('?', '#ffffff', '#000000'))
                        self.canvas.create_text(cx + 1, cy + 1, text=symbol, font=("Segoe UI Symbol", 36, "bold"), fill=outline_col)
                        self.canvas.create_text(cx, cy, text=symbol, font=("Segoe UI Symbol", 36, "bold"), fill=fill_col)

                if file == 0:
                    lbl_color = self.color_dark_sq if is_light else self.color_light_sq
                    self.canvas.create_text(x1 + 8, y1 + 10, text=str(display_rank + 1), font=("Helvetica", 9, "bold"), fill=lbl_color)

                if rank == 7:
                    lbl_color = self.color_dark_sq if is_light else self.color_light_sq
                    self.canvas.create_text(x2 - 10, y2 - 10, text=chess.FILE_NAMES[display_file], font=("Helvetica", 9, "bold"), fill=lbl_color)

    def on_square_click(self, event):
        # Handle user piece selection and move execution
        if self.is_model_thinking or self.board.turn != self.player_color:
            return

        file_idx = event.x // self.sq_size
        rank_idx = event.y // self.sq_size

        if self.flipped:
            file_idx = 7 - file_idx
        else:
            rank_idx = 7 - rank_idx

        if not (0 <= file_idx <= 7 and 0 <= rank_idx <= 7):
            return

        sq = chess.square(file_idx, rank_idx)

        if self.selected_square is not None and sq in self.legal_destinations:
            move = chess.Move(self.selected_square, sq)
            piece = self.board.piece_at(self.selected_square)
            if piece and piece.piece_type == chess.PAWN:
                if (piece.color == chess.WHITE and rank_idx == 7) or (piece.color == chess.BLACK and rank_idx == 0):
                    move = chess.Move(self.selected_square, sq, promotion=chess.QUEEN)

            if move in self.board.legal_moves:
                san = self.board.san(move)
                self.board.push(move)
                self.last_move = move
                self.selected_square = None
                self.legal_destinations = []
                self.draw_board()
                self.update_move_history()

                player_name = "White" if self.player_color == chess.WHITE else "Black"
                model_name = "Black" if self.player_color == chess.WHITE else "White"

                if self.board.is_game_over():
                    self.update_status(f"{player_name} played: {san} | Game Over: {self.get_game_result()}")
                else:
                    self.update_status(f"{player_name} played: {san}. Chaturanga ({model_name}) is thinking...")
                    self.is_model_thinking = True
                    self.make_model_move()
                return

        piece = self.board.piece_at(sq)
        if piece and piece.color == self.board.turn:
            self.selected_square = sq
            self.legal_destinations = [m.to_square for m in self.board.legal_moves if m.from_square == sq]
        else:
            self.selected_square = None
            self.legal_destinations = []

        self.draw_board()

    def build_pgn_prompt(self):
        temp_board = chess.Board()
        pgn_parts = []
        moves = self.board.move_stack
        for i, move in enumerate(moves):
            san_move = temp_board.san(move)
            temp_board.push(move)
            move_number = (i // 2) + 1
            if i % 2 == 0:
                pgn_parts.append(f"{move_number}. {san_move}")
            else:
                pgn_parts.append(f"{san_move}")

        if len(moves) % 2 == 1:
            return " ".join(pgn_parts) + " "
        else:
            move_number = (len(moves) // 2) + 1
            return (" ".join(pgn_parts) + f" {move_number}. ") if pgn_parts else "1. "

    def make_model_move(self):
        if not self.is_model_loaded or self.model is None or self.board.is_game_over():
            self.is_model_thinking = False
            return

        model_name = "White" if self.player_color == chess.BLACK else "Black"
        self.update_status(f"Chaturanga ({model_name}) is thinking...")
        self.is_model_thinking = True
        threading.Thread(target=self._generate_and_apply_model_move, daemon=True).start()

    def _generate_and_apply_model_move(self):
        chosen_move, chosen_san, is_valid = self.select_model_legal_move()
        self.root.after(0, lambda: self._apply_model_move_ui(chosen_move, chosen_san, is_valid))

    def select_model_legal_move(self):
        # Score legal moves using model likelihood
        prompt = self.build_pgn_prompt()
        legal_moves = list(self.board.legal_moves)
        if not legal_moves:
            return None, None, False

        prompt_ids = self.tokenizer.encode(prompt)
        prompt_len = len(prompt_ids)

        cand_ids = []
        max_len = 0
        valid_moves = []

        for move in legal_moves:
            san = self.board.san(move)
            full_ids = self.tokenizer.encode(prompt + san)
            if len(full_ids) > prompt_len:
                cand_ids.append(full_ids)
                valid_moves.append((san, move))
                if len(full_ids) > max_len:
                    max_len = len(full_ids)

        if not cand_ids:
            fallback = legal_moves[0]
            return fallback, self.board.san(fallback), False

        pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
        batch = [ids + [pad_id] * (max_len - len(ids)) for ids in cand_ids]
        input_tensor = torch.tensor(batch)
        if self.device == "cuda":
            input_tensor = input_tensor.to("cuda")

        with torch.no_grad():
            outputs = self.model(input_tensor)
            logits = outputs.logits

        move_scores = []
        for idx, (san, move) in enumerate(valid_moves):
            full_ids = cand_ids[idx]
            target_ids = torch.tensor(full_ids[prompt_len:])
            if self.device == "cuda":
                target_ids = target_ids.to("cuda")

            log_probs = torch.log_softmax(logits[idx, prompt_len - 1:len(full_ids) - 1], dim=-1)
            token_scores = log_probs[torch.arange(len(target_ids)), target_ids]
            avg_score = token_scores.sum().item()
            move_scores.append((avg_score, san, move))

        move_scores.sort(key=lambda x: x[0], reverse=True)

        top_candidates = move_scores[:min(5, len(move_scores))]
        scores_tensor = torch.tensor([m[0] for m in top_candidates])
        probs = torch.softmax(scores_tensor / 0.7, dim=0).tolist()
        chosen_idx = random.choices(range(len(top_candidates)), weights=probs, k=1)[0]

        best_score, best_san, best_move = top_candidates[chosen_idx]
        return best_move, best_san, True

    def _apply_model_move_ui(self, move, san, is_valid):
        if move is not None and move in self.board.legal_moves:
            model_name = "White" if self.player_color == chess.BLACK else "Black"
            player_name = "Black" if self.player_color == chess.BLACK else "White"

            self.board.push(move)
            self.last_move = move
            self.selected_square = None
            self.legal_destinations = []
            self.draw_board()
            self.update_move_history()

            status_msg = f"{model_name} (Chaturanga) played: {san}. Your turn ({player_name})."
            if self.board.is_game_over():
                status_msg += f" | Game Over: {self.get_game_result()}"

            self.update_status(status_msg)
        else:
            self.update_status("Model failed to make a legal move.")

        self.is_model_thinking = False

    def update_move_history(self):
        pgn_text = self.build_pgn_prompt().strip()
        self.history_text.config(state=tk.NORMAL)
        self.history_text.delete("1.0", tk.END)
        self.history_text.insert(tk.END, pgn_text)
        self.history_text.config(state=tk.DISABLED)
        self.history_text.see(tk.END)

    def get_game_result(self):
        if self.board.is_checkmate():
            winner = "Black" if self.board.turn == chess.WHITE else "White"
            return f"Checkmate! {winner} wins."
        elif self.board.is_stalemate():
            return "Stalemate!"
        elif self.board.is_insufficient_material():
            return "Draw (Insufficient material)"
        elif self.board.can_claim_draw():
            return "Draw claimable"
        return "Game Over"

    def reset_game(self):
        if self.is_model_thinking:
            return
        self.board.reset()
        self.selected_square = None
        self.legal_destinations = []
        self.last_move = None
        self.draw_board()
        self.update_move_history()

        player_name = "White" if self.player_color == chess.WHITE else "Black"
        if self.board.turn != self.player_color:
            self.update_status(f"New game started. Player is {player_name}. Chaturanga (White) is thinking...")
            if self.is_model_loaded:
                self.make_model_move()
        else:
            self.update_status(f"New game started. Player is {player_name}. Your turn.")

    def undo_move(self):
        if self.is_model_thinking:
            return

        if self.board.turn == self.player_color:
            if len(self.board.move_stack) >= 2:
                self.board.pop()
                self.board.pop()
            elif len(self.board.move_stack) == 1:
                self.board.pop()
        else:
            if len(self.board.move_stack) >= 1:
                self.board.pop()

        self.last_move = self.board.peek() if self.board.move_stack else None
        self.selected_square = None
        self.legal_destinations = []
        self.draw_board()
        self.update_move_history()
        player_name = "White" if self.player_color == chess.WHITE else "Black"
        self.update_status(f"Moves undone. Your turn ({player_name}).")

    def flip_board(self):
        self.flipped = not self.flipped
        self.draw_board()


def main():
    root = tk.Tk()
    app = ChessLLMGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
