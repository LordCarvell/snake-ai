# neural_snake.py
# Genetic AI trains a snake to eat apples using a neural network.
# Requirements: pip install numpy
# tkinter comes with Python. Linux: sudo apt install python3-tk
# Run: python3 neural_snake.py
#
# Controls (visual trainer)
#   Space       pause / resume
#   + / -       speed up / slow down
#   B           toggle watch-best
#   V           toggle vision rays
#   S           save now
#   R           restart training

# SECTION 1 - IMPORTS #

import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import random, math, time, os, json
from collections import deque

# SECTION 2 - COLOURS #

BG       = "#050a0e"
PANEL_BG = "#080f14"
BORDER   = "#0d2233"
GRID_COL = "#091520"
CYAN     = "#00ffe7"
CYAN_D   = "#006055"
RED      = "#ff3e6c"
RED_D    = "#5a1428"
GOLD     = "#ffe600"
DIM      = "#1a3a4a"
TEXT     = "#a0d8ef"
WHITE    = "#e0f7ff"
MID      = "#0d1f2d"

def lerp_hex(a, b, t):
    t = max(0.0, min(1.0, t))
    ar, ag, ab = int(a[1:3], 16), int(a[3:5], 16), int(a[5:7], 16)
    br, bg, bb = int(b[1:3], 16), int(b[3:5], 16), int(b[5:7], 16)
    return "#{:02x}{:02x}{:02x}".format(
        int(ar + (br - ar) * t),
        int(ag + (bg - ag) * t),
        int(ab + (bb - ab) * t),
    )

# SECTION 3 - CONFIG #

MODELS_DIR   = "snake_models"
HEADER_EVERY = 25   # reprint table header every N gens in headless mode

class Cfg:
    GRID              = 20
    CELL              = 22      # px per cell
    POPULATION        = 60
    HIDDEN_SIZE       = 20
    ACTIVATION        = "tanh"  # tanh | relu | sigmoid | leaky
    MUTATION_RATE     = 0.08
    MUTATION_STRENGTH = 0.35
    ELITE_PERCENT     = 0.20
    MOVE_LIMIT        = 220
    SPEED             = 10      # ticks/sec in visual mode (0 = max)
    WATCH_BEST        = True
    SHOW_RAYS         = False

    def to_dict(self):
        return {k: getattr(self, k) for k in (
            "GRID", "POPULATION", "HIDDEN_SIZE", "ACTIVATION",
            "MUTATION_RATE", "MUTATION_STRENGTH", "ELITE_PERCENT", "MOVE_LIMIT")}

    def apply_dict(self, d):
        for k, v in d.items():
            if hasattr(self, k):
                setattr(self, k, v)

CFG = Cfg()

# SECTION 4 - SAVE / LOAD #

def model_dir(name):
    return os.path.join(MODELS_DIR, name)

def save_model(name, snakes, gen, all_best, score_hist):
    d = model_dir(name)
    os.makedirs(d, exist_ok=True)
    best = max(snakes, key=lambda s: s.fit)
    np.savez(os.path.join(d, "weights.npz"),
             W1=best.net.W1, b1=best.net.b1,
             W2=best.net.W2, b2=best.net.b2)
    meta = {
        "name":       name,
        "gen":        gen,
        "all_best":   all_best,
        "score_hist": score_hist[-60:],
        "cfg":        CFG.to_dict(),
        "saved_at":   time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(os.path.join(d, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

def load_meta(name):
    path = os.path.join(model_dir(name), "meta.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)

def load_weights(name):
    path = os.path.join(model_dir(name), "weights.npz")
    if not os.path.exists(path):
        return None
    return np.load(path)

def list_models():
    if not os.path.exists(MODELS_DIR):
        return []
    out = []
    for name in sorted(os.listdir(MODELS_DIR)):
        meta = load_meta(name)
        if meta:
            out.append((name, meta))
    return out

# SECTION 5 - NEURAL NETWORK #

# Activation functions as plain numpy lambdas - avoids dict lookup overhead in forward pass
_ACTS = {
    "tanh":    np.tanh,
    "relu":    lambda x: np.maximum(0, x),
    "sigmoid": lambda x: 1 / (1 + np.exp(-np.clip(x, -500, 500))),
    "leaky":   lambda x: np.where(x > 0, x, 0.01 * x),
}

class Net:
    def __init__(self):
        ni, nh, no = 24, CFG.HIDDEN_SIZE, 4
        s1, s2 = math.sqrt(2 / ni), math.sqrt(2 / nh)
        self.W1  = np.random.randn(ni, nh) * s1
        self.b1  = np.zeros(nh)
        self.W2  = np.random.randn(nh, no) * s2
        self.b2  = np.zeros(no)
        self.h   = np.zeros(nh)
        self.o   = np.zeros(no)
        self.inp = np.zeros(ni, dtype=np.float32)  # reused every forward call
        self._act = _ACTS[CFG.ACTIVATION]

    def forward(self, x):
        # write into pre-allocated arrays to avoid allocation on hot path
        np.dot(x, self.W1, out=self.h); self.h += self.b1; self.h[:] = self._act(self.h)
        np.dot(self.h, self.W2, out=self.o); self.o += self.b2; np.tanh(self.o, out=self.o)
        return self.o

    def clone(self):
        c = Net.__new__(Net)
        c.W1  = self.W1.copy(); c.b1  = self.b1.copy()
        c.W2  = self.W2.copy(); c.b2  = self.b2.copy()
        c.h   = self.h.copy();  c.o   = self.o.copy()
        c.inp = self.inp.copy()
        c._act = self._act
        return c

    def mutate(self):
        r, s = CFG.MUTATION_RATE, CFG.MUTATION_STRENGTH
        for a in (self.W1, self.b1, self.W2, self.b2):
            a += (np.random.rand(*a.shape) < r) * np.random.randn(*a.shape) * s

    def crossover(self, other):
        c = self.clone()
        for sa, oa, ca in (
            (self.W1, other.W1, c.W1), (self.b1, other.b1, c.b1),
            (self.W2, other.W2, c.W2), (self.b2, other.b2, c.b2),
        ):
            m = np.random.rand(*sa.shape) < 0.5
            ca[:] = np.where(m, sa, oa)
        return c

    def load_weights(self, npz):
        self.W1 = npz["W1"].copy(); self.b1 = npz["b1"].copy()
        self.W2 = npz["W2"].copy(); self.b2 = npz["b2"].copy()
        self.h   = np.zeros(self.b1.shape)
        self.o   = np.zeros(self.b2.shape)
        self.inp = np.zeros(24, dtype=np.float32)
        self._act = _ACTS[CFG.ACTIVATION]

# SECTION 6 - SNAKE #

# Direction vectors: Up Right Down Left, then 4 diagonals
# Stored as plain tuples - faster than numpy array indexing in the tight inner loop
DIRS  = ((0, -1), (1, 0), (0, 1), (-1, 0))
DIAGS = ((1, -1), (1, 1), (-1, 1), (-1, -1))
ALL_DIRS = DIRS + DIAGS   # all 8, used for raycasting

# numpy version of just the 4 cardinal dirs, used for argmax -> direction lookup
DIRS_NP = np.array(DIRS, dtype=np.int8)

class Snake:
    def __init__(self, net=None):
        self.net = net or Net()
        self.reset()

    def reset(self):
        cx, cy = CFG.GRID // 2, CFG.GRID // 2
        self.body  = deque([(cx, cy), (cx - 1, cy), (cx - 2, cy)])
        self.bset  = set(self.body)
        self.dir   = (1, 0)
        self.score = 0
        self.moves = 0
        self.left  = CFG.MOVE_LIMIT
        self.alive = True
        self.fit   = 0
        # flat grid: 0=empty, 1=body, 2=food — updated incrementally, used in _sense
        g = CFG.GRID
        self._grid = np.zeros(g * g, dtype=np.int8)
        for bx, by in self.body:
            self._grid[by * g + bx] = 1
        self.food = self._spawn()  # _spawn sets grid cell to 2

    def _spawn(self):
        g = CFG.GRID
        while True:
            p = (random.randint(0, g - 1), random.randint(0, g - 1))
            if p not in self.bset:
                self._grid[p[1] * g + p[0]] = 2
                return p

    # Writes directly into the network's pre-allocated input buffer — no allocation.
    # Uses _grid (flat int8 array) for O(1) cell lookups instead of set.__contains__.
    def _sense(self):
        inp = self.net.inp   # write directly into pre-allocated buffer
        hx, hy = self.body[0]
        g  = CFG.GRID
        gr = self._grid
        i  = 0
        for dx, dy in ALL_DIRS:
            wd = fs = bs = 0.0
            x, y = hx + dx, hy + dy
            step = 1
            while 0 <= x < g and 0 <= y < g:
                cell = gr[y * g + x]
                if cell == 1 and bs == 0.0:
                    bs = 1.0 / step
                elif cell == 2 and fs == 0.0:
                    fs = 1.0
                wd = 1.0 / step
                x += dx; y += dy; step += 1
            inp[i]     = wd
            inp[i + 1] = fs
            inp[i + 2] = bs
            i += 3
        return inp

    def think(self):
        self._sense()   # writes into self.net.inp in-place
        idx = int(np.argmax(self.net.forward(self.net.inp)))
        nd  = DIRS[idx]
        if nd[0] != -self.dir[0] or nd[1] != -self.dir[1]:
            self.dir = nd

    def step(self):
        if not self.alive:
            return
        self.think()
        hx, hy = self.body[0]
        dx, dy = self.dir
        nx, ny = hx + dx, hy + dy
        self.moves += 1
        self.left  -= 1
        g = CFG.GRID

        if not (0 <= nx < g and 0 <= ny < g) \
                or (nx, ny) in self.bset \
                or self.left <= 0:
            self._die()
            return

        self.body.appendleft((nx, ny))
        self.bset.add((nx, ny))
        self._grid[ny * g + nx] = 1   # mark new head as body

        if nx == self.food[0] and ny == self.food[1]:
            self.score += 1
            self.left   = min(self.left + CFG.MOVE_LIMIT, CFG.MOVE_LIMIT * 3)
            self.food   = self._spawn()   # _spawn sets grid cell to 2
        else:
            tx, ty = self.body.pop()
            self.bset.discard((tx, ty))
            self._grid[ty * g + tx] = 0  # clear tail

    def _die(self):
        self.alive = False
        # Score dominates heavily — moves are a tiebreaker only.
        # Exponential score term means each extra apple is worth far more than the last.
        self.fit = self.score * self.score * 500 + self.score * 200 + self.moves

    def run_to_end(self):
        while self.alive:
            self.step()

# SECTION 7 - GENETICS #

def _tournament(pool, k=6):
    return max(random.sample(pool, min(k, len(pool))), key=lambda s: s.fit)

def evolve(snakes):
    snakes.sort(key=lambda s: s.fit, reverse=True)
    n_elite = max(2, int(CFG.POPULATION * CFG.ELITE_PERCENT))
    nxt = [Snake(s.net.clone()) for s in snakes[:n_elite]]
    while len(nxt) < CFG.POPULATION:
        child_net = _tournament(snakes).net.crossover(_tournament(snakes).net)
        child_net.mutate()
        nxt.append(Snake(child_net))
    return nxt

# SECTION 8 - UI HELPERS #

def apply_style(root):
    root.configure(bg=BG)
    try:
        s = ttk.Style(root)
        s.theme_use("default")
        s.configure("TCombobox",
            fieldbackground=MID, background=PANEL_BG,
            foreground=CYAN, selectbackground=BORDER,
            selectforeground=CYAN, bordercolor=BORDER, arrowcolor=CYAN)
        s.configure("TScrollbar",
            background=PANEL_BG, troughcolor=BORDER,
            arrowcolor=CYAN, bordercolor=BORDER)
    except Exception:
        pass

def make_btn(parent, text, cmd, fg=CYAN, bg=PANEL_BG, pad=(12, 6)):
    b = tk.Button(parent, text=text, command=cmd,
                  bg=bg, fg=fg, activebackground=DIM, activeforeground=WHITE,
                  font=("Courier", 10, "bold"), relief="flat",
                  bd=0, padx=pad[0], pady=pad[1], cursor="hand2")
    b.bind("<Enter>", lambda e: b.configure(bg=DIM))
    b.bind("<Leave>", lambda e: b.configure(bg=bg))
    return b

def make_label(parent, text, fg=TEXT, size=10, bold=False, bg=None):
    return tk.Label(parent, text=text, fg=fg,
                    bg=bg or PANEL_BG,
                    font=("Courier", size, "bold" if bold else "normal"))

def make_sep(parent):
    tk.Frame(parent, bg=BORDER, height=1).pack(fill="x", padx=8, pady=5)

# SECTION 9 - BOOT SCREEN #

BOOT_W = 700
BOOT_H = 520

ASCII_LOGO = [
    "███╗   ██╗███████╗██╗   ██╗██████╗  █████╗ ██╗",
    "████╗  ██║██╔════╝██║   ██║██╔══██╗██╔══██╗██║",
    "██╔██╗ ██║█████╗  ██║   ██║██████╔╝███████║██║",
    "██║╚██╗██║██╔══╝  ██║   ██║██╔══██╗██╔══██║██║",
    "██║ ╚████║███████╗╚██████╔╝██║  ██║██║  ██║███████╗",
    "╚═╝  ╚═══╝╚══════╝ ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝╚══════╝",
    "          ── GENETIC AI TRAINING LAB ──",
]

class BootScreen:
    # result is set to {"action": "new"|"load", "name": str, "visual": bool, "meta": dict|None}
    # or left as None if the window is just closed
    def __init__(self):
        self.result = None
        self.root   = tk.Tk()
        self.root.title("Neural Snake")
        self.root.resizable(False, False)
        apply_style(self.root)
        self._build()
        self._animate_logo(0)
        self.root.mainloop()

    def _build(self):
        root = self.root

        self.logo_canvas = tk.Canvas(root, width=BOOT_W, height=130,
                                     bg=BG, highlightthickness=0)
        self.logo_canvas.pack(fill="x")

        body = tk.Frame(root, bg=BG)
        body.pack(fill="both", expand=True)

        left = tk.Frame(body, bg=PANEL_BG)
        left.pack(side="left", fill="both", expand=True, padx=(12, 6), pady=12)
        self._build_new_panel(left)

        right = tk.Frame(body, bg=PANEL_BG)
        right.pack(side="right", fill="both", expand=True, padx=(6, 12), pady=12)
        self._build_load_panel(right)

        foot = tk.Frame(root, bg=BG)
        foot.pack(fill="x", pady=(0, 10))
        make_label(foot, "pip install numpy   |   python3 neural_snake.py",
                   fg=DIM, size=9, bg=BG).pack()

    def _build_new_panel(self, parent):
        make_label(parent, "NEW MODEL", CYAN, 11, bold=True).pack(anchor="w", padx=10, pady=(10, 4))
        tk.Frame(parent, bg=BORDER, height=1).pack(fill="x", padx=8, pady=(0, 8))

        nf = tk.Frame(parent, bg=PANEL_BG)
        nf.pack(fill="x", padx=10, pady=(0, 6))
        make_label(nf, "NAME:", DIM, 9).pack(side="left")
        self.name_var = tk.StringVar(value=f"run_{time.strftime('%H%M%S')}")
        tk.Entry(nf, textvariable=self.name_var,
                 bg=MID, fg=CYAN, insertbackground=CYAN,
                 font=("Courier", 10), relief="flat", bd=4, width=18
                 ).pack(side="left", padx=(6, 0))

        sf = tk.Frame(parent, bg=PANEL_BG)
        sf.pack(fill="x", padx=10, pady=2)

        sliders = [
            ("POPULATION",   "POPULATION",        10,  200, 10,  None),
            ("HIDDEN SIZE",  "HIDDEN_SIZE",         4,   48,  4,  None),
            ("MUT RATE",     "MUTATION_RATE",    0.01, 0.40, 0.01, ".2f"),
            ("MUT STRENGTH", "MUTATION_STRENGTH",0.05, 1.00, 0.05, ".2f"),
            ("ELITE %",      "ELITE_PERCENT",    0.05, 0.50, 0.05, ".0%"),
            ("MOVE LIMIT",   "MOVE_LIMIT",         50,  600,  50,  None),
        ]
        for i, (lbl, attr, lo, hi, res, fmt) in enumerate(sliders):
            tk.Label(sf, text=lbl, bg=PANEL_BG, fg=DIM,
                     font=("Courier", 8), anchor="w", width=13
                     ).grid(row=i, column=0, sticky="w", pady=1)
            sv = tk.StringVar()
            cur = getattr(CFG, attr)

            def _cb(v, a=attr, s=sv, f=fmt):
                val = float(v)
                setattr(CFG, a, int(val) if f is None else round(val, 3))
                s.set(f"{int(val)}" if f is None else
                      (f"{val:.0%}" if f == ".0%" else f"{val:{f}}"))

            sv.set(f"{int(cur)}" if fmt is None else
                   (f"{cur:.0%}" if fmt == ".0%" else f"{cur:{fmt}}"))
            tk.Scale(sf, from_=lo, to=hi, resolution=res,
                     orient="horizontal", bg=PANEL_BG, fg=TEXT,
                     troughcolor=BORDER, activebackground=CYAN,
                     highlightthickness=0, sliderrelief="flat",
                     length=160, showvalue=False, command=_cb
                     ).grid(row=i, column=1, padx=4, sticky="ew")
            tk.Label(sf, textvariable=sv, bg=PANEL_BG, fg=CYAN,
                     font=("Courier", 8), width=6, anchor="e"
                     ).grid(row=i, column=2, sticky="e")

        af = tk.Frame(parent, bg=PANEL_BG)
        af.pack(fill="x", padx=10, pady=(4, 8))
        make_label(af, "ACTIVATION:", DIM, 8).pack(side="left")
        self.act_var = tk.StringVar(value=CFG.ACTIVATION)
        ttk.Combobox(af, textvariable=self.act_var, width=10,
                     values=["tanh", "relu", "sigmoid", "leaky"],
                     state="readonly", font=("Courier", 9)
                     ).pack(side="left", padx=6)
        self.act_var.trace_add("write", lambda *_: setattr(CFG, "ACTIVATION", self.act_var.get()))

        tk.Frame(parent, bg=BORDER, height=1).pack(fill="x", padx=8, pady=(0, 8))

        bf = tk.Frame(parent, bg=PANEL_BG)
        bf.pack(fill="x", padx=10, pady=(0, 10))
        make_label(bf, "TRAINING MODE:", DIM, 9).pack(anchor="w", pady=(0, 4))
        bb = tk.Frame(bf, bg=PANEL_BG)
        bb.pack(fill="x")
        make_btn(bb, "▶  VISUAL TRAINING", lambda: self._start_new(True),
                 fg=CYAN, pad=(10, 8)).pack(side="left", expand=True, fill="x", padx=(0, 4))
        make_btn(bb, "⚡  HEADLESS (FAST)", lambda: self._start_new(False),
                 fg=GOLD, pad=(10, 8)).pack(side="left", expand=True, fill="x")

    def _build_load_panel(self, parent):
        make_label(parent, "SAVED MODELS", CYAN, 11, bold=True).pack(anchor="w", padx=10, pady=(10, 4))
        tk.Frame(parent, bg=BORDER, height=1).pack(fill="x", padx=8, pady=(0, 8))

        lf = tk.Frame(parent, bg=PANEL_BG)
        lf.pack(fill="both", expand=True, padx=8)
        sb = tk.Scrollbar(lf, orient="vertical")
        sb.pack(side="right", fill="y")
        self.model_lb = tk.Listbox(lf, bg=MID, fg=TEXT, selectbackground=DIM,
                                   selectforeground=CYAN, font=("Courier", 9),
                                   relief="flat", bd=0, activestyle="none",
                                   yscrollcommand=sb.set, height=8)
        self.model_lb.pack(side="left", fill="both", expand=True)
        sb.config(command=self.model_lb.yview)
        self.model_lb.bind("<<ListboxSelect>>", self._on_model_select)

        self.detail_canvas = tk.Canvas(parent, height=120, bg=MID,
                                       highlightthickness=1, highlightbackground=BORDER)
        self.detail_canvas.pack(fill="x", padx=8, pady=(6, 6))

        tk.Frame(parent, bg=BORDER, height=1).pack(fill="x", padx=8, pady=(0, 8))

        bf = tk.Frame(parent, bg=PANEL_BG)
        bf.pack(fill="x", padx=10, pady=(0, 10))
        make_label(bf, "CONTINUE WITH:", DIM, 9).pack(anchor="w", pady=(0, 4))
        bb = tk.Frame(bf, bg=PANEL_BG)
        bb.pack(fill="x")
        make_btn(bb, "▶  VISUAL", lambda: self._start_load(True),
                 fg=CYAN, pad=(10, 8)).pack(side="left", expand=True, fill="x", padx=(0, 4))
        make_btn(bb, "⚡  HEADLESS", lambda: self._start_load(False),
                 fg=GOLD, pad=(10, 8)).pack(side="left", expand=True, fill="x")

        self._all_models    = list_models()
        self._selected_meta = None
        self._populate_model_list()

    def _populate_model_list(self):
        lb = self.model_lb
        lb.delete(0, "end")
        if not self._all_models:
            lb.insert("end", "  (no saved models)")
            return
        for name, meta in self._all_models:
            gen  = meta.get("gen", "?")
            best = meta.get("all_best", "?")
            lb.insert("end", f"  {name:<18}  gen:{gen:<5} best:{best}")

    def _on_model_select(self, event):
        sel = self.model_lb.curselection()
        if not sel or not self._all_models or sel[0] >= len(self._all_models):
            return
        name, meta = self._all_models[sel[0]]
        self._selected_meta = (name, meta)
        self._draw_detail(meta)

    def _draw_detail(self, meta):
        c = self.detail_canvas
        c.delete("all")
        w, h = c.winfo_width() or 330, 120
        cfg_d = meta.get("cfg", {})
        rows = [
            ("MODEL",      meta.get("name", "?")),
            ("SAVED",      meta.get("saved_at", "?")),
            ("GENERATION", str(meta.get("gen", "?"))),
            ("BEST SCORE", str(meta.get("all_best", "?"))),
            ("POPULATION", str(cfg_d.get("POPULATION", "?"))),
            ("HIDDEN",     str(cfg_d.get("HIDDEN_SIZE", "?"))),
            ("ACTIVATION", str(cfg_d.get("ACTIVATION", "?"))),
            ("MUT RATE",   str(cfg_d.get("MUTATION_RATE", "?"))),
        ]
        cw = w // 2
        for i, (k, v) in enumerate(rows):
            col, row = i % 2, i // 2
            x = 10 + col * cw
            y = 14 + row * 24
            c.create_text(x,      y, text=k + ":", fill=DIM,
                          font=("Courier", 8), anchor="w")
            c.create_text(x + 70, y, text=v, fill=CYAN,
                          font=("Courier", 8, "bold"), anchor="w")
        # mini sparkline
        hist = meta.get("score_hist", [])
        if len(hist) > 1:
            mx, sw, sh, sy = max(max(hist), 1), w - 20, 18, h - 22
            bw = sw // len(hist)
            for i, v in enumerate(hist):
                bh = max(1, int(sh * v / mx))
                c.create_rectangle(10 + i * bw, sy + sh - bh,
                                   10 + i * bw + max(1, bw - 1), sy + sh,
                                   fill=lerp_hex(RED_D, CYAN_D, i / len(hist)), outline="")

    def _animate_logo(self, frame):
        c = self.logo_canvas
        c.delete("all")
        for i, line in enumerate(ASCII_LOGO):
            fade = min(1.0, max(0.0, (frame - i * 3) / 12))
            if fade <= 0:
                continue
            col = lerp_hex(BG, CYAN if i < 6 else GOLD, fade)
            c.create_text(BOOT_W // 2, 12 + i * 17, text=line,
                          fill=col, font=("Courier", 8, "bold"), anchor="center")
        if frame > 40 and (frame // 15) % 2 == 0:
            c.create_text(BOOT_W // 2, 120, text="▮", fill=CYAN,
                          font=("Courier", 10), anchor="center")
        if frame < 80:
            self.root.after(40, lambda: self._animate_logo(frame + 1))

    def _start_new(self, visual):
        name = "".join(c for c in self.name_var.get().strip() if c.isalnum() or c in "_-")
        if not name:
            messagebox.showerror("Bad name", "Use letters, numbers, _ or -")
            return
        self.result = {"action": "new", "name": name, "visual": visual, "meta": None}
        self.root.destroy()

    def _start_load(self, visual):
        if not self._selected_meta:
            messagebox.showinfo("Pick a model", "Click a model in the list first.")
            return
        name, meta = self._selected_meta
        CFG.apply_dict(meta.get("cfg", {}))
        self.result = {"action": "load", "name": name, "visual": visual, "meta": meta}
        self.root.destroy()

# SECTION 10 - HEADLESS TRAINER #

def _print_table_header(name):
    print()
    print(f"  Neural Snake  model={name}  pop={CFG.POPULATION}  "
          f"hidden={CFG.HIDDEN_SIZE}  act={CFG.ACTIVATION}")
    print(f"  {'GEN':>5}  {'GEN BEST':>8}  {'ALL TIME':>8}  "
          f"{'AVG SCORE':>9}  {'AVG FIT':>10}  {'G/s':>6}  NOTE")
    print("  " + "─" * 68)

def run_headless(name, snakes, gen, all_best, score_hist):
    # Runs in the main thread - blocks until Ctrl-C
    start     = time.perf_counter()
    gens_done = 0
    _print_table_header(name)

    try:
        while True:
            for s in snakes:
                s.run_to_end()

            scores     = [s.score for s in snakes]
            fits       = [s.fit   for s in snakes]
            gen_best   = max(scores)
            avg_sc     = sum(scores) / len(scores)
            avg_fit    = sum(fits)   / len(fits)
            new_record = gen_best > all_best
            all_best   = max(all_best, gen_best)
            score_hist.append(gen_best)
            if len(score_hist) > 60:
                score_hist.pop(0)

            elapsed = time.perf_counter() - start
            gps     = (gens_done + 1) / max(elapsed, 1e-9)
            note    = "★ NEW BEST" if new_record else ""

            if gens_done > 0 and gens_done % HEADER_EVERY == 0:
                print("  " + "─" * 68)
                print(f"  {'GEN':>5}  {'GEN BEST':>8}  {'ALL TIME':>8}  "
                      f"{'AVG SCORE':>9}  {'AVG FIT':>10}  {'G/s':>6}  NOTE")
                print("  " + "─" * 68)

            print(f"  {gen:>5}  {gen_best:>8}  {all_best:>8}  "
                  f"{avg_sc:>9.2f}  {avg_fit:>10.0f}  {gps:>6.1f}  {note}",
                  flush=True)

            snakes     = evolve(snakes)
            gen       += 1
            gens_done += 1

            if gens_done % 50 == 0:
                save_model(name, snakes, gen, all_best, score_hist)
                print(f"  {'':>5}  ── auto-saved at gen {gen} ──", flush=True)

    except KeyboardInterrupt:
        print("\n  Saving...", flush=True)
        save_model(name, snakes, gen, all_best, score_hist)
        print(f"  Saved  snake_models/{name}/  (gen {gen}, best {all_best})", flush=True)

# SECTION 11 - VISUAL TRAINER #

PANEL_W = 270
NET_H   = 130
CHART_H = 60

class VisualTrainer:
    def __init__(self, root, name, snakes, gen, all_best, score_hist):
        self.root       = root
        self.name       = name
        self.snakes     = snakes
        self.gen        = gen
        self.all_best   = all_best
        self.score_hist = score_hist
        self.paused     = False
        self.tick_acc   = 0.0
        self.last_t     = time.perf_counter()

        board_px = CFG.GRID * CFG.CELL
        root.title(f"Neural Snake — {name}")
        root.configure(bg=BG)
        root.resizable(False, False)
        apply_style(root)

        self.board_c = tk.Canvas(root, width=board_px, height=board_px,
                                 bg=BG, highlightthickness=1,
                                 highlightbackground=BORDER)
        self.board_c.grid(row=0, column=0, padx=(8, 0), pady=8)

        right = tk.Frame(root, bg=PANEL_BG, width=PANEL_W)
        right.grid(row=0, column=1, sticky="ns", padx=8, pady=8)
        right.grid_propagate(False)
        self._build_panel(right)

        for key, fn in (
            ("<space>",       lambda e: self._toggle_pause()),
            ("<b>",           lambda e: self._toggle_best()),
            ("<B>",           lambda e: self._toggle_best()),
            ("<v>",           lambda e: self._toggle_rays()),
            ("<V>",           lambda e: self._toggle_rays()),
            ("<r>",           lambda e: self._restart()),
            ("<R>",           lambda e: self._restart()),
            ("<s>",           lambda e: self._save()),
            ("<S>",           lambda e: self._save()),
            ("<plus>",        lambda e: self._adj_speed(1)),
            ("<equal>",       lambda e: self._adj_speed(1)),
            ("<minus>",       lambda e: self._adj_speed(-1)),
            ("<KP_Add>",      lambda e: self._adj_speed(1)),
            ("<KP_Subtract>", lambda e: self._adj_speed(-1)),
        ):
            root.bind(key, fn)

        self._loop()

    # SECTION 11.1 - PANEL BUILD

    def _build_panel(self, p):
        def lbl(text, fg=TEXT, size=10, bold=False, pady=0):
            tk.Label(p, text=text, bg=PANEL_BG, fg=fg,
                     font=("Courier", size, "bold" if bold else "normal"),
                     anchor="w").pack(fill="x", padx=8, pady=(pady, 0))

        lbl("NEURAL SNAKE", CYAN, 14, bold=True, pady=6)
        lbl(f"/{self.name}", DIM, 9)
        make_sep(p)

        # Stats grid
        sf = tk.Frame(p, bg=PANEL_BG)
        sf.pack(fill="x", padx=8, pady=4)
        self.sv = {}
        for i, (text, key, col) in enumerate([
            ("GEN",   "gen",      CYAN),
            ("BEST",  "all_best", GOLD),
            ("ALIVE", "alive",    TEXT),
            ("SCORE", "score",    TEXT),
            ("MOVES", "moves",    TEXT),
            ("LEFT",  "left",     RED),
        ]):
            r, c = divmod(i, 2)
            tk.Label(sf, text=text + ":", bg=PANEL_BG, fg=DIM,
                     font=("Courier", 9), anchor="w"
                     ).grid(row=r, column=c * 2, sticky="w", padx=(0, 2))
            sv = tk.StringVar(value="0")
            self.sv[key] = sv
            tk.Label(sf, textvariable=sv, bg=PANEL_BG, fg=col,
                     font=("Courier", 9, "bold"), anchor="w"
                     ).grid(row=r, column=c * 2 + 1, sticky="w", padx=(0, 10))

        make_sep(p)
        tk.Label(p, text="MOVES LEFT", bg=PANEL_BG, fg=DIM,
                 font=("Courier", 8), anchor="w").pack(fill="x", padx=8)
        self.life_c = tk.Canvas(p, height=6, bg=BORDER, highlightthickness=0)
        self.life_c.pack(fill="x", padx=8, pady=(2, 4))

        make_sep(p)
        tk.Label(p, text="SCORE HISTORY", bg=PANEL_BG, fg=DIM,
                 font=("Courier", 8, "bold"), anchor="w").pack(fill="x", padx=8)
        self.chart_c = tk.Canvas(p, height=CHART_H, bg=MID,
                                 highlightthickness=1, highlightbackground=BORDER)
        self.chart_c.pack(fill="x", padx=8, pady=(2, 4))

        make_sep(p)
        tk.Label(p, text="NETWORK", bg=PANEL_BG, fg=DIM,
                 font=("Courier", 8, "bold"), anchor="w").pack(fill="x", padx=8)
        self.net_c = tk.Canvas(p, height=NET_H, bg=MID,
                               highlightthickness=1, highlightbackground=BORDER)
        self.net_c.pack(fill="x", padx=8, pady=(2, 4))

        make_sep(p)
        self._build_settings(p)

        make_sep(p)
        cf = tk.Frame(p, bg=PANEL_BG)
        cf.pack(fill="x", padx=8)
        for i, (k, v) in enumerate([
            ("SPACE", "pause"), ("+ / -", "speed"), ("B", "best"),
            ("V",     "rays"),  ("S",     "save"),  ("R", "restart"),
        ]):
            r, c = divmod(i, 2)
            tk.Label(cf, text=f"[{k}]", bg=PANEL_BG, fg=GOLD,
                     font=("Courier", 8)).grid(row=r, column=c * 2, sticky="w")
            tk.Label(cf, text=v, bg=PANEL_BG, fg=DIM,
                     font=("Courier", 8)).grid(row=r, column=c * 2 + 1, sticky="w", padx=(2, 8))

        make_sep(p)
        bf = tk.Frame(p, bg=PANEL_BG)
        bf.pack(fill="x", padx=8, pady=(0, 6))
        make_btn(bf, "💾 SAVE", self._save, fg=GOLD, pad=(8, 5)).pack(side="left", padx=(0, 4))
        make_btn(bf, "↩ MENU", self._to_boot, fg=DIM, pad=(8, 5)).pack(side="right")

        self.status_var = tk.StringVar(value="RUNNING")
        tk.Label(p, textvariable=self.status_var, bg=PANEL_BG, fg=CYAN,
                 font=("Courier", 9, "bold"), anchor="w").pack(fill="x", padx=8, pady=(0, 4))

    def _build_settings(self, parent):
        frame = tk.Frame(parent, bg=PANEL_BG)
        frame.pack(fill="x", padx=8, pady=2)
        sliders = [
            ("POPULATION",  "POPULATION",        10, 200, 10,   None),
            ("HIDDEN",      "HIDDEN_SIZE",         4,  48,  4,   None),
            ("MUT RATE",    "MUTATION_RATE",    0.01, 0.40, 0.01, ".2f"),
            ("MUT STR",     "MUTATION_STRENGTH",0.05, 1.00, 0.05, ".2f"),
            ("ELITE %",     "ELITE_PERCENT",    0.05, 0.50, 0.05, ".2f"),
            ("MOVE LIMIT",  "MOVE_LIMIT",         50, 600,  50,  None),
        ]
        for i, (lbl_t, attr, lo, hi, res, fmt) in enumerate(sliders):
            tk.Label(frame, text=lbl_t, bg=PANEL_BG, fg=DIM,
                     font=("Courier", 8), anchor="w", width=9
                     ).grid(row=i, column=0, sticky="w", pady=1)
            sv  = tk.StringVar()
            cur = getattr(CFG, attr)

            def _cb(v, a=attr, s=sv, f=fmt):
                val = float(v)
                setattr(CFG, a, int(val) if f is None else round(val, 3))
                s.set(f"{int(val)}" if f is None else f"{val:{f}}")

            sv.set(f"{int(cur)}" if fmt is None else f"{cur:{fmt}}")
            tk.Scale(frame, from_=lo, to=hi, resolution=res,
                     orient="horizontal", bg=PANEL_BG, fg=TEXT,
                     troughcolor=BORDER, activebackground=CYAN,
                     highlightthickness=0, sliderrelief="flat",
                     length=130, showvalue=False, command=_cb
                     ).grid(row=i, column=1, sticky="ew", padx=4)
            tk.Label(frame, textvariable=sv, bg=PANEL_BG, fg=CYAN,
                     font=("Courier", 8), width=5, anchor="e"
                     ).grid(row=i, column=2, sticky="e")

        r = len(sliders)
        tk.Label(frame, text="ACTIV.", bg=PANEL_BG, fg=DIM,
                 font=("Courier", 8), width=9, anchor="w"
                 ).grid(row=r, column=0, sticky="w")
        self.act_var = tk.StringVar(value=CFG.ACTIVATION)
        om = ttk.Combobox(frame, textvariable=self.act_var, width=9,
                          values=["tanh", "relu", "sigmoid", "leaky"],
                          state="readonly", font=("Courier", 8))
        om.grid(row=r, column=1, columnspan=2, sticky="ew", padx=4, pady=2)
        om.bind("<<ComboboxSelected>>",
                lambda e: setattr(CFG, "ACTIVATION", self.act_var.get()))

    # SECTION 11.2 - CONTROLS

    def _toggle_pause(self):
        self.paused = not self.paused
        self.last_t = time.perf_counter()
        self.status_var.set("⏸ PAUSED" if self.paused else "▶ RUNNING")

    def _toggle_best(self): CFG.WATCH_BEST = not CFG.WATCH_BEST
    def _toggle_rays(self): CFG.SHOW_RAYS  = not CFG.SHOW_RAYS

    def _restart(self):
        self.snakes     = [Snake() for _ in range(CFG.POPULATION)]
        self.gen        = 1
        self.all_best   = 0
        self.score_hist = []
        self.tick_acc   = 0.0
        self.last_t     = time.perf_counter()

    def _adj_speed(self, d):
        if d > 0:
            CFG.SPEED = 1 if CFG.SPEED == 0 else min(CFG.SPEED + (5 if CFG.SPEED >= 10 else 2), 120)
        else:
            CFG.SPEED = 0 if CFG.SPEED <= 2 else max(CFG.SPEED - (5 if CFG.SPEED > 10 else 2), 1)

    def _save(self):
        save_model(self.name, self.snakes, self.gen, self.all_best, self.score_hist)
        self.status_var.set(f"💾 SAVED gen {self.gen}")
        self.root.after(2000, lambda: self.status_var.set("▶ RUNNING"))

    def _to_boot(self):
        self._save()
        self.root.destroy()
        launch()

    # SECTION 11.3 - SIMULATION

    def _best_snake(self):
        alive = [s for s in self.snakes if s.alive]
        pool  = alive if alive else self.snakes
        return max(pool, key=lambda s: (s.score, s.fit)) if CFG.WATCH_BEST else pool[0]

    def _next_gen(self):
        bs = max(s.score for s in self.snakes)
        self.all_best = max(self.all_best, bs)
        self.score_hist.append(bs)
        if len(self.score_hist) > 60:
            self.score_hist.pop(0)
        self.snakes = evolve(self.snakes)
        self.gen   += 1
        if self.gen % 25 == 0:
            save_model(self.name, self.snakes, self.gen, self.all_best, self.score_hist)

    def _loop(self):
        now = time.perf_counter()
        dt  = now - self.last_t
        self.last_t = now

        if not self.paused:
            if CFG.SPEED == 0:
                for s in self.snakes:
                    if s.alive:
                        s.run_to_end()
                if not any(s.alive for s in self.snakes):
                    self._next_gen()
            else:
                self.tick_acc += dt
                per_tick = 1.0 / CFG.SPEED
                steps    = 0
                while self.tick_acc >= per_tick and steps < 40:
                    alive = [s for s in self.snakes if s.alive]
                    if not alive:
                        self._next_gen()
                    else:
                        for s in alive:
                            s.step()
                    self.tick_acc -= per_tick
                    steps         += 1

        self._draw_board()
        self._update_panel()
        self.root.after(16, self._loop)

    # SECTION 11.4 - DRAWING

    def _draw_board(self):
        c = self.board_c
        c.delete("all")
        C = CFG.CELL
        G = CFG.GRID
        bp = G * C

        for i in range(G + 1):
            c.create_line(i * C, 0, i * C, bp, fill=GRID_COL)
            c.create_line(0, i * C, bp, i * C, fill=GRID_COL)

        snake = self._best_snake()
        if not snake:
            return

        if CFG.SHOW_RAYS:
            hx, hy = snake.body[0]
            for dx, dy in ALL_DIRS:
                x, y = hx + dx, hy + dy
                while 0 <= x < G and 0 <= y < G:
                    if (x, y) in snake.bset or (x, y) == snake.food:
                        break
                    x += dx; y += dy
                c.create_line(hx * C + C // 2, hy * C + C // 2,
                              x  * C + C // 2,  y * C + C // 2,
                              fill="#004a40", width=1)

        fx, fy = snake.food
        c.create_rectangle(fx * C - 4, fy * C - 4, fx * C + C + 4, fy * C + C + 4,
                           fill="#330010", outline="")
        c.create_rectangle(fx * C + 2, fy * C + 2, fx * C + C - 2, fy * C + C - 2,
                           fill=RED, outline="")

        body = list(snake.body)
        n    = len(body)
        for i, (bx, by) in enumerate(body):
            t = i / max(n - 1, 1)
            if i == 0:
                c.create_rectangle(bx * C, by * C, bx * C + C, by * C + C,
                                   fill="#003330", outline="")
                col, pad = CYAN, 1
            else:
                col, pad = lerp_hex(CYAN_D, "#001a14", t), 2
            c.create_rectangle(bx * C + pad, by * C + pad,
                                bx * C + C - pad, by * C + C - pad,
                                fill=col, outline="")

        c.create_text(bp // 2, bp // 2, text=str(snake.score),
                      font=("Courier", 52, "bold"), fill="#003328")

        if self.paused:
            c.create_rectangle(0, 0, bp, bp, fill=BG, stipple="gray50")
            c.create_text(bp // 2, bp // 2, text="PAUSED",
                          font=("Courier", 32, "bold"), fill=CYAN)
            c.create_text(bp // 2, bp // 2 + 40, text="press SPACE to resume",
                          font=("Courier", 12), fill=DIM)

    def _update_panel(self):
        snake     = self._best_snake()
        alive_cnt = sum(1 for s in self.snakes if s.alive)
        self.sv["gen"     ].set(str(self.gen))
        self.sv["all_best"].set(str(self.all_best))
        self.sv["alive"   ].set(f"{alive_cnt}/{CFG.POPULATION}")
        if snake:
            self.sv["score"].set(str(snake.score))
            self.sv["moves"].set(str(snake.moves))
            self.sv["left" ].set(str(snake.left))
            self._draw_life_bar(snake.left)
            self._draw_network(snake.net)
        self._draw_chart()
        if not self.paused:
            spd = "MAX" if CFG.SPEED == 0 else f"{CFG.SPEED} t/s"
            self.status_var.set(f"▶ {spd}{' [BEST]' if CFG.WATCH_BEST else ''}")

    def _draw_life_bar(self, left):
        c = self.life_c
        c.delete("all")
        w   = c.winfo_width() or (PANEL_W - 16)
        pct = max(0.0, min(left / CFG.MOVE_LIMIT, 1.0))
        c.create_rectangle(0, 0, w, 6, fill=BORDER, outline="")
        if pct > 0:
            c.create_rectangle(0, 0, int(w * pct), 6,
                               fill=lerp_hex(RED, CYAN, pct), outline="")

    def _draw_chart(self):
        c    = self.chart_c
        c.delete("all")
        hist = self.score_hist
        if len(hist) < 2:
            return
        w  = c.winfo_width() or (PANEL_W - 16)
        h  = CHART_H
        mx = max(max(hist), 1)
        bw = max(1, w // len(hist))
        for i, v in enumerate(hist):
            bh = max(1, int((h - 4) * v / mx))
            c.create_rectangle(i * bw, h - bh, i * bw + max(1, bw - 1), h,
                               fill=lerp_hex(RED_D, CYAN_D, i / len(hist)), outline="")
        pts = []
        for i, v in enumerate(hist):
            pts += [i * bw + bw // 2, h - max(1, int((h - 4) * v / mx))]
        if len(pts) >= 4:
            c.create_line(*pts, fill=CYAN, width=1, smooth=True)

    def _draw_network(self, net):
        c = self.net_c
        c.delete("all")
        w  = c.winfo_width() or (PANEL_W - 16)
        h  = NET_H
        ni, nh, no = 24, CFG.HIDDEN_SIZE, 4
        lx = [int(w * 0.07), int(w * 0.50), int(w * 0.93)]

        def ys(n):
            if n == 1:
                return [h // 2]
            return [int(4 + i * (h - 8) / (n - 1)) for i in range(n)]

        in_y, hid_y, out_y = ys(ni), ys(nh), ys(no)

        for i in range(0, ni, 4):
            x1, y1 = lx[0], in_y[i]
            for j in range(nh):
                wv = float(net.W1[i, j])
                c.create_line(x1, y1, lx[1], hid_y[j],
                              fill=lerp_hex(BORDER, CYAN_D if wv > 0 else RED_D,
                                           min(abs(wv) * 0.6, 1)))

        for i in range(nh):
            x1, y1 = lx[1], hid_y[i]
            for j in range(no):
                wv = float(net.W2[i, j])
                c.create_line(x1, y1, lx[2], out_y[j],
                              fill=lerp_hex(BORDER, CYAN_D if wv > 0 else RED_D,
                                           min(abs(wv) * 0.8, 1)))

        for y in in_y:
            c.create_oval(lx[0] - 2, y - 2, lx[0] + 2, y + 2, fill=BORDER, outline="")

        for j, y in enumerate(hid_y):
            act = float(np.clip(net.h[j], -1, 1))
            col = lerp_hex(DIM, CYAN, act) if act >= 0 else lerp_hex(DIM, RED, abs(act))
            c.create_oval(lx[1] - 4, y - 4, lx[1] + 4, y + 4, fill=col, outline="")

        for j, y in enumerate(out_y):
            act = float(np.clip(net.o[j], -1, 1))
            col = lerp_hex(DIM, CYAN, act) if act >= 0 else lerp_hex(DIM, RED, abs(act))
            c.create_oval(lx[2] - 5, y - 5, lx[2] + 5, y + 5, fill=col, outline="")
            c.create_text(lx[2] + 12, y, text=("↑", "→", "↓", "←")[j],
                          fill=WHITE, font=("Courier", 8))

# SECTION 12 - LAUNCH #

def _build_population(result):
    # Returns (snakes, gen, all_best, score_hist) ready to pass to a trainer
    if result["action"] == "new":
        return [Snake() for _ in range(CFG.POPULATION)], 1, 0, []

    meta       = result["meta"]
    gen        = meta.get("gen", 1)
    all_best   = meta.get("all_best", 0)
    score_hist = meta.get("score_hist", [])
    npz        = load_weights(result["name"])
    snakes     = []

    if npz is not None:
        n_elite = max(2, int(CFG.POPULATION * CFG.ELITE_PERCENT))
        for _ in range(n_elite):
            net = Net()
            net.load_weights(npz)
            snakes.append(Snake(net))

    while len(snakes) < CFG.POPULATION:
        if snakes:
            cn = random.choice(snakes).net.clone()
            cn.mutate()
            snakes.append(Snake(cn))
        else:
            snakes.append(Snake())

    return snakes, gen, all_best, score_hist

def launch():
    boot = BootScreen()
    if boot.result is None:
        return

    result = boot.result
    name   = result["name"]
    snakes, gen, all_best, score_hist = _build_population(result)

    if result["visual"]:
        root = tk.Tk()
        apply_style(root)
        VisualTrainer(root, name, snakes, gen, all_best, score_hist)
        root.mainloop()
    else:
        run_headless(name, snakes, gen, all_best, score_hist)

# SECTION 13 - ENTRY POINT #

if __name__ == "__main__":
    launch()