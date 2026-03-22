# Neural Snake

A snake game where a neural network learns to play through a genetic algorithm. Built with Python, tkinter, and numpy. You can watch it train in real time or run it headless from the terminal, save models, and pick up where you left off.

> **Author:** LordCarvell

---

## Features

* **Boot screen** - animated launch screen lets you start a new training run or load a saved model. Shows generation count, best score, and a score history sparkline for each saved model
* **Visual training** - watch the best snake play in real time with a live neural network diagram, score history chart, and all training settings adjustable while it runs
* **Headless training** - no window, just a table in the terminal. Faster because nothing is being drawn. Prints generation, best score, all-time best, average score, average fitness, and generations per second
* **Save and load** - models save to `snake_models/<name>/` as a weights file and a metadata JSON. Loading seeds the population with the saved weights so training picks up from where it left off
* **Auto-save** - saves every 25 generations in visual mode, every 50 in headless
* **Live settings sliders** - population size, hidden layer size, mutation rate, mutation strength, elite percentage, move limit, and activation function can all be changed mid-run without restarting
* **Vision rays** - toggle a debug overlay that shows what the snake can see in each direction

---

## Requirements

* Python 3.9 or newer
* numpy

tkinter comes with Python on Windows and macOS. On Linux you may need to install it separately.

```
pip install numpy
```

Linux tkinter:

```
sudo apt install python3-tk
```

---

## Installation

**1. Clone or download the repo**

```
git clone https://github.com/LordCarvell/NeuralSnake.git
cd NeuralSnake
```

Or just download `neural_snake.py` on its own.

**2. Install the dependency**

```
pip install numpy
```

**3. Run it**

```
python neural_snake.py
```

---

## How It Works

The snake sees the board through 8 raycasts fired from its head in all cardinal and diagonal directions. Each ray reports three values: distance to the wall, whether food is on that ray, and how close the nearest body segment is. That gives 24 inputs total.

Those inputs go into a small neural network with one hidden layer and four outputs (up, right, down, left). Whichever output fires strongest becomes the snake's next move.

The network is not trained with backprop. Instead a genetic algorithm runs a population of snakes every generation, scores them by fitness, then breeds the next generation from the best performers using tournament selection, crossover, and mutation.

Fitness is calculated at death:

```
score * score * 500 + score * 200 + moves
```

The squared score term means each extra apple is worth more than the last, so the network is always under pressure to keep eating rather than plateau.

---

## Controls

### Visual trainer

| Key | Action |
| --- | --- |
| `Space` | Pause and resume |
| `+` / `-` | Speed up or slow down |
| `B` | Toggle watching best snake only |
| `V` | Toggle vision rays |
| `S` | Save now |
| `R` | Restart training from scratch |

### Headless trainer

Just `Ctrl-C` to stop. It saves automatically before exiting.

---

## Training settings

| Setting | What it does |
| --- | --- |
| Population | Number of snakes per generation. More snakes means more variety but slower generations |
| Hidden size | Neurons in the hidden layer. Bigger network can learn more complex behaviour but takes longer to converge |
| Mutation rate | Probability that any given weight gets mutated each generation |
| Mutation strength | How large those mutations are |
| Elite % | Fraction of the top performers that survive unchanged into the next generation |
| Move limit | Steps before a snake starves. Resets partially on each apple eaten |
| Activation | Activation function for the hidden layer. Tanh works well to start |

---

## Project Structure

```
NeuralSnake/
├── neural_snake.py     # whole application in one file
├── snake_models/       # created automatically when you first save
│   └── <model_name>/
│       ├── weights.npz
│       └── meta.json
└── README.md
```

---

## Known Issues

* **Slow early training** - the first 20 or 30 generations look like random noise. The snakes are dying almost immediately and fitness scores are near zero. This is normal, just leave it running
* **Plateau around 5 to 10 apples** - this is a known hard point for genetic approaches to snake. The snake learns to reach nearby food quickly but struggles to plan a route that avoids trapping itself as the body gets longer. Increasing population size and running more generations helps
* **Headless is faster but gives no visual feedback** - if you want to see what the snake learned after headless training, save and reload the model in visual mode
* **Loading a model reseeds the population** - it does not restore every individual from the saved run, just the best weights. The rest of the population is generated by mutating copies of those weights. You will see a dip in performance for a few generations while diversity rebuilds
* **Changing hidden size requires a fresh model** - the network dimensions are fixed at creation time. Changing the hidden size slider mid-run on a loaded model will not take effect until you start a new run
* **tkinter on some Linux setups can feel sluggish** - if the visual trainer is dropping frames try reducing the population or switching to headless

---

## Roadmap

* Parallel generation evaluation using multiprocessing to get around the GIL
* Configurable number of hidden layers
* Export a replay of the best snake as a GIF
* Graph of fitness distribution per generation, not just best score
* Option to watch any individual snake in the population, not just the best

---

## Built With

* [numpy](https://numpy.org/) - neural network forward pass and weight operations
* [tkinter](https://docs.python.org/3/library/tkinter.html) - GUI (comes with Python)