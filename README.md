# ADAWorld Agent 🤖

A real-time demonstration of learning Adaptable World Models with Latent Actions using the Agno framework.

## Overview ⭐

This implementation demonstrates how an AI agent can learn adaptable world models by discovering latent action representations. The key components include:

- **World Model**: A neural network that learns to predict environment dynamics
- **Latent Action Space**: Automatically discovered action representations
- **Real-time Training**: Visual feedback of the learning process

## Project Structure 📁

```
ADAWorld/
├── database/           # SQLite database for storing training data
├── docs/              # Documentation files
├── logs/              # Training and runtime logs
├── models/            # Saved model checkpoints
├── src/               # Source code
│   ├── agent/        # World model and trainer implementations
│   ├── config/       # Configuration and settings
│   ├── data/         # Data loading and processing
│   └── utils/        # Utility functions and visualizers
└── tests/            # Unit and integration tests
```

## Features 🚀

- Dynamic world model learning
- Latent action space discovery
- Real-time training visualization
- Integration with OpenAI Gym environments
- SQLite database for training data persistence
- Modular and extensible architecture

## Installation 🔧

```bash
pip install -r requirements.txt
```

## Usage 🎮

1. Initialize the environment:
```bash
python src/agent/trainer.py --init
```

2. Train the world model:
```bash
python src/agent/trainer.py --train
```

3. Visualize results:
```bash
python src/utils/visualizer.py
```

## Database Schema 💾

- **training_runs**: Stores training session metadata
- **world_states**: Captures environment states during training

## Documentation 📚

Detailed documentation available in the `docs/` directory:
- API Reference: `docs/API.md`
- Training Guide: `docs/training.md`

## Testing 🧪

Run the test suite:
```bash
python -m pytest tests/
```

## Interactive Demonstrations 🎮

The project includes real-time interactive demonstrations to showcase the adaptive world model learning:

### Real-time Learning Demo

Watch the agent learn to predict environment dynamics in real-time:

```bash
python demos/interactive/real_time_demo.py
```

This demo shows:
- Live environment visualization
- Real-time prediction error metrics
- Learning curve visualization
- Training data being saved to the database

### Features of the Demo
- **Visual Feedback**: See the pendulum environment and predictions
- **Live Metrics**: Watch prediction errors decrease in real-time
- **Learning Progress**: View the learning curve as it develops
- **Data Persistence**: All training data is saved for analysis