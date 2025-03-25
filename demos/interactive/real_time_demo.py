#!/usr/bin/env python3
"""
Real-time demonstration of ADAWorld's adaptive world model learning.
This demo shows the agent learning to predict and adapt to environment dynamics.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pygame
import numpy as np
import torch
import gymnasium as gym
from datetime import datetime
from pathlib import Path
from threading import Thread
from queue import Queue
import sqlite3
from src.agent.world_model import WorldModel

class RealTimeDemo:
    def __init__(self):
        # Initialize Pygame
        pygame.init()
        self.width = 800
        self.height = 600
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("ADAWorld Real-time Demo")
        
        # Colors
        self.BLACK = (0, 0, 0)
        self.WHITE = (255, 255, 255)
        self.BLUE = (0, 0, 255)
        self.GREEN = (0, 255, 0)
        
        # Initialize environment and model
        self.env = gym.make('Pendulum-v1', render_mode='rgb_array')
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]
        self.world_model = WorldModel(self.state_dim, self.action_dim)
        
        # Training data queue
        self.data_queue = Queue()
        self.running = True
        
        # Database path
        self.db_path = Path('database/adaworld.db')
        
        # Metrics
        self.episode_count = 0
        self.total_reward = 0
        self.prediction_errors = []
        
    def start_training_thread(self):
        """Start the background training thread"""
        self.training_thread = Thread(target=self._training_loop)
        self.training_thread.daemon = True
        self.training_thread.start()
        
    def _training_loop(self):
        """Background training process"""
        while self.running:
            if not self.data_queue.empty():
                state, action, next_state = self.data_queue.get()
                
                # Convert to tensors
                state = torch.FloatTensor(state)
                action = torch.FloatTensor(action)
                next_state = torch.FloatTensor(next_state)
                
                # Train world model
                predicted_next_state, _ = self.world_model(state.unsqueeze(0), 
                                                         action.unsqueeze(0))
                
                # Calculate prediction error
                error = torch.nn.MSELoss()(predicted_next_state.squeeze(),
                                          next_state)
                self.prediction_errors.append(error.item())
                
                # Save to database
                self._save_training_data(state, action, next_state, error.item())
    
    def _save_training_data(self, state, action, next_state, error):
        """Save training data to SQLite database"""
        # Create a new connection in this thread
        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.cursor()
            timestamp = datetime.now().isoformat()
            cursor.execute("""
                INSERT INTO training_runs 
                (timestamp, model_config, metrics) 
                VALUES (?, ?, ?)
            """, (timestamp, str(self.world_model), 
                  '{"error": %f, "reward": %f}' % (error, self.total_reward)))
    
    def render_stats(self):
        """Render training statistics"""
        font = pygame.font.Font(None, 36)
        
        # Render episode count
        episode_text = font.render(f'Episode: {self.episode_count}', True, self.WHITE)
        self.screen.blit(episode_text, (10, 10))
        
        # Render total reward
        reward_text = font.render(f'Total Reward: {self.total_reward:.2f}', 
                                True, self.WHITE)
        self.screen.blit(reward_text, (10, 50))
        
        # Render prediction error
        if self.prediction_errors:
            error_text = font.render(
                f'Prediction Error: {self.prediction_errors[-1]:.4f}', 
                True, self.WHITE)
            self.screen.blit(error_text, (10, 90))
        
        # Render learning curve
        if len(self.prediction_errors) > 1:
            points = [(i * 2, 500 - error * 100) 
                     for i, error in enumerate(self.prediction_errors[-100:])]
            if len(points) > 1:
                pygame.draw.lines(self.screen, self.GREEN, False, points, 2)
    
    def run(self):
        """Main demo loop"""
        self.start_training_thread()
        clock = pygame.time.Clock()
        state, _ = self.env.reset()
        
        try:
            while self.running:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.running = False
                
                # Get action and step environment
                action = self.env.action_space.sample()  # Random action for demo
                next_state, reward, done, _, _ = self.env.step(action)
                
                # Add to training queue
                self.data_queue.put((state, action, next_state))
                
                # Update metrics
                self.total_reward += reward
                
                # Render
                self.screen.fill(self.BLACK)
                
                # Render environment
                env_image = self.env.render()
                env_surface = pygame.surfarray.make_surface(
                    np.transpose(env_image, (1, 0, 2)))
                env_surface = pygame.transform.scale(env_surface, (400, 400))
                self.screen.blit(env_surface, (350, 100))
                
                # Render stats
                self.render_stats()
                
                pygame.display.flip()
                clock.tick(30)
                
                if done:
                    state, _ = self.env.reset()
                    self.episode_count += 1
                else:
                    state = next_state
                    
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Cleanup resources"""
        self.running = False
        self.env.close()
        pygame.quit()

def main():
    demo = RealTimeDemo()
    demo.run()

if __name__ == "__main__":
    main()