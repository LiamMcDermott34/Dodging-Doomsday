#!/usr/bin/env python3
"""
Asteroid_Deflection_Env.py

This file combines a Pygame-based solar system simulation with an RL agent.
The RL agent is used to choose the spacecraft’s initial velocity (delta‑v) from a continuous range.
Once launched, the spacecraft is affected by gravity from the Sun and planets.
The agent now receives an extended observation including the asteroid's initial position,
velocity, and mass.
Modified so that in demo mode the agent picks the best delta‑v for each unique asteroid.
"""

import sys
import pygame
import gym
import math
import random
import copy
import numpy as np
import pickle
import os
from gym import spaces

#############################
# Pygame Setup and Constants
#############################
pygame.init()
WIDTH, HEIGHT = 1200, 800
WIN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Asteroid Deflection RL - Initial Velocity Selection")

# Colors and font definitions
WHITE      = (255, 255, 255)
YELLOW     = (255, 255, 0)
BLUE       = (100, 149, 237)
RED        = (188, 39, 50)
DARK_GREY  = (80, 78, 81)
ORANGE     = (255, 165, 0)
LIGHT_BLUE = (173, 216, 230)
GREEN      = (0, 255, 0)
PURPLE     = (128, 0, 128)
FONT = pygame.font.SysFont("comicsans", 16)

closest_approach = 0

#############################
# Simulation: Celestial Bodies
#############################
class CelestialBody:
    AU = 149.6e6 * 1000  # Astronomical unit in meters
    G = 6.67428e-11      # Gravitational constant
    SCALE = 250 / AU     # Scale for display: 1 AU = 250 pixels
    TIMESTEP = 3600 * 24  # One day per physics step

    def __init__(self, x, y, radius, color, mass, name, sun=False, physical_radius=None):
        self.x = x
        self.y = y
        self.radius = radius         # Graphical radius for drawing
        self.color = color
        self.mass = mass
        self.name = name
        self.sun = sun
        self.orbit = []
        self.x_vel = 0
        self.y_vel = 0
        self.physical_radius = physical_radius if physical_radius is not None else radius * 1e6

        # Additional properties for asteroid physics
        self.rotation = 0
        self.material = "rock"  # default material

        if not sun:
            self.distance_to_sun = math.sqrt(self.x**2 + self.y**2)
            if self.distance_to_sun != 0:
                # Approximate circular orbital velocity about the Sun
                v = math.sqrt(CelestialBody.G * 1.98892e30 / self.distance_to_sun)
                self.y_vel = v * (self.x / self.distance_to_sun)
                self.x_vel = -v * (self.y / self.distance_to_sun)
            else:
                self.x_vel = 0
                self.y_vel = 0

    def draw(self, win, scale_factor, offset_x, offset_y):
        x = self.x * CelestialBody.SCALE * scale_factor + offset_x
        y = self.y * CelestialBody.SCALE * scale_factor + offset_y

        if len(self.orbit) > 2:
            updated_points = [(p[0] * CelestialBody.SCALE * scale_factor + offset_x,
                               p[1] * CelestialBody.SCALE * scale_factor + offset_y)
                              for p in self.orbit]
            pygame.draw.lines(win, self.color, False, updated_points, 2)

        pygame.draw.circle(win, self.color, (int(x), int(y)), int(self.radius * scale_factor))

    def attraction(self, other):
        distance_x = other.x - self.x
        distance_y = other.y - self.y
        distance = math.sqrt(distance_x**2 + distance_y**2)
        if distance == 0:
            return 0.0, 0.0
        force = CelestialBody.G * self.mass * other.mass / distance**2
        theta = math.atan2(distance_y, distance_x)
        fx = math.cos(theta) * force
        fy = math.sin(theta) * force
        return fx, fy

#############################
# Simulation Utility Functions
#############################
def load_asteroid_parameters(file_path):
    """Load asteroid parameters from file.
       Each line should have 4 numbers separated by commas (x, y, x_vel, y_vel).
       (Mass will be randomized if not provided.)"""
    asteroids = []
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split(', ')
            if len(parts) == 4:
                # Convert to float and store as tuple
                asteroids.append(tuple(map(float, parts)))
    print("Loaded asteroid parameters:", asteroids)
    return asteroids

def create_spacecraft(earth, delta_vx, delta_vy, mass=1e5, fuel=1e4):
    """Create the spacecraft at Earth’s position with the chosen initial delta‑v."""
    spacecraft = CelestialBody(
        earth.x, earth.y,
        5, WHITE, mass, "Spacecraft", physical_radius=10  # Example physical radius in meters
    )
    spacecraft.x_vel = earth.x_vel + delta_vx
    spacecraft.y_vel = earth.y_vel + delta_vy
    spacecraft.fuel = fuel  # Add fuel constraint attribute
    return spacecraft

def check_collision(obj1, obj2):
    distance = math.sqrt((obj1.x - obj2.x)**2 + (obj1.y - obj2.y)**2)
    scaled_distance = distance * CelestialBody.SCALE
    effective_radius = obj1.radius + obj2.radius
    return scaled_distance < effective_radius

def run_simulation_episode(asteroid_params, delta_vx, delta_vy):
    """
    Run one simulation episode using the given initial velocity for the spacecraft.
    The simulation creates the celestial bodies, including the spacecraft (with the AI-chosen delta‑v),
    and then runs until a collision event occurs or maximum time is reached.
    """
    # Create the main bodies
    sun = CelestialBody(0, 0, 30, YELLOW, 1.98892e30, "Sun", sun=True, physical_radius=696340e3)
    earth = CelestialBody(-1 * CelestialBody.AU, 0, 16, BLUE, 5.9742e24, "Earth", physical_radius=6.371e6)
    original_planets = [
        sun,
        earth,
        CelestialBody(-1.524 * CelestialBody.AU, 0, 12, RED, 6.39e23, "Mars", physical_radius=3.3895e6),
        CelestialBody(0.387 * CelestialBody.AU, 0, 8, DARK_GREY, 3.30e23, "Mercury", physical_radius=2.4397e6),
        CelestialBody(0.723 * CelestialBody.AU, 0, 14, WHITE, 4.8685e24, "Venus", physical_radius=6.0518e6),
        CelestialBody(-5.2 * CelestialBody.AU, 0, 20, ORANGE, 1.898e27, "Jupiter", physical_radius=69.911e6),
        CelestialBody(-9.5 * CelestialBody.AU, 0, 18, LIGHT_BLUE, 5.683e26, "Saturn", physical_radius=58.232e6),
        CelestialBody(-19.8 * CelestialBody.AU, 0, 17, PURPLE, 8.681e25, "Uranus", physical_radius=25.362e6),
        CelestialBody(-30.1 * CelestialBody.AU, 0, 16, LIGHT_BLUE, 1.024e26, "Neptune", physical_radius=24.622e6),
    ]
    planets = [copy.deepcopy(p) for p in original_planets]

    # Create the asteroid from parameters.
    # If asteroid_params only has 4 values, randomize the mass and add a small velocity perturbation.
    if len(asteroid_params) == 4:
        mass = random.uniform(1e9, 1e12)
        asteroid_params = asteroid_params + (mass,)
    asteroid = CelestialBody(*asteroid_params[:2], 5, GREEN, asteroid_params[4], "Asteroid", physical_radius=1e4)
    asteroid.x_vel, asteroid.y_vel = asteroid_params[2], asteroid_params[3]
    asteroid.rotation = 0
    asteroid.material = random.choice(["regolith", "rock"])

    # Create the spacecraft using Earth’s velocity plus the chosen initial delta‑v
    spacecraft = create_spacecraft(earth, delta_vx, delta_vy)
    all_bodies = planets + [asteroid, spacecraft]

    time_elapsed = 0
    max_time = 365 * 100 * CelestialBody.TIMESTEP  # Maximum simulation time
    collision_info = {
        'spacecraft_asteroid': False,
        'spacecraft_planet': None,
        'asteroid_planet': None,
        'earth_collision': False
    }
    spacecraft_left_earth = False

    # Initialize min_distance to a large value so we can track the closest approach
    min_distance = float('inf')

    running = True
    clock = pygame.time.Clock()
    
    while running and time_elapsed < max_time:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                pygame.quit()
                return None

        # Calculate gravitational forces on each body
        accelerations = {body: [0, 0] for body in all_bodies}
        for body in all_bodies:
            for other in all_bodies:
                if body is other:
                    continue
                fx, fy = body.attraction(other)
                accelerations[body][0] += fx / body.mass
                accelerations[body][1] += fy / body.mass

        # Update positions and velocities
        for body in all_bodies:
            ax, ay = accelerations[body]
            body.x_vel += ax * CelestialBody.TIMESTEP
            body.y_vel += ay * CelestialBody.TIMESTEP
            body.x += body.x_vel * CelestialBody.TIMESTEP
            body.y += body.y_vel * CelestialBody.TIMESTEP
            body.orbit.append((body.x, body.y))

        # Update min_distance between the spacecraft and the asteroid
        current_distance = math.sqrt((spacecraft.x - asteroid.x)**2 + (spacecraft.y - asteroid.y)**2)
        if current_distance < min_distance:
            min_distance = current_distance

        # Check collision between the spacecraft and the asteroid
        if not collision_info['spacecraft_asteroid']:
            if check_collision(spacecraft, asteroid):
                collision_info['spacecraft_asteroid'] = True
                # Use a coefficient of restitution to simulate inelastic collision
                restitution = 0.7
                if asteroid.material == "regolith":
                    restitution *= 0.9
                total_mass = asteroid.mass + spacecraft.mass
                asteroid.x_vel = ((asteroid.mass * asteroid.x_vel + spacecraft.mass * spacecraft.x_vel) / total_mass) * restitution
                asteroid.y_vel = ((asteroid.mass * asteroid.y_vel + spacecraft.mass * spacecraft.y_vel) / total_mass) * restitution
                asteroid.mass = total_mass
                impact_speed = math.sqrt(delta_vx**2 + delta_vy**2)
                asteroid.rotation += (spacecraft.mass * impact_speed) / total_mass
                if impact_speed > 3000:
                    asteroid.fragmented = True
                    asteroid.mass *= 0.5  # simulate fragmentation

                # Remove spacecraft from simulation so it is no longer updated or drawn
                if spacecraft in all_bodies:
                    all_bodies.remove(spacecraft)

        # Check if spacecraft has left Earth’s vicinity
        if not spacecraft_left_earth:
            distance = math.sqrt((spacecraft.x - earth.x)**2 + (spacecraft.y - earth.y)**2)
            if distance > earth.physical_radius + spacecraft.radius:
                spacecraft_left_earth = True

        # Check collision between the spacecraft and any non‑Earth body
        for planet in planets:
            if planet.name == "Earth":
                continue
            if check_collision(spacecraft, planet):
                collision_info['spacecraft_planet'] = planet
                running = False  # End simulation immediately on collision
                break

        # Check collision between the asteroid and any planet
        if not collision_info['asteroid_planet']:
            for planet in planets:
                if check_collision(asteroid, planet):
                    collision_info['asteroid_planet'] = planet
                    collision_info['earth_collision'] = (planet.name == "Earth")
                    running = False  # End simulation immediately on collision
                    break

        time_elapsed += CelestialBody.TIMESTEP
        clock.tick(60)

        # Visualization
        WIN.fill((0, 0, 0))
        distance_to_center = math.sqrt(asteroid.x**2 + asteroid.y**2)
        scale_factor = max(0.5, 1 - distance_to_center / (2 * CelestialBody.AU))
        offset_x = WIDTH/2 - asteroid.x * CelestialBody.SCALE * scale_factor
        offset_y = HEIGHT/2 - asteroid.y * CelestialBody.SCALE * scale_factor

        for body in all_bodies:
            body.draw(WIN, scale_factor, offset_x, offset_y)

        time_text = FONT.render(f"Time: {time_elapsed / CelestialBody.TIMESTEP:.1f} days", True, WHITE)
        WIN.blit(time_text, (10, 10))
        pygame.display.update()

    collision_info['min_distance'] = int(min_distance)
    return collision_info

def calculate_reward(collision_info, delta_vx, delta_vy, dis):
    """
    Compute a reward for the episode.
    Reward is positive only if:
      - The spacecraft hit the asteroid (deflection occurred)
      - The asteroid did NOT hit Earth (or another planet) afterward.
    A bonus based on the magnitude of the initial delta‑v (i.e. fuel cost) is applied.
    Additionally, a bonus is provided for greater deflection efficiency (larger min distance).
    """
    delta_v = math.sqrt(delta_vx**2 + delta_vy**2)

    if collision_info['spacecraft_asteroid'] == True:
        reward = 10
        if collision_info['earth_collision'] == False and collision_info['asteroid_planet'].name != "Earth":
            reward = 25
            if collision_info['asteroid_planet'].name == "Sun":
                reward = 30
        reward -= max(0, 5.0 - delta_v/1000)

    elif collision_info['earth_collision']:
        reward = -1 * (dis / 1e10)
    elif collision_info['spacecraft_planet']:
        reward = -1 * (dis / 1e10)
    elif not collision_info['spacecraft_asteroid']:
        print("b")
        reward = -1 * (dis / 1e10)
    return reward

#############################
# RL Environment for Initial Velocity
#############################
X_LOW = -2 * CelestialBody.AU
X_HIGH = 2 * CelestialBody.AU
Y_LOW = -2 * CelestialBody.AU
Y_HIGH = 2 * CelestialBody.AU

class InitialVelocityEnv(gym.Env):
    """
    A minimal RL environment whose purpose is to let the AI choose the spacecraft's
    initial velocity, with the extended asteroid initial conditions as observation.
    Observation: [x, y, x_vel, y_vel, mass]
    Action: a 2D vector for delta‑v, with each component in [-2000, 2000] m/s.
    """
    def __init__(self):
        super(InitialVelocityEnv, self).__init__()
        self.observation_space = spaces.Box(
            low=np.array([X_LOW, Y_LOW, -5000, -5000, 1e9], dtype=np.float32),
            high=np.array([X_HIGH, Y_HIGH, 5000, 5000, 1e12], dtype=np.float32),
            dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=np.array([-2000, -2000], dtype=np.float32),
            high=np.array([2000, 2000], dtype=np.float32),
            dtype=np.float32
        )
        self.state = np.zeros(5, dtype=np.float32)

    def reset(self, asteroid_params):
        """
        asteroid_params: tuple of (x, y, x_vel, y_vel, mass)
        """
        self.state = np.array([asteroid_params[0], asteroid_params[1],
                                asteroid_params[2], asteroid_params[3],
                                asteroid_params[4]], dtype=np.float32)
        return self.state

    def step(self, action):
        reward = 0
        done = True
        return self.state, reward, done, {}

    def render(self, mode='human'):
        print(f"Asteroid state: {self.state}")

def process_action(action):
    """
    Given a continuous action (a numpy array of shape (2,)), ensure it is not (0,0).
    If the action is (0,0) (or extremely close), adjust it slightly.
    """
    if np.allclose(action, np.array([0.0, 0.0]), atol=1e-3):
        action[0] = 1.0
    return action[0], action[1]

#############################
# RL Agent with Saving & Loading
#############################
class RLAgent:
    """
    A simple RL agent that uses a best‑reward heuristic.
    It saves and loads its best actions (and rewards) for each asteroid state from disk to allow continuous learning.
    """
    def __init__(self):
        # Dictionary mapping asteroid key (x, y) to (reward, action)
        self.best_by_state = {}
        self.model_file = "agent_model.pkl"
        self.load_model()

    def load_model(self):
        if os.path.exists(self.model_file):
            with open(self.model_file, "rb") as f:
                data = pickle.load(f)
                self.best_by_state = data.get("best_by_state", {})
                print("Agent model loaded.")

    def save_model(self):
        data = {"best_by_state": self.best_by_state}
        with open(self.model_file, "wb") as f:
            pickle.dump(data, f)
        print("Agent model saved.")

    def choose_action(self, state):
        # Use (x, y) as a key. We round to 2 decimals to allow slight variations.
        key = (round(state[0], 2), round(state[1], 2))
        if key in self.best_by_state and random.random() < 0.5:
            return self.best_by_state[key][1]
        else:
            return np.array([random.uniform(-2000, 2000), random.uniform(-2000, 2000)], dtype=np.float32)

    def update(self, state, action, reward):
        key = (round(state[0], 2), round(state[1], 2))
        if key not in self.best_by_state or reward > self.best_by_state[key][0]:
            self.best_by_state[key] = (reward, action)

#############################
# Main Loop: Episodes, Training & Demo Modes
#############################
def main():
    # Determine mode from command-line arguments
    mode = "train"  # default mode
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()

    if mode not in ["train", "demo"]:
        print("Usage: python Asteroid_Deflection_Env.py [train|demo]")
        sys.exit(1)

    # Load asteroid parameters (from file positions.txt)
    asteroids = load_asteroid_parameters('positions.txt')
    if not asteroids:
        print("No asteroids found!")
        return

    if mode == "train":
        # Training mode: run many episodes and update the agent.
        episodes = 1000
        env = InitialVelocityEnv()
        agent = RLAgent()
        best_reward_overall = -float('inf')
        best_action_overall = None

        for episode in range(episodes):
            asteroid_params = random.choice(asteroids)
            if len(asteroid_params) == 4:
                mass = random.uniform(1e9, 1e12)
                asteroid_params = asteroid_params + (mass,)
            state = env.reset(asteroid_params)
            action = agent.choose_action(state)
            delta_vx, delta_vy = process_action(action)
            print(f"Episode {episode+1}: Chosen action -> delta_v = ({delta_vx:.2f}, {delta_vy:.2f}) m/s")
            
            collision_info = run_simulation_episode(asteroid_params, delta_vx, delta_vy)
            if collision_info is None:
                continue

            final_reward = calculate_reward(collision_info, delta_vx, delta_vy, collision_info['min_distance'])
            print(f"Episode {episode+1}: Final Reward {final_reward:.2f}")
        
            with open("reward_data.txt", "a") as file1:
                file1.write(f"{final_reward:.2f}\n")

            agent.update(state, action, final_reward)
            agent.save_model()

            if final_reward > best_reward_overall:
                best_reward_overall = final_reward
                best_action_overall = (delta_vx, delta_vy)

        print(f"\nBest reward overall: {best_reward_overall:.2f}")
        if best_action_overall is not None:
            print(f"Best overall initial delta-v: x = {best_action_overall[0]:.2f} m/s, y = {best_action_overall[1]:.2f} m/s")
        else:
            print("No valid episode completed.")

    elif mode == "demo":
        # Demo mode: load the agent and run a single simulation using the best action for the specific asteroid.
        env = InitialVelocityEnv()
        agent = RLAgent()
        asteroid_params = random.choice(asteroids)
        if len(asteroid_params) == 4:
            mass = random.uniform(1e9, 1e12)
            asteroid_params = asteroid_params + (mass,)
        state = env.reset(asteroid_params)
        key = (round(state[0], 2), round(state[1], 2))
        if key not in agent.best_by_state:
            print("No best action available for this asteroid. Please run training mode first.")
            return
        delta_vx, delta_vy = process_action(agent.best_by_state[key][1])
        print(f"Demo Mode: Using best action for asteroid at position ({state[0]:.2f}, {state[1]:.2f}) -> delta_v = ({delta_vx:.2f}, {delta_vy:.2f}) m/s")
        collision_info = run_simulation_episode(asteroid_params, delta_vx, delta_vy)
        print("Demo Mode: Simulation ended.")
        print("Collision Info:", collision_info)

    pygame.quit()

if __name__ == "__main__":
    main()
