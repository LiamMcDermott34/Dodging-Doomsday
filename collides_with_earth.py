import csv
import pygame
import math
import random


pygame.init()

# Screen settings
WIDTH, HEIGHT = 1200, 800
WIN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Accurate Solar System Simulation")

# Colors
WHITE = (255, 255, 255)
YELLOW = (255, 255, 0)
BLUE = (100, 149, 237)
RED = (188, 39, 50)
DARK_GREY = (80, 78, 81)
ORANGE = (255, 165, 0)
LIGHT_BLUE = (173, 216, 230)
GREEN = (0, 255, 0)
PURPLE = (128, 0, 128)

FONT = pygame.font.SysFont("comicsans", 16)


class Planet:
    AU = 149.6e6 * 1000  # Astronomical unit in meters
    G = 6.67428e-11      # Gravitational constant
    SCALE = 250 / AU     # Scale for display: 1 AU = 100 pixels
    TIMESTEP = 3600 * 24 # One day in seconds

    def __init__(self, x, y, radius, color, mass, name, sun=False):
        self.initial_x = x  # Store initial x position
        self.initial_y = y  # Store initial y position
        self.x = x
        self.y = y
        self.radius = radius
        self.color = color
        self.mass = mass
        self.name = name
        self.sun = sun
        self.orbit = []
        self.distance_to_sun = 0
        self.x_vel = 0
        self.y_vel = 0

        if not self.sun:
            self.calculate_initial_velocity()

    def calculate_initial_velocity(self):
        self.distance_to_sun = math.sqrt(self.x**2 + self.y**2)
        if self.distance_to_sun > 0:
            v = math.sqrt(Planet.G * 1.98892e30 / self.distance_to_sun)
            self.x_vel = -v * (self.y / self.distance_to_sun)
            self.y_vel = v * (self.x / self.distance_to_sun)

    def draw(self, win, scale_factor, offset_x, offset_y):
        x = self.x * Planet.SCALE * scale_factor + offset_x
        y = self.y * Planet.SCALE * scale_factor + offset_y

        if len(self.orbit) > 2:
            updated_points = []
            for point in self.orbit:
                x_p, y_p = point
                x_p = x_p * Planet.SCALE * scale_factor + offset_x
                y_p = y_p * Planet.SCALE * scale_factor + offset_y
                updated_points.append((x_p, y_p))
            pygame.draw.lines(win, self.color, False, updated_points, 2)

        pygame.draw.circle(win, self.color, (int(x), int(y)), self.radius * scale_factor)

    def attraction(self, other):
        distance_x = other.x - self.x
        distance_y = other.y - self.y
        distance = math.sqrt(distance_x**2 + distance_y**2)
        if distance == 0:
            return 0, 0
        force = self.G * self.mass * other.mass / distance**2
        theta = math.atan2(distance_y, distance_x)
        return math.cos(theta) * force, math.sin(theta) * force

    def update_position(self, planets):
        total_fx = total_fy = 0
        for planet in planets:
            if self == planet:
                continue
            fx, fy = self.attraction(planet)
            total_fx += fx
            total_fy += fy
        self.x_vel += total_fx / self.mass * self.TIMESTEP
        self.y_vel += total_fy / self.mass * self.TIMESTEP
        self.x += self.x_vel * self.TIMESTEP
        self.y += self.y_vel * self.TIMESTEP
        self.orbit.append((self.x, self.y))

def spawn_asteroid():
    x = random.uniform(-15 * Planet.AU, 15 * Planet.AU)
    y = random.uniform(-15 * Planet.AU, 15 * Planet.AU)
    x_vel = random.uniform(-5e2, 5e2) #(-5e3, 5e3)
    y_vel = random.uniform(-5e2, 5e2) #(-5e3, 5e3)
    asteroid = Planet(x, y, 5, GREEN, 1e10, "Asteroid")
    asteroid.x_vel = x_vel
    asteroid.y_vel = y_vel
    return asteroid

def check_collision(asteroid, planets, time_elapsed):
    for planet in planets:
        distance = math.sqrt((asteroid.x - planet.x) ** 2 + (asteroid.y - planet.y) ** 2)
        scaled_distance = distance * Planet.SCALE
        if scaled_distance < planet.radius + asteroid.radius:
            return planet, time_elapsed
    return None, None

def run_single_simulation(planets, asteroid):
    run = True
    clock = pygame.time.Clock()
    time_elapsed = 0
    max_time_days = 365 * 100  # Simulation duration in days
    max_time = max_time_days * Planet.TIMESTEP
    collision = None

    while run:
        clock.tick(60)
        time_elapsed += Planet.TIMESTEP
        WIN.fill((0, 0, 0))

        # Calculate scale and offset based on asteroid's position
        distance_to_center = math.sqrt(asteroid.x**2 + asteroid.y**2)
        scale_factor = max(0.5, 1 - distance_to_center / (2 * Planet.AU))
        offset_x = WIDTH / 2 - asteroid.x * Planet.SCALE * scale_factor
        offset_y = HEIGHT / 2 - asteroid.y * Planet.SCALE * scale_factor

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                pygame.quit()
                return None
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    Planet.TIMESTEP *= 2
                elif event.key == pygame.K_DOWN:
                    Planet.TIMESTEP /= 2

        asteroid.update_position(planets)
        asteroid.draw(WIN, scale_factor, offset_x, offset_y)

        for planet in planets:
            planet.update_position(planets)
            planet.draw(WIN, scale_factor, offset_x, offset_y)

        collision, collision_time = check_collision(asteroid, planets, time_elapsed)
        if collision or time_elapsed > max_time:
            run = False
        
        time_text = FONT.render(f"Time: {time_elapsed / Planet.TIMESTEP:.2f} days | Timestep: {Planet.TIMESTEP:.2f}s", True, WHITE)
        WIN.blit(time_text, (10, 10))
        

        pygame.display.update()
    return collision

def main():
    required_collisions = 8  # Set the desired number of collisions
    collision_count = 0
    collision_data = []
    
    while collision_count < required_collisions:
        # Reinitialize the planets each simulation
        sun = Planet(0, 0, 30, YELLOW, 1.98892 * 10**30, "Sun", sun=True)
        planets = [
            sun,
            Planet(-1 * Planet.AU, 0, 16, BLUE, 5.9742 * 10**24, "Earth"),
            Planet(-1.524 * Planet.AU, 0, 12, RED, 6.39 * 10**23, "Mars"),
            Planet(0.387 * Planet.AU, 0, 8, DARK_GREY, 3.30 * 10**23, "Mercury"),
            Planet(0.723 * Planet.AU, 0, 14, WHITE, 4.8685 * 10**24, "Venus"),
            Planet(-5.2 * Planet.AU, 0, 20, ORANGE, 1.898 * 10**27, "Jupiter"),
            Planet(-9.5 * Planet.AU, 0, 18, LIGHT_BLUE, 5.683 * 10**26, "Saturn"),
            Planet(-19.8 * Planet.AU, 0, 17, PURPLE, 8.681 * 10**25, "Uranus"),
            Planet(-30.1 * Planet.AU, 0, 16, LIGHT_BLUE, 1.024 * 10**26, "Neptune"),
        ]
        asteroid = spawn_asteroid()
        collision_data = [asteroid.x,asteroid.y,asteroid.x_vel,asteroid.y_vel]
        collided_planet = run_single_simulation(planets, asteroid)
        
        row = [collided_planet.name,collision_data[0],collision_data[1],collision_data[2],collision_data[3]]

        with open("Collision_Data.csv", mode="a", newline="") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(row)

        if collided_planet and collided_planet.name == "Earth":
            with open("positions.txt", mode="a") as file:
                file.write(f"{collision_data[0]}, {collision_data[1]}, {collision_data[2]}, {collision_data[3]}\n")
            collision_count += 1
            print(f"Collision {collision_count} recorded with Earth.")

    pygame.quit()

if __name__ == "__main__":
    main()