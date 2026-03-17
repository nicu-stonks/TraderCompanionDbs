import pygame
import math
import random

# Initialize Pygame
pygame.init()

# Constants
WIDTH, HEIGHT = 800, 600
HALF_WIDTH, HALF_HEIGHT = WIDTH // 2, HEIGHT // 2
FPS = 60

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
GRAY = (100, 100, 100)
LIGHT_GRAY = (200, 200, 200)

# Player settings
PLAYER_POS = [WIDTH // 2, HEIGHT // 2]
PLAYER_ANGLE = 0
PLAYER_SPEED = 5
TURN_SPEED = 0.05

# Ray casting settings
FOV = math.pi / 3  # 60 degrees
HALF_FOV = FOV / 2
NUM_RAYS = 120
MAX_DEPTH = 800
DELTA_ANGLE = FOV / NUM_RAYS
DIST = NUM_RAYS / (2 * math.tan(HALF_FOV))
SCALE = WIDTH // NUM_RAYS
WALL_HEIGHT = 50  # Height of walls

# Map settings
MAP_SIZE = 10
TILE_SIZE = 100
MAP = [
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 1, 1, 0, 0, 1, 0, 0, 1],
    [1, 0, 1, 0, 0, 0, 1, 0, 0, 1],
    [1, 0, 1, 0, 0, 0, 1, 0, 0, 1],
    [1, 0, 1, 1, 1, 0, 1, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
]

class Player:
    def __init__(self):
        self.x, self.y = PLAYER_POS
        self.angle = PLAYER_ANGLE
        self.health = 100
        self.ammo = 30
        self.score = 0
        self.shooting = False
        self.shot_cooldown = 0
    
    def movement(self):
        keys = pygame.key.get_pressed()
        
        # Forward and backward movement
        if keys[pygame.K_w]:
            self.x += PLAYER_SPEED * math.cos(self.angle)
            self.y += PLAYER_SPEED * math.sin(self.angle)
        if keys[pygame.K_s]:
            self.x -= PLAYER_SPEED * math.cos(self.angle)
            self.y -= PLAYER_SPEED * math.sin(self.angle)
        
        # Strafing left and right
        if keys[pygame.K_a]:
            self.x += PLAYER_SPEED * math.cos(self.angle - math.pi/2)
            self.y += PLAYER_SPEED * math.sin(self.angle - math.pi/2)
        if keys[pygame.K_d]:
            self.x += PLAYER_SPEED * math.cos(self.angle + math.pi/2)
            self.y += PLAYER_SPEED * math.sin(self.angle + math.pi/2)
        
        # Check for collisions with walls
        col = int(self.x / TILE_SIZE)
        row = int(self.y / TILE_SIZE)
        
        if col >= 0 and col < MAP_SIZE and row >= 0 and row < MAP_SIZE:
            if MAP[row][col] == 1:
                # Move player back if they hit a wall
                if keys[pygame.K_w]:
                    self.x -= PLAYER_SPEED * math.cos(self.angle)
                    self.y -= PLAYER_SPEED * math.sin(self.angle)
                if keys[pygame.K_s]:
                    self.x += PLAYER_SPEED * math.cos(self.angle)
                    self.y += PLAYER_SPEED * math.sin(self.angle)
                if keys[pygame.K_a]:
                    self.x -= PLAYER_SPEED * math.cos(self.angle - math.pi/2)
                    self.y -= PLAYER_SPEED * math.sin(self.angle - math.pi/2)
                if keys[pygame.K_d]:
                    self.x -= PLAYER_SPEED * math.cos(self.angle + math.pi/2)
                    self.y -= PLAYER_SPEED * math.sin(self.angle + math.pi/2)
        
        # Handle mouse movement for looking around
        rel_x, rel_y = pygame.mouse.get_rel()
        self.angle += rel_x * TURN_SPEED * 0.5
        
        # Handle shooting
        if self.shot_cooldown > 0:
            self.shot_cooldown -= 1
            
        mouse_buttons = pygame.mouse.get_pressed()
        if mouse_buttons[0] and self.shot_cooldown == 0 and self.ammo > 0:
            self.shooting = True
            self.shot_cooldown = 10
            self.ammo -= 1
        else:
            self.shooting = False

class Enemy:
    def __init__(self):
        # Place enemy at random position (not on a wall)
        while True:
            col = random.randint(1, MAP_SIZE - 2)
            row = random.randint(1, MAP_SIZE - 2)
            if MAP[row][col] == 0:
                self.x = col * TILE_SIZE + TILE_SIZE // 2
                self.y = row * TILE_SIZE + TILE_SIZE // 2
                break
        
        self.angle = random.uniform(0, 2 * math.pi)
        self.sprite = pygame.Surface((20, 20))
        self.sprite.fill(RED)
        self.speed = 1
        self.size = 20
        self.health = 100
        self.active = True
    
    def update(self, player):
        if not self.active:
            return False
            
        # Simple AI: move toward player
        dx = player.x - self.x
        dy = player.y - self.y
        distance = math.sqrt(dx*dx + dy*dy)
        
        if distance > 50:  # Don't get too close
            self.x += self.speed * dx / distance
            self.y += self.speed * dy / distance
            
        # Check if player shot enemy
        if player.shooting:
            # Calculate direction of player's view
            player_dir_x = math.cos(player.angle)
            player_dir_y = math.sin(player.angle)
            
            # Check if enemy is in front of player
            enemy_angle = math.atan2(self.y - player.y, self.x - player.x)
            angle_diff = abs(player.angle - enemy_angle)
            angle_diff = min(angle_diff, 2 * math.pi - angle_diff)
            
            # If enemy is in front of player and within FOV/2
            if angle_diff < FOV / 4 and distance < 300:
                self.health -= 20
                if self.health <= 0:
                    self.active = False
                    return True
        
        return False

def ray_casting(player, enemies, screen):
    # Draw floor
    pygame.draw.rect(screen, GRAY, (0, HALF_HEIGHT, WIDTH, HALF_HEIGHT))
    
    # Draw ceiling
    pygame.draw.rect(screen, LIGHT_GRAY, (0, 0, WIDTH, HALF_HEIGHT))
    
    # Cast rays
    start_angle = player.angle - HALF_FOV
    
    for ray in range(NUM_RAYS):
        # Current ray angle
        current_angle = start_angle + ray * DELTA_ANGLE
        
        # Ray direction
        ray_dir_x = math.cos(current_angle)
        ray_dir_y = math.sin(current_angle)
        
        # Distance to wall
        distance_to_wall = 0
        wall_hit = False
        
        # Step size for ray casting
        step_x = ray_dir_x * 5
        step_y = ray_dir_y * 5
        
        # Current position
        current_x = player.x
        current_y = player.y
        
        # Cast ray until hit wall or max depth
        while not wall_hit and distance_to_wall < MAX_DEPTH:
            distance_to_wall += 5
            current_x += step_x
            current_y += step_y
            
            # Check if ray hit wall
            map_x = int(current_x / TILE_SIZE)
            map_y = int(current_y / TILE_SIZE)
            
            if map_x < 0 or map_x >= MAP_SIZE or map_y < 0 or map_y >= MAP_SIZE:
                wall_hit = True
                distance_to_wall = MAX_DEPTH
            elif MAP[map_y][map_x] == 1:
                wall_hit = True
        
        # Calculate wall height
        wall_height = min(int(DIST * WALL_HEIGHT / (distance_to_wall * math.cos(player.angle - current_angle) + 0.0001)), HEIGHT)
        
        # Draw wall slice
        wall_color = max(50, min(255, 255 - distance_to_wall // 4))
        pygame.draw.rect(screen, (wall_color, wall_color, wall_color), 
                        (ray * SCALE, HALF_HEIGHT - wall_height // 2, SCALE, wall_height))
    
    # Draw enemies
    for enemy in enemies:
        if not enemy.active:
            continue
            
        # Calculate direction and distance to enemy
        dx = enemy.x - player.x
        dy = enemy.y - player.y
        distance = math.sqrt(dx*dx + dy*dy)
        
        # Calculate angle to enemy
        enemy_angle = math.atan2(dy, dx)
        
        # Check if enemy is in front of player
        angle_diff = player.angle - enemy_angle
        if angle_diff > math.pi:
            angle_diff -= 2 * math.pi
        if angle_diff < -math.pi:
            angle_diff += 2 * math.pi
        
        # Only draw if enemy is in field of view
        if abs(angle_diff) < HALF_FOV:
            # Project enemy position to screen
            enemy_size = min(100, int(3000 / distance))
            enemy_x = HALF_WIDTH + int(angle_diff / FOV * WIDTH)
            enemy_y = HALF_HEIGHT
            
            # Draw enemy
            pygame.draw.circle(screen, RED, (enemy_x, enemy_y), enemy_size // 2)

def draw_minimap(player, enemies, screen):
    # Scale for minimap
    scale = 5
    
    # Draw map background
    pygame.draw.rect(screen, BLACK, (0, 0, MAP_SIZE * scale, MAP_SIZE * scale))
    
    # Draw walls
    for row in range(MAP_SIZE):
        for col in range(MAP_SIZE):
            if MAP[row][col] == 1:
                pygame.draw.rect(screen, WHITE, 
                                (col * scale, row * scale, scale, scale))
    
    # Draw player
    pygame.draw.circle(screen, GREEN, 
                      (int(player.x / TILE_SIZE * scale), int(player.y / TILE_SIZE * scale)), 2)
    
    # Draw player direction
    end_x = int(player.x / TILE_SIZE * scale + 3 * math.cos(player.angle))
    end_y = int(player.y / TILE_SIZE * scale + 3 * math.sin(player.angle))
    pygame.draw.line(screen, GREEN, 
                    (int(player.x / TILE_SIZE * scale), int(player.y / TILE_SIZE * scale)), 
                    (end_x, end_y), 1)
    
    # Draw enemies
    for enemy in enemies:
        if enemy.active:
            pygame.draw.circle(screen, RED, 
                              (int(enemy.x / TILE_SIZE * scale), int(enemy.y / TILE_SIZE * scale)), 1)

def draw_hud(player, screen):
    # Draw crosshair
    pygame.draw.line(screen, WHITE, (HALF_WIDTH - 10, HALF_HEIGHT), (HALF_WIDTH + 10, HALF_HEIGHT), 2)
    pygame.draw.line(screen, WHITE, (HALF_WIDTH, HALF_HEIGHT - 10), (HALF_WIDTH, HALF_HEIGHT + 10), 2)
    
    # Draw health bar
    health_width = int(player.health * 2)
    pygame.draw.rect(screen, RED, (20, HEIGHT - 30, 200, 20), 2)
    pygame.draw.rect(screen, RED, (20, HEIGHT - 30, health_width, 20))
    
    # Draw ammo
    font = pygame.font.Font(None, 36)
    ammo_text = font.render(f"Ammo: {player.ammo}", True, WHITE)
    screen.blit(ammo_text, (WIDTH - 150, HEIGHT - 30))
    
    # Draw score
    score_text = font.render(f"Score: {player.score}", True, WHITE)
    screen.blit(score_text, (20, 20))

def main():
    # Initialize the screen
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Python 3D FPS Game")
    clock = pygame.time.Clock()
    
    # Hide mouse cursor
    pygame.mouse.set_visible(False)
    pygame.event.set_grab(True)
    
    player = Player()
    enemies = [Enemy() for _ in range(5)]
    
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:  # Reload
                    player.ammo = 30
        
        # Update player
        player.movement()
        
        # Update enemies
        for enemy in enemies:
            if enemy.update(player):
                player.score += 10
        
        # Respawn enemies if all are inactive
        if all(not enemy.active for enemy in enemies):
            enemies = [Enemy() for _ in range(5)]
        
        # Clear screen
        screen.fill(BLACK)
        
        # Draw 3D view
        ray_casting(player, enemies, screen)
        
        # Draw HUD
        draw_hud(player, screen)
        
        # Draw minimap
        draw_minimap(player, enemies, screen)
        
        pygame.display.flip()
        clock.tick(FPS)
    
    pygame.quit()

if __name__ == "__main__":
    main()