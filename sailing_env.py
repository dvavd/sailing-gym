import math
import pygame
import numpy as np
import gymnasium as gym
from gymnasium import spaces

class SailingEnv(gym.Env):
    """
    A simple 2D sailing environment in Gymnasium.
    The agent controls the sail angle to navigate a boat toward a target.
    """
    metadata = {
        "render_modes": ["human"],
        "render_fps": 30
    }

    def __init__(self, 
                 render_mode=None,
                 window_size=600,
                 max_episode_steps=500):
        super().__init__()

        # --- Environment parameters ---
        self.window_size = window_size
        self.max_episode_steps = max_episode_steps
        self.current_step = 0
        
        # Boat's maximum sail angle adjustments and orientation adjustments
        self.max_sail_angle = math.pi  # [-pi, pi]
        self.max_rudder_change = 0.1   # per step
        self.max_sail_change = 0.1     # per step
        
        # Wind parameters (could be made random or dynamic)
        self.wind_direction = 0.0      # angle in radians (0 = East)
        self.wind_strength = 0.05      # simplistic multiplier for boat's velocity

        # Boat's dynamic parameters
        self.boat_position = np.array([0.0, 0.0])
        self.boat_orientation = 0.0    # angle in radians
        self.boat_speed = 0.0         # forward speed
        self.sail_angle = 0.0

        # Target position
        self.target_position = np.array([2.0, 2.0])  # meters (arbitrary units)
        self.goal_tolerance = 0.2

        # --- Action & Observation Spaces ---
        # Actions = [rudder_delta, sail_angle_delta]
        #   rudder_delta: change the boat orientation slightly
        #   sail_angle_delta: change the sail angle
        self.action_space = spaces.Box(
            low=np.array([-self.max_rudder_change, -self.max_sail_change]),
            high=np.array([ self.max_rudder_change,  self.max_sail_change]),
            shape=(2,),
            dtype=np.float32
        )

        # Observations = [boat_x, boat_y, boat_orientation, sail_angle,
        #                 wind_direction, target_x, target_y]
        self.observation_space = spaces.Box(
            low=np.array([-np.inf, -np.inf, -math.pi, -math.pi, -math.pi, -np.inf, -np.inf]),
            high=np.array([ np.inf,  np.inf,  math.pi,  math.pi,  math.pi,  np.inf,  np.inf]),
            dtype=np.float32
        )

        # Rendering variables
        self.render_mode = render_mode
        self.screen = None
        self.clock = None
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0

        # Randomize boat start and orientation (optional):
        self.boat_position = np.random.uniform(low=-1.0, high=1.0, size=(2,))
        self.boat_orientation = np.random.uniform(low=-math.pi, high=math.pi)
        self.sail_angle = 0.0
        self.boat_speed = 0.0

        # Randomize target (optional):
        self.target_position = np.random.uniform(low=-2.0, high=2.0, size=(2,))
        
        # (Optionally) randomize wind too
        # self.wind_direction = np.random.uniform(low=-math.pi, high=math.pi)
        # self.wind_strength = 0.05

        obs = self._get_observation()
        return obs, {}

    def step(self, action):
        """
        action = [rudder_delta, sail_angle_delta]
        """
        rudder_delta, sail_delta = action
        
        # Update orientation
        self.boat_orientation += rudder_delta
        self.boat_orientation = self._normalize_angle(self.boat_orientation)
        
        # Update sail angle
        self.sail_angle += sail_delta
        # Clip sail angle to [-pi, pi]
        self.sail_angle = np.clip(self.sail_angle, -self.max_sail_angle, self.max_sail_angle)

        # Compute boat dynamics
        self._update_boat_dynamics()

        # Check if done
        distance_to_target = np.linalg.norm(self.boat_position - self.target_position)
        done = distance_to_target < self.goal_tolerance

        # Alternatively, we can have a max number of steps:
        truncated = self.current_step >= self.max_episode_steps

        # Reward: simple negative distance to target at each step
        # (Agent gets 0 if it reaches the target, else negative for larger distance)
        reward = -distance_to_target

        # Randomly adjust wind direction and strength
        # Direction changes are smaller when wind is stronger
        direction_change = np.random.normal(0, 0.1 / (self.wind_strength + 0.1))
        self.wind_direction += direction_change
        self.wind_direction = self._normalize_angle(self.wind_direction)
        
        # Wind strength changes are relative to current strength
        # Using log-normal distribution to ensure wind stays positive
        strength_multiplier = np.random.lognormal(mean=0.0, sigma=0.1)
        self.wind_strength *= strength_multiplier
        # Clip wind strength to reasonable bounds
        self.wind_strength = np.clip(self.wind_strength, 0.01, 0.2)

        self.current_step += 1

        obs = self._get_observation()

        return obs, reward, done, truncated, {}

    def _update_boat_dynamics(self):
        """
        Very simplified boat dynamics:
        - Boat speed depends on alignment of boat orientation with the wind and the sail angle.
        - Then update boat position.
        """
        # angle difference between boat and wind
        angle_diff = self._normalize_angle(self.wind_direction - self.boat_orientation)

        # Calculate boat speed based on wind and sail angles
        # No speed when sailing directly into the wind (within 45 degrees)
        if abs(angle_diff) < math.pi/4:
            self.boat_speed = 0
        else:
            # Optimal sail angle is halfway between boat direction and wind
            optimal_sail = angle_diff/2
            # Reduce speed based on how far sail is from optimal
            sail_efficiency = math.cos(self.sail_angle - optimal_sail)
            # Reduce speed when sailing too close to or directly downwind
            point_of_sail_efficiency = math.sin(abs(angle_diff))
            # Combine factors
            self.boat_speed = self.wind_strength * sail_efficiency * point_of_sail_efficiency

        # Update position
        delta_x = self.boat_speed * math.cos(self.boat_orientation)
        delta_y = self.boat_speed * math.sin(self.boat_orientation)
        self.boat_position += np.array([delta_x, delta_y])

    def _normalize_angle(self, angle):
        """Normalize angle to [-pi, pi]."""
        return (angle + math.pi) % (2 * math.pi) - math.pi

    def _get_observation(self):
        return np.array([
            self.boat_position[0],
            self.boat_position[1],
            self.boat_orientation,
            self.sail_angle,
            self.wind_direction,
            self.target_position[0],
            self.target_position[1]
        ], dtype=np.float32)

    def render(self):
        """
        Renders the environment using pygame.
        """
        if self.render_mode == "human":
            if self.screen is None:
                pygame.init()
                pygame.display.set_caption("2D Sailing Environment")
                self.screen = pygame.display.set_mode((self.window_size, self.window_size))
            if self.clock is None:
                self.clock = pygame.time.Clock()

            

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

            self.screen.fill((130, 200, 230))  # sky-blue background (arbitrary)
            
            # Coordinate transforms (map [-2,2] range to the window size)
            def world_to_screen(pos):
                scale = self.window_size / 4.0  # if we consider -2..2 world range
                x = int((pos[0] + 2.0) * scale)
                y = int((2.0 - pos[1]) * scale)  # invert y for screen
                return (x, y)

            # Display wind info
            font = pygame.font.Font(None, 36)
            wind_text = f"Wind: {self.wind_strength:.2f} @ {math.degrees(self.wind_direction):.0f}°"
            wind_surface = font.render(wind_text, True, (0, 0, 0))
            self.screen.blit(wind_surface, (10, 10))

            # Draw target
            target_screen = world_to_screen(self.target_position)
            pygame.draw.circle(self.screen, (200, 0, 0), target_screen, 5)

            # Draw boat
            boat_screen = world_to_screen(self.boat_position)
            boat_direction = self.boat_orientation
            
            # Define boat shape points relative to center
            boat_length = 20
            boat_width = 10
            
            # Calculate boat hull points
            bow = (
                boat_screen[0] + int(boat_length * math.cos(boat_direction)),
                boat_screen[1] - int(boat_length * math.sin(boat_direction))
            )
            stern = (
                boat_screen[0] - int((boat_length/2) * math.cos(boat_direction)),
                boat_screen[1] + int((boat_length/2) * math.sin(boat_direction))
            )
            port = (
                boat_screen[0] - int((boat_width/2) * math.sin(boat_direction)),
                boat_screen[1] - int((boat_width/2) * math.cos(boat_direction))
            )
            starboard = (
                boat_screen[0] + int((boat_width/2) * math.sin(boat_direction)),
                boat_screen[1] + int((boat_width/2) * math.cos(boat_direction))
            )
            
            # Draw boat hull as a sleek white polygon
            hull_points = [bow, starboard, stern, port]
            pygame.draw.polygon(self.screen, (200, 200, 240), hull_points)  # Modern white hull

            # Draw sail
            # Let's represent sail as a line from boat's center in orientation + sail_angle
            # Draw sail as a curved polygon to look more realistic
            sail_direction = self.boat_orientation + self.sail_angle
            sail_length = 20
            sail_width = 8
            
            # Calculate sail curve points
            curve_points = []
            for t in np.linspace(0, 1, 5):
                # Create a curved sail shape using quadratic bezier
                x = boat_screen[0] + sail_length * t * math.cos(sail_direction)
                y = boat_screen[1] - sail_length * t * math.sin(sail_direction)
                
                # Add curve to the sail
                curve_factor = math.sin(t * math.pi) * sail_width
                normal_x = -math.sin(sail_direction) * curve_factor
                normal_y = -math.cos(sail_direction) * curve_factor
                
                curve_points.append((int(x + normal_x), int(y + normal_y)))
            
            # Add points back to boat to complete the polygon
            curve_points.append(boat_screen)
            
            # Draw the sail as a filled polygon
            pygame.draw.polygon(self.screen, (255, 255, 255), curve_points)  # White sail
            # pygame.draw.polygon(self.screen, (128, 128, 128), curve_points, 1)  # Gray outline

            # Flip the display
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None
            self.clock = None
