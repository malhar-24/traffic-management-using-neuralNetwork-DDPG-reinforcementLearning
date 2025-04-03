import pygame
import sys
import math
import time

import random
import torch
import torch.nn as nn
import torch.optim as optim
import random

def generate_input():
    cars_in_lane = [random.randint(0, 100) for _ in range(4)]  # 4 values from 0-30
    wait_times = [0,0,0,0]
    return cars_in_lane + wait_times

def rotate_list(lst, n):
    return lst[n:] + lst[:n]

def rotate_update(wait_times, nn_output):
    selected_lane = 0  # Always the first lane
    green_time = nn_output[0]
    carsinlane = wait_times[:4]
    carswaittime = wait_times[4:8]

    cars_per_second = 1
    cars_passed = min(carsinlane[selected_lane], cars_per_second * green_time)
    carsinlane[selected_lane] -= cars_passed
    carswaittime[selected_lane] = 0

    for i in range(4):
        if i != selected_lane:
            if carsinlane[i] > 0:
                carswaittime[i] += green_time
            else:
                carswaittime[i] = 0

    carsinlane = rotate_list(carsinlane, 1)
    carswaittime = rotate_list(carswaittime, 1)
    return carsinlane + carswaittime
# Define the neural network
class TrafficNN(nn.Module):
    def __init__(self):
        super(TrafficNN, self).__init__()
        self.fc1 = nn.Linear(8, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))  # Output in range (0,1)
        return x * 120  # Rescale to 0-120
# Load the trained model
model2 = TrafficNN()
criterion = nn.MSELoss()
model2.load_state_dict(torch.load("traffic_nn2.pth"))
model2.eval()
listgreensq=[]
# Generate initial state (cars in lanes and wait times)
ini=generate_input()
inputcars=ini
print("Begin:")
while any(ini[:4]):
  test_state = torch.tensor([ini], dtype=torch.float32) / 100  # Normalize test input
  print(ini)
  test_state = test_state.view(1, -1)  # Ensure correct shape
  predicted_action = round(model2(test_state).item() * 120 ) # Rescale prediction
  listgreensq.append(predicted_action)
  print(f"Predicted Green Time: {predicted_action:.2f} seconds")
  ini=rotate_update(ini, [predicted_action])
print(ini)
print("End:")
print(listgreensq)




# Initialize Pygame
pygame.init()

# Set screen dimensions and load assets
screen = pygame.display.set_mode((600, 600))
pygame.display.set_caption("City Traffic Simulation")

# Load city map and car image
city_map = pygame.image.load('road1.jpg')  # Ensure correct image path
city_map = pygame.transform.scale(city_map, (600, 600))
car_image = pygame.image.load('car.png')  # Ensure correct image path
car_image = pygame.transform.scale(car_image, (25, 50))
car_image = pygame.transform.rotate(car_image, 270)  # Adjust initial orientation

# Define paths for cars
paths = {
    "r": [(596, 315), (283, 310), (285, 1)],
    "t": [(346, 3), (341, 253), (596, 254)],
    "b": [(282, 595), (282, 292), (255, 265), (254, 205), (256, 1)],
    "l": [(3, 253), (286, 256), (282, 2)]
}



# Traffic signal states for each direction
signals = {"r": "red", "t": "red", "b": "red", "l": "red"}

def is_too_close(car_pos, next_car_pos, min_distance):
    if next_car_pos is None:
        return False
    distance = math.hypot(car_pos['x'] - next_car_pos['x'], car_pos['y'] - next_car_pos['y'])
    return distance < min_distance

# Function to move a car along its specified path with signal and collision handling
def move_car_along_path(car_pos, path, speed, next_car_pos=None):
    if car_pos['index'] >= len(path) - 1:
        car_pos['active'] = False  # Deactivate car when it reaches the end
        return

    start = (car_pos['x'], car_pos['y'])
    end = path[car_pos['index'] + 1]

    signal_positions = {
        "r": [(395, 360),(390, 320)],
        "t": [(303, 210), (352, 200)],
        "b": [(245, 395), (297, 398)],
        "l": [(204, 243), (210, 297)]
    }

    car_lane = car_pos['lane']
    signal_start, signal_end = signal_positions[car_lane]

    if signal_positions[car_lane] == [(395, 360),(390, 320)]  :  
        if (
            signal_start[0] >= car_pos['x'] >= signal_end[0] 
            and signals[car_lane] == "red"
        ):
            return

    if signal_positions[car_lane] == [(204, 243), (210, 297)]  :  
        if (
            signal_start[0] <= car_pos['x'] <= signal_end[0] 
            and signals[car_lane] == "red"
        ):
            return

    if signal_positions[car_lane] == [(245, 395), (297, 398)]  :  
        if (
            signal_start[1] <= car_pos['y'] <= signal_end[1] 
            and signals[car_lane] == "red"
        ):
            return
    
    if signal_positions[car_lane] == [(303, 210), (352, 200)]  :  
        if (
            signal_start[1] >= car_pos['y'] >= signal_end[1] 
            and signals[car_lane] == "red"
        ):
            return

    if is_too_close(car_pos, next_car_pos, min_distance=60):
        return

    dx, dy = end[0] - start[0], end[1] - start[1]
    distance = math.hypot(dx, dy)
    step = speed / distance if distance != 0 else 0

    car_pos['x'] += step * dx
    car_pos['y'] += step * dy

    angle = math.degrees(math.atan2(-dy, dx))
    car_pos['angle'] = angle

    if distance <= speed:
        car_pos['index'] += 1

# Main loop setup
running = True
clock = pygame.time.Clock()
cars = []

c1,c2,c3,c4=inputcars[:4]
car_count = {
    "r": 0,  # Number of cars for right lane
    "b": 0,  # Number of cars for top lane
    "l": 0,  # Number of cars for bottom lane
    "t": 0   # Number of cars for left lane
}
car_count['r']=c1
car_count['b']=c2
car_count['l']=c3
car_count['t']=c4


change_interval = 0  # Change light every 30 seconds 


# Lane cycle order and timing
lanes = ["r", "b", "l", "t"]
current_lane = 3
last_green_time = time.time()

# Spawning cars based on the car_count variables
for lane, path in paths.items():
    for _ in range(car_count[lane]):
        cars.append({'x': path[0][0], 'y': path[0][1], 'index': 0, 'angle': 0, 'active': True, 'path': path, 'lane': lane})
index=0
# Main game loop
while running:
    screen.blit(city_map, (0, 0))

    if time.time() - last_green_time > change_interval:
        #predict new green time
        change_interval=listgreensq[index]*0.3
        signals[lanes[current_lane]] = "red"  # Turn previous lane red
        current_lane = (current_lane + 1) % len(lanes)
        signals[lanes[current_lane]] = "green"  # Turn new lane green
        index+=1
        last_green_time = time.time()
    
    cars = [car for car in cars if car['active']]  # Remove inactive cars
    for i, car in enumerate(cars):
        next_car = cars[i - 1] if i > 0 and cars[i - 1]['lane'] == car['lane'] else None
        move_car_along_path(car, car['path'], speed=4, next_car_pos=next_car)
        if car['active']:
            rotated_car = pygame.transform.rotate(car_image, car['angle'])
            car_rect = rotated_car.get_rect(center=(car['x'], car['y']))
            screen.blit(rotated_car, car_rect.topleft)
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_g:
                for key in signals:
                    signals[key] = "green"
            elif event.key == pygame.K_r:
                for key in signals:
                    signals[key] = "red"

    font = pygame.font.Font(None, 36)
    signal_text = font.render(f"Signal: {signals['l'].upper()}", True, (255, 0, 0) if signals['l'] == "red" else (0, 255, 0))
    screen.blit(signal_text, (30, 100))

    font = pygame.font.Font(None, 36)
    signal_text1= font.render(f"Signal: {signals['t'].upper()}", True, (255, 0, 0) if signals['t'] == "red" else (0, 255, 0))
    screen.blit(signal_text1, (420, 100))

    font = pygame.font.Font(None, 36)
    signal_text2 = font.render(f"Signal: {signals['b'].upper()}", True, (255, 0, 0) if signals['b'] == "red" else (0, 255, 0))
    screen.blit(signal_text2, (30, 400))

    font = pygame.font.Font(None, 36)
    signal_text3 = font.render(f"Signal: {signals['r'].upper()}", True, (255, 0, 0) if signals['r'] == "red" else (0, 255, 0))
    screen.blit(signal_text3, (420, 400))


    pygame.display.update()
    clock.tick(60)

pygame.quit()
sys.exit()
