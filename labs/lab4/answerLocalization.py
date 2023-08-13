from typing import List
import numpy as np
from utils import Particle
import math

### 可以在这里写下一些你需要的变量和函数 ###
COLLISION_DISTANCE = 1
MAX_ERROR = 50000

### 可以在这里写下一些你需要的变量和函数 ###

sigma_x = 0.08
sigma_y = 0.08
sigma_theta = 0.06
exp_k = -1.2

def valid_pos(walls, x, y):
    x_min, x_max, y_min, y_max = walls[:, 0].min(), walls[:, 0].max(), walls[:, 1].min(), walls[:, 1].max()
    if x < x_min or x > x_max or y < y_min or y > y_max:
        return False
    
    x_floor, y_floor = np.floor(x), np.floor(y)
    x_rst, y_rst = x - x_floor, y - y_floor
    occupied = []
    if x_rst < 0.75 and y_rst < 0.75:
        occupied.append([x_floor, y_floor])
    if x_rst < 0.75 and y_rst > 0.25:
        occupied.append([x_floor, y_floor + 1])
    if x_rst > 0.25 and y_rst < 0.75:
        occupied.append([x_floor + 1, y_floor])
    if x_rst > 0.25 and y_rst > 0.25:
        occupied.append([x_floor + 1, y_floor + 1])
    for block in occupied:
        for wall in walls:
            if block[0] == wall[0] and block[1] == wall[1]:
                return False
    
    return True 

def generate_uniform_particles(walls, N):
    """
    输入：
    walls: 维度为(xxx, 2)的np.array, 地图的墙壁信息，具体设定请看README关于地图的部分
    N: int, 采样点数量
    输出：
    particles: List[Particle], 返回在空地上均匀采样出的N个采样点的列表，每个点的权重都是1/N
    """
    # print("exp_k: ", exp_k)
    x_min, x_max, y_min, y_max = walls[:, 0].min(), walls[:, 0].max(), walls[:, 1].min(), walls[:, 1].max()
    x_max -= 0.75
    y_max -= 0.75
    x_min += 0.75
    y_min += 0.75
    all_particles: List[Particle] = []
    '''
    empty_block = []
    walls_list = walls.tolist()
    for x in range(int(x_min), int(x_max) + 1):
        for y in range(int(y_min), int(y_max) + 1):
            if not [x, y] in walls_list:
                empty_block.append([x, y])
    # print("Begin generating particles...")
    
    for _ in range(N):
        ind = np.random.randint(0, len(empty_block))
        x, y = empty_block[ind][0], empty_block[ind][1]
        x += np.random.uniform(-0.5, 0.5)
        y += np.random.uniform(-0.5, 0.5)
        theta = np.random.uniform(-np.pi, np.pi)
        all_particles.append(Particle(x, y, theta, 1.0 / N))
    # print("Generating particles finished!")
    '''
    for _ in range(N):
        while True:
            y = np.random.uniform(y_min, y_max)
            x = np.random.uniform(x_min, x_max)
            
            theta = np.random.uniform(-np.pi, np.pi)
            if valid_pos(walls, x, y):
                all_particles.append(Particle(x, y, theta, 1.0 / N))
                break
    return all_particles


def calculate_particle_weight(estimated, gt):
    """
    输入：
    estimated: np.array, 该采样点的距离传感器数据
    gt: np.array, Pacman实际位置的距离传感器数据
    输出：
    weight: float, 该采样点的权重
    """

    weight = np.linalg.norm(estimated - gt)
    weight = np.exp(weight * exp_k)
    return weight


def resample_particles(walls, particles: List[Particle]):
    """
    输入：
    walls: 维度为(xxx, 2)的np.array, 地图的墙壁信息，具体设定请看README关于地图的部分
    particles: List[Particle], 上一次采样得到的粒子，注意是按权重从大到小排列的
    输出：
    particles: List[Particle], 返回重采样后的N个采样点的列表
    """
    resampled_particles: List[Particle] = []
    N = len(particles)
    # print("Begin resampling...")
    for p in particles:
        num = int(N * 0.94 * p.weight)
        # sum += p.weight
        mu_x = p.position[0]
        mu_y = p.position[1]
        mu_theta = p.theta
        for _ in range(num):
            while True:
                now_x = np.random.normal(mu_x, sigma_x)
                now_y = np.random.normal(mu_y, sigma_y)
                now_theta = np.random.normal(mu_theta, sigma_theta)
                while now_theta > np.pi:
                    now_theta -= 2 * np.pi
                while now_theta < -np.pi:
                    now_theta += 2 * np.pi
                
                if valid_pos(walls, now_x, now_y):
                    resampled_particles.append(Particle(now_x, now_y, now_theta, 1.0 / N))
                    break
    # print("sum: ", sum)
    assert(len(resampled_particles) <= N)
    rest = N - len(resampled_particles)
    # print("resample unfinished: ", rest)
    rest_particles = generate_uniform_particles(walls, rest)
    for p in rest_particles:
        p.weight = 1.0 / N
    resampled_particles.extend(rest_particles)
    # print("Resampling finished!")
    return resampled_particles

def apply_state_transition(p: Particle, traveled_distance, dtheta):
    """
    输入：
    p: 采样的粒子
    traveled_distance, dtheta: ground truth的Pacman这一步相对于上一步运动方向改变了dtheta，并移动了traveled_distance的距离
    particle: 按照相同方式进行移动后的粒子
    """
    theta = p.theta + dtheta
    p.theta += dtheta
    p.position[0] += traveled_distance*math.cos(theta)
    p.position[1] += traveled_distance*math.sin(theta)
    while p.theta > np.pi:
        p.theta -= 2 * np.pi
    while p.theta < -np.pi:
        p.theta += 2 * np.pi
    return p


def get_estimate_result(particles: List[Particle]):
    """
    输入：
    particles: List[Particle], 全部采样粒子
    输出：
    final_result: Particle, 最终的猜测结果

    移动结束，根据当前估计的众多粒子得到最终结果，当然你也可以直接选权重最大的
    """
    final_result = None
    x, y, theta, prob = 0, 0, 0, 0
    for i in range(5):
        prob += particles[i].weight
    for i in range(5):
        x += particles[i].position[0] * particles[i].weight / prob
        y += particles[i].position[1] * particles[i].weight / prob
        theta += particles[i].theta * particles[i].weight / prob
    while theta > np.pi:
        theta -= 2 * np.pi
    while theta < -np.pi:
        theta += 2 * np.pi
    final_result = Particle(x, y, theta, 1.0)
    return final_result