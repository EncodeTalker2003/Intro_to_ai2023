import numpy as np
import optuna
from typing import List
from loadMap import tryToLoad
from simuScene import Scene2D
from answerLocalization import *
import answerLocalization as al
from utils import Particle
import argparse

PYTHON_PATH = "python"

class MontoCarloLocalization:
    def __init__(self, scene: Scene2D, num_samples: int, odometry: np.array, lidar_gt: np.array) -> None:
        self.scene = scene
        self.odometry = odometry
        self.lidar_gt = lidar_gt
        self.particles: List[Particle] = []
        self.num_samples = num_samples
    
    def __generate_uniform_particles(self):
        generated_particles = generate_uniform_particles(self.scene.walls, self.num_samples)
        assert len(generated_particles) == self.num_samples
        self.particles = generated_particles

    #@profile
    def __calc_particle_weights(self, iter: int):
        for p in self.particles:
            lidar_result = self.scene.lidar_sensor(p.position, p.theta)
            lidar_gt = self.lidar_gt[iter]
            p.weight = calculate_particle_weight(lidar_result, lidar_gt)
        self.__normalize_paritcle_weights()
        pass

    def __resample_particles(self):
        resampled_particles = resample_particles(self.scene.walls, self.particles)
        assert len(resampled_particles) == self.num_samples
        self.particles = resampled_particles

    def __normalize_paritcle_weights(self):
        """
        Normalize weights of particles to satisfy: sum(weights) = 1.0
        """
        cumsum = 0.0
        for p in self.particles:
            cumsum += p.weight
        for p in self.particles:
            p.weight /= (cumsum + 1e-6)
    
    def __get_odometry_update(self, iter: int):
        distance = np.linalg.norm(self.odometry[iter, :2] - self.odometry[iter-1, :2])
        dtheta = self.odometry[iter, 2] - self.odometry[iter-1, 2]
        return distance, dtheta
    
    def __get_estimate_result(self):
        return get_estimate_result(self.particles)
    
    #@profile
    def run_localization_gen(self):
        self.__generate_uniform_particles()
        self.__calc_particle_weights(iter=0)
        ### Particle Filter Main Loop ###
        for i in range(1, self.lidar_gt.shape[0]):
            # 按weight将particles从大到小排序
            self.particles.sort(key=Particle.get_weight, reverse=True)
            self.__resample_particles()
            
            yield self.particles
            
            distance, dtheta = self.__get_odometry_update(iter=i)
            for p in self.particles:
                p = apply_state_transition(p, distance, dtheta)
            self.__calc_particle_weights(i)
        self.particles.sort(key=Particle.get_weight, reverse=True)
        return self.__get_estimate_result()
    
    def run_localization(self):
        gen = self.run_localization_gen()
        try:
            while True:
                _ = next(gen)
        except StopIteration as e:
            return e.value

def optimal_error(test_idx):
    np.random.seed(3407)
    layout_name = 'layouts/office.lay'
    scene = Scene2D(tryToLoad(layout_name))
    odometry = np.load(f'data_q1/odom_{test_idx}.npy')
    lidar = np.load(f'data_q1/lidar_{test_idx}.npy')
    localizer = MontoCarloLocalization(scene, 500, odometry, lidar)
    result_particle = localizer.run_localization()
    x = np.abs(result_particle.theta - odometry[-1, 2]) / (2 * np.pi)
    x -= int(x)
    dtheta = 2 * np.pi * x
    dtheta = 2 * np.pi - dtheta if dtheta > np.pi else dtheta
    error = np.sqrt((result_particle.position[0]-odometry[-1, 0])**2+(result_particle.position[1]-odometry[-1, 1])**2+5*(dtheta)**2)
    return error

def obj(trail): 
    print("q1 Localization")
    expkk = trail.suggest_float("exp_k", -1, 0)
    sigma = trail.suggest_float("sigma", 0, 1)
    sigma_2 = trail.suggest_float("sigma_theta", 0, 1)
    al.exp_k = expkk
    al.sigma_x = sigma
    al.sigma_theta = sigma_2
    al.sigma_y = sigma
    errors = []
    for i in range(17):
        output = optimal_error(i)
        errors.append(output)
        # print(f"Case {i}, error={errors[-1]:.4f}", flush=True)
    result = np.exp(-np.mean(errors))
    # print(f"Result: {result:.4f}", flush=True)
    return result

stu = optuna.create_study(study_name = "q1", storage = "sqlite:///optimal_q1.db", load_if_exists = True, direction = "maximize", sampler = optuna.samplers.TPESampler(multivariate = True))
print(stu.best_params)
# stu.enqueue_trial({"exp_k": -0.75, "sigma": 0.5, "sigma_theta": 0.3})
stu.enqueue_trial({"exp_k": -0.8, "sigma": 0.5, "sigma_theta": 0.3})
stu.optimize(obj, n_trials = 1000)
print(stu.best_params)

