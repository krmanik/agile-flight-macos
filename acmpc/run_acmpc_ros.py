#!/usr/bin/env python3
"""Fly a trained ACMPC policy in the ROS simulator (CTBR feedthrough commands).

Drop-in replacement for envtest/ros/run_competition.py in the upstream workflow
(state-based mode). Start the simulator and evaluation_node.py as usual, then:

  python acmpc/run_acmpc_ros.py --run acmpc/saved/MPC_1 --iter 500
  rostopic pub /kingfisher/start_navigation std_msgs/Empty "{}" -1
"""
import argparse
import os
import sys

import numpy as np
import rospy
import torch
from ruamel.yaml import YAML
from scipy.spatial.transform import Rotation

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.environ["FLIGHTMARE_PATH"], "..", "envtest", "ros"))
from acmpc_policy import MlpMpcPolicy, MlpOnlyPolicy  # noqa: E402
from run_competition import AgilePilotNode  # noqa: E402
from utils import AgileCommand, AgileCommandMode  # noqa: E402

POLICIES = {"MPC": MlpMpcPolicy, "MLP": MlpOnlyPolicy}


class ACMPCController:
    def __init__(self, run_dir, iteration):
        cfg = YAML().load(open(os.path.join(run_dir, "config.yaml")))
        dyn = cfg["quadrotor_dynamics"]
        self.goal_vel = np.array(cfg["environment"]["goal_vel"], dtype=np.float64)
        # Same action scaling as VisionEnv (see acmpc_env.py)
        thrust_max = 4 * dyn["thrust_map"][0] * dyn["motor_omega_max"] ** 2
        half = thrust_max / dyn["mass"] / 2
        self.act_mean = np.array([half, 0.0, 0.0, 0.0])
        self.act_std = np.array([half, *dyn["omega_max"]])

        rms = np.load(os.path.join(run_dir, "RMS/iter_{0:05d}.npz".format(iteration)))
        self.obs_mean = np.mean(rms["mean"], axis=0)
        self.obs_var = np.mean(rms["var"], axis=0)

        # Run directories are named <MPC|MLP>_<n> by train_acmpc.py
        kind = os.path.basename(os.path.normpath(run_dir)).split("_")[0]
        os.environ.setdefault("ACMPC_T", "5")
        # Own checkpoint: SB3's policy.save() pickles the constructor args
        saved = torch.load(os.path.join(run_dir, "Policy/iter_{0:05d}.pth".format(iteration)),
                           map_location="cpu", weights_only=False)
        self.policy = POLICIES[kind](**saved["data"])
        self.policy.load_state_dict(saved["state_dict"])
        self.policy.set_training_mode(False)

    def command(self, state, obstacles):
        obstacle_obs = []
        for o in obstacles.obstacles:
            obstacle_obs += [o.position.x, o.position.y, o.position.z, o.scale]
        # Rotation matrix flattened column-major, as Eigen does in VisionEnv::getObs
        rot = Rotation.from_quat([state.att[1], state.att[2], state.att[3], state.att[0]])
        obs = np.concatenate([self.goal_vel, rot.as_matrix().reshape(9, order="F"),
                              state.vel, obstacle_obs])
        obs = (obs - self.obs_mean) / np.sqrt(self.obs_var + 1e-8)
        drone_state = np.concatenate([state.pos, state.att, state.vel, state.omega])

        action, _ = self.policy.predict(np.concatenate([obs, drone_state])[None], deterministic=True)
        action = action[0] * self.act_std + self.act_mean

        command = AgileCommand(AgileCommandMode.CTBR)
        command.t = state.t
        command.collective_thrust = float(action[0])
        command.bodyrates = action[1:4].tolist()
        return command


class ACMPCPilotNode(AgilePilotNode):
    def __init__(self, controller):
        self.controller = controller
        super().__init__(vision_based=False)

    def obstacle_callback(self, obs_data):
        if self.state is None:
            return
        self.publish_command(self.controller.command(self.state, obs_data))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, help="training run, e.g. acmpc/saved/MPC_1")
    parser.add_argument("--iter", type=int, required=True, help="checkpoint iteration")
    args = parser.parse_args(rospy.myargv()[1:])
    ACMPCPilotNode(ACMPCController(args.run, args.iter))
    rospy.spin()
