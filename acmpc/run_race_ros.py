#!/usr/bin/env python3
"""Race a trained policy (train_race.py) through the gates in the ROS simulator.

The ROS sim (dodgeros, rotor-level model with its own low-level controller)
differs from the training simulator in race_env.py, so this is a sim-to-sim
transfer test, like the paper's evaluation in a higher-fidelity simulator.

The node flies the drone to the track start (go_to_pose), then sends the
policy's CTBR commands at 50 Hz, detects gate passes and crashes with the same
geometry as training, and prints lap times. Gates and the flown path are drawn
in rviz (/kingfisher/race/gates, /kingfisher/race/path).

Run everything with scripts/race_sim.sh, or by hand after taking off:
  python acmpc/run_race_ros.py --run acmpc/saved/race/MPC_splits_1
"""
import argparse
import os
import sys

import numpy as np
import rospy
from dodgeros_msgs.msg import Command, QuadState
from geometry_msgs.msg import Point, PoseStamped
from nav_msgs.msg import Path
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_race import load_run  # noqa: E402
from race_env import C_HALF, OMEGA_MAX, RunningMeanStd, quat_rot  # noqa: E402
from race_track import Track  # noqa: E402

QUAD = "/kingfisher"
PILOT = QUAD + "/dodgeros_pilot"


class RacePilot:
    def __init__(self, args):
        self.policy, rms_path, self.cfg = load_run(args.run)
        self.track = Track(args.track or self.cfg["track_path"])
        self.offset = np.array(args.offset, dtype=np.float64)
        self.laps = args.laps or self.cfg["laps"]
        self.timeout = args.timeout
        self.rms = RunningMeanStd(36)
        d = np.load(rms_path)
        self.rms.mean, self.rms.var = d["mean"], d["var"]

        self.state = None
        self.racing = False
        self.result = None
        self.cmd_pub = rospy.Publisher(PILOT + "/feedthrough_command", Command, queue_size=1)
        self.pose_pub = rospy.Publisher(PILOT + "/go_to_pose", PoseStamped, queue_size=1, latch=True)
        self.gate_pub = rospy.Publisher(QUAD + "/race/gates", MarkerArray, queue_size=1, latch=True)
        self.path_pub = rospy.Publisher(QUAD + "/race/path", Path, queue_size=1, latch=True)
        self.path = Path()
        self.path.header.frame_id = "world"
        rospy.Subscriber(PILOT + "/state", QuadState, self.state_cb, queue_size=1, tcp_nodelay=True)

    # ---- helpers ------------------------------------------------------------
    def drone(self):
        """State in the track frame: p, q(wxyz), v, w."""
        s = self.state
        p = np.array([s.pose.position.x, s.pose.position.y, s.pose.position.z]) - self.offset
        q = np.array([s.pose.orientation.w, s.pose.orientation.x, s.pose.orientation.y, s.pose.orientation.z])
        v = np.array([s.velocity.linear.x, s.velocity.linear.y, s.velocity.linear.z])
        w = np.array([s.velocity.angular.x, s.velocity.angular.y, s.velocity.angular.z])
        return p, q, v, w

    def observation(self, p, q, v, w):
        # Same layout and normalization as RaceEnv._obs
        tr = self.track
        nxt = tr.corners[self.target]
        after = tr.corners[(self.target + 1) % tr.num_gates]
        obs = np.concatenate([v, quat_rot(q[None])[0].reshape(9), (nxt - p).reshape(12), (after - nxt).reshape(12)])
        obs = np.clip((obs - self.rms.mean) / np.sqrt(self.rms.var + 1e-8), -10, 10)
        return np.concatenate([obs, p, q, v, w])[None].astype(np.float32)

    def publish_gates(self):
        arr = MarkerArray()
        for i, corners in enumerate(self.track.corners + self.offset):
            m = Marker()
            m.header.frame_id = "world"
            m.ns, m.id, m.type, m.action = "gates", i, Marker.LINE_STRIP, Marker.ADD
            m.pose.orientation.w = 1.0
            m.scale.x = 0.1
            target = self.racing and i == self.target
            m.color = ColorRGBA(0.1, 0.9, 0.1, 1.0) if target else ColorRGBA(1.0, 0.45, 0.0, 1.0)
            m.points = [Point(*c) for c in np.vstack([corners, corners[:1]])]
            arr.markers.append(m)
            t = Marker()
            t.header.frame_id = "world"
            t.ns, t.id, t.type, t.action = "labels", i, Marker.TEXT_VIEW_FACING, Marker.ADD
            t.pose.position = Point(*(self.track.centers[i] + self.offset + [0, 0, 1.1]))
            t.pose.orientation.w = 1.0
            t.scale.z = 0.5
            t.color = ColorRGBA(1, 1, 1, 1)
            t.text = str(i + 1)
            arr.markers.append(t)
        self.gate_pub.publish(arr)

    def go_to_start(self):
        tr = self.track
        start = tr.start_pos + self.offset
        d = tr.centers[0] - tr.start_pos
        yaw = np.arctan2(d[1], d[0])
        msg = PoseStamped()
        msg.header.frame_id = "world"
        msg.pose.position = Point(*start)
        msg.pose.orientation.w, msg.pose.orientation.z = np.cos(yaw / 2), np.sin(yaw / 2)
        self.pose_pub.publish(msg)
        rospy.loginfo("Flying to the track start %s", np.round(start, 2))
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            p, _, v, _ = self.drone()
            if np.linalg.norm(p - tr.start_pos) < 0.2 and np.linalg.norm(v) < 0.2:
                break
            rate.sleep()
        rospy.sleep(1.0)

    # ---- race -----------------------------------------------------------------
    def start_race(self):
        p, _, _, _ = self.drone()
        self.target, self.passed, self.gate_times = 0, 0, []
        self.p_prev, self.dist, self.v_max = p, 0.0, 0.0
        self.t0 = self.t_cmd = self.state.t
        self.racing = True
        self.publish_gates()
        rospy.loginfo("GO: %d laps of %s with %s", self.laps, self.track.name, self.cfg["name"])

    def state_cb(self, msg):
        self.state = msg
        if not self.racing:
            return
        p, q, v, w = self.drone()
        t = msg.t - self.t0
        passed, hit = self.track.crossing(self.p_prev[None], p[None])
        self.dist += np.linalg.norm(p - self.p_prev)
        self.v_max = max(self.v_max, np.linalg.norm(v))
        self.p_prev = p
        self.path.poses.append(PoseStamped(pose=msg.pose))
        self.path.poses[-1].header.frame_id = "world"

        if passed[0, self.target]:
            self.passed += 1
            self.gate_times.append(t)
            rospy.loginfo("gate %d passed at %.2f s (%.1f m/s)", self.target + 1, t, np.linalg.norm(v))
            self.target = (self.target + 1) % self.track.num_gates
            self.publish_gates()
        out = np.any((p < self.track.bounds[0]) | (p > self.track.bounds[1]))
        if hit.any() or out:
            return self.finish("CRASH ({0})".format("gate frame" if hit.any() else "left the arena"), t)
        if self.passed >= self.laps * self.track.num_gates:
            return self.finish("FINISHED", t)
        if t > self.timeout:
            return self.finish("TIMEOUT", t)

        # 50 Hz control, the rate the policy was trained at
        if msg.t - self.t_cmd < 0.02 - 1e-4:
            return
        self.t_cmd = msg.t
        action, _ = self.policy.predict(self.observation(p, q, v, w), deterministic=True)
        a = np.clip(action[0], -1, 1)
        cmd = Command()
        cmd.t = msg.t
        cmd.header.stamp = rospy.Time(msg.t)
        cmd.is_single_rotor_thrust = False
        cmd.collective_thrust = float((a[0] + 1) * C_HALF)
        cmd.bodyrates.x, cmd.bodyrates.y, cmd.bodyrates.z = (a[1:] * OMEGA_MAX).tolist()
        self.cmd_pub.publish(cmd)
        if len(self.path.poses) % 10 == 0:
            self.path_pub.publish(self.path)

    def finish(self, status, t):
        self.racing = False
        self.path_pub.publish(self.path)
        self.publish_gates()
        n = self.track.num_gates
        laps = [self.gate_times[k * n + n - 1] - (self.gate_times[k * n - 1] if k else 0.0)
                for k in range(len(self.gate_times) // n)]
        self.result = dict(status=status, time=t, gates=self.passed, laps=laps,
                           avg_speed=self.dist / max(t, 1e-6), max_speed=self.v_max)
        rospy.loginfo("%s after %.2f s: %d gates, lap times %s, avg %.2f m/s, max %.2f m/s", status, t,
                      self.passed, ["%.2f" % x for x in laps], self.result["avg_speed"], self.v_max)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True, help="RUN_DIR or RUN_DIR:ITER from train_race.py")
    p.add_argument("--track", help="default: the training track")
    p.add_argument("--offset", type=float, nargs=3, default=(0.0, 0.0, 0.0),
                   help="track position in the sim world (must match race_sim.sh)")
    p.add_argument("--laps", type=int)
    p.add_argument("--trials", type=int, default=1)
    p.add_argument("--timeout", type=float, default=30.0)
    args = p.parse_args(rospy.myargv()[1:])

    rospy.init_node("race_pilot")
    pilot = RacePilot(args)
    rospy.loginfo("Waiting for the drone state (take off first: GUI 'Arm Bridge' + 'Start')")
    while pilot.state is None and not rospy.is_shutdown():
        rospy.sleep(0.1)
    pilot.publish_gates()
    results = []
    for k in range(args.trials):
        pilot.go_to_start()
        pilot.path.poses = []
        pilot.start_race()
        while pilot.racing and not rospy.is_shutdown():
            rospy.sleep(0.05)
        results.append(pilot.result)
        rospy.sleep(2.0)
    ok = [r for r in results if r and r["status"] == "FINISHED"]
    print("\n{0}: {1}/{2} finished".format(pilot.cfg["name"], len(ok), len(results)))
    for r in results:
        print("  {status:10s} t={time:.2f}s gates={gates} laps={laps} avg={avg_speed:.2f} m/s max={max_speed:.2f} m/s"
              .format(**dict(r, laps=["%.2f" % x for x in r["laps"]])))


if __name__ == "__main__":
    main()
