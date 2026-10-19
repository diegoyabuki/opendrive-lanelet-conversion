#!/usr/bin/env python3
"""Demo: load a converted map in CARLA, run autopilot traffic and record it.

    python carla_demo.py output/l2/r01.xodr output/carla/demo [--seconds 40] [--vehicles 14]

Needs a running CARLA server (see docs/l2_to_odr_procedure.md §6), the matching carla
client wheel in the interpreter, and ffmpeg on PATH. Writes to <out_dir>:
  topdown_lanegraph.png        whole map, lane graph drawn (green lanes, red junction stubs,
                               orange dead ends)
  junction_zoom_lanegraph.png  the junction with most movements, 45 m above ground
  topdown_traffic.png          whole map with traffic, halfway through the run
  chase_still_1..3.png         chase-camera frames at 30 / 60 / 90 % of the run
  chase_cam.mp4 / .gif         chase camera behind an autopilot Tesla (10 fps; gif = first 12 s)
The CARLA window's spectator follows the same car, so the run can be watched live.
"""
import argparse
import glob
import math
import os
import random
import subprocess
import time

import carla

ap = argparse.ArgumentParser()
ap.add_argument("xodr")
ap.add_argument("out_dir")
ap.add_argument("--seconds", type=float, default=40.0)
ap.add_argument("--vehicles", type=int, default=14)
ap.add_argument("--port", type=int, default=2000)
a = ap.parse_args()
os.makedirs(a.out_dir, exist_ok=True)
frames_dir = os.path.join(a.out_dir, "frames")
os.makedirs(frames_dir, exist_ok=True)
for f in glob.glob(os.path.join(frames_dir, "*.png")):
    os.remove(f)
random.seed(7)


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


client = carla.Client("127.0.0.1", a.port)
client.set_timeout(120.0)
xodr = open(a.xodr).read()
world = client.generate_opendrive_world(xodr, carla.OpendriveGenerationParameters(
    vertex_distance=2.0, max_road_length=50.0, wall_height=0.0, additional_width=0.6,
    smooth_junctions=True, enable_mesh_visibility=True))
cmap = world.get_map()
log(f"map loaded: {cmap.name}")
world.set_weather(carla.WeatherParameters.ClearNoon)

settings = world.get_settings()
settings.synchronous_mode = True
settings.fixed_delta_seconds = 0.05
world.apply_settings(settings)
tm = client.get_trafficmanager(a.port + 6000)
tm.set_synchronous_mode(True)
tm.set_global_distance_to_leading_vehicle(3.0)

wps = cmap.generate_waypoints(2.0)
xs = [w.transform.location.x for w in wps]
ys = [w.transform.location.y for w in wps]
cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
ext = max(max(xs) - min(xs), max(ys) - min(ys))

GRAPH_SECONDS = 4.0


def draw_lane_graph(life):
    """ordinary lanes green, junction stubs red, dead ends orange."""
    for w in wps:
        nxt = w.next(2.0)
        for n in nxt:
            col = carla.Color(200, 30, 30) if (w.is_junction or n.is_junction) else carla.Color(30, 170, 60)
            world.debug.draw_line(w.transform.location + carla.Location(z=0.15),
                                  n.transform.location + carla.Location(z=0.15),
                                  thickness=0.08, color=col, life_time=life)
        if not nxt:
            world.debug.draw_line(w.transform.location + carla.Location(z=0.15),
                                  w.transform.location + carla.Location(z=1.5),
                                  thickness=0.3, color=carla.Color(255, 140, 0), life_time=life)

# --- traffic
bps = [b for b in world.get_blueprint_library().filter("vehicle.*")
       if int(b.get_attribute("number_of_wheels")) == 4 and "cybertruck" not in b.id]
cands = [w for w in wps if not w.is_junction and w.s > 6.0 and w.next(25.0)]
random.shuffle(cands)
vehicles = []
for w in cands:
    if len(vehicles) >= a.vehicles:
        break
    tf = w.transform
    tf.location.z += 0.3
    bp = random.choice(bps)
    if bp.has_attribute("color"):
        bp.set_attribute("color", random.choice(bp.get_attribute("color").recommended_values))
    v = world.try_spawn_actor(bp, tf)
    if v:
        vehicles.append(v)
        v.set_autopilot(True, tm.get_port())
        tm.vehicle_percentage_speed_difference(v, -10)
log(f"spawned {len(vehicles)} autopilot vehicles")


def respawn_missing():
    alive = [v for v in vehicles if v.is_alive]
    missing = len(vehicles) - len(alive)
    for v in list(vehicles):
        if not v.is_alive:
            vehicles.remove(v)
    tries = 0
    while missing > 0 and tries < 40:
        tries += 1
        w = random.choice(cands)
        tf = w.transform
        tf.location.z += 0.3
        v = world.try_spawn_actor(random.choice(bps), tf)
        if v:
            vehicles.append(v)
            v.set_autopilot(True, tm.get_port())
            missing -= 1


# --- cameras: chase cam attached to a hero, top-down fixed cam
cam_bp = world.get_blueprint_library().find("sensor.camera.rgb")
cam_bp.set_attribute("image_size_x", "1280")
cam_bp.set_attribute("image_size_y", "720")
cam_bp.set_attribute("fov", "90")
hero = vehicles[0]
sedan = world.get_blueprint_library().find("vehicle.tesla.model3")
for w in cands:
    tf = w.transform
    tf.location.z += 0.3
    v = world.try_spawn_actor(sedan, tf)
    if v:
        vehicles.append(v)
        v.set_autopilot(True, tm.get_port())
        hero = v
        break
chase = world.spawn_actor(cam_bp, carla.Transform(carla.Location(x=-7.0, z=3.2), carla.Rotation(pitch=-12)),
                          attach_to=hero, attachment_type=carla.AttachmentType.SpringArm)
chase_frames = []
chase.listen(lambda img: chase_frames.append(img))

top_bp = world.get_blueprint_library().find("sensor.camera.rgb")
top_bp.set_attribute("image_size_x", "2000")
top_bp.set_attribute("image_size_y", "2000")
top_bp.set_attribute("fov", "90")
top = world.spawn_actor(top_bp, carla.Transform(carla.Location(cx, cy, ext * 0.55 + 30), carla.Rotation(pitch=-90)))
top_frames = []
top.listen(lambda img: top_frames.append(img))
# zoomed view on the junction with most stubs
junc = {}
for w in wps:
    if w.is_junction:
        junc.setdefault(w.get_junction().id, []).append(w.transform.location)
jid, locs = max(junc.items(), key=lambda kv: len(kv[1]))
jx, jy = sum(l.x for l in locs) / len(locs), sum(l.y for l in locs) / len(locs)
zoom = world.spawn_actor(top_bp, carla.Transform(carla.Location(jx, jy, 45.0), carla.Rotation(pitch=-90)))
zoom_frames = []
zoom.listen(lambda img: zoom_frames.append(img))

# lane-graph overlay shots first (graph lines expire before the chase recording)
draw_lane_graph(GRAPH_SECONDS)
for _ in range(12):
    world.tick()
top_frames[-1].save_to_disk(os.path.join(a.out_dir, "topdown_lanegraph.png"))
zoom_frames[-1].save_to_disk(os.path.join(a.out_dir, "junction_zoom_lanegraph.png"))
zoom.stop()
zoom.destroy()
for _ in range(int(GRAPH_SECONDS / 0.05)):
    world.tick()
chase_frames.clear()
top_frames.clear()

spectator = world.get_spectator()
n_ticks = int(a.seconds / 0.05)
saved = 0
stills = {}
t0 = time.time()
for i in range(n_ticks):
    world.tick()
    if not hero.is_alive:
        alive = [v for v in vehicles if v.is_alive]
        if alive:
            hero = alive[0]
            chase.stop()
            chase.destroy()
            chase = world.spawn_actor(cam_bp, carla.Transform(carla.Location(x=-7.0, z=3.2), carla.Rotation(pitch=-12)),
                                      attach_to=hero, attachment_type=carla.AttachmentType.SpringArm)
            chase.listen(lambda img: chase_frames.append(img))
    if i % 20 == 0:
        respawn_missing()
    # spectator follows the hero in the CARLA window
    if hero.is_alive:
        ht = hero.get_transform()
        back = ht.get_forward_vector() * -9.0
        spectator.set_transform(carla.Transform(ht.location + carla.Location(back.x, back.y, 4.0),
                                                carla.Rotation(pitch=-15, yaw=ht.rotation.yaw)))
    if chase_frames and i % 2 == 0:          # 10 fps video
        img = chase_frames[-1]
        img.save_to_disk(os.path.join(frames_dir, f"chase_{saved:05d}.png"))
        saved += 1
    if i in (int(n_ticks * 0.3), int(n_ticks * 0.6), int(n_ticks * 0.9)) and chase_frames:
        chase_frames[-1].save_to_disk(os.path.join(a.out_dir, f"chase_still_{len(stills) + 1}.png"))
        stills[i] = True
    if i == n_ticks // 2 and top_frames:
        top_frames[-1].save_to_disk(os.path.join(a.out_dir, "topdown_traffic.png"))
    chase_frames.clear()
    top_frames.clear()
log(f"simulated {a.seconds:.0f} s in {time.time() - t0:.0f} s wall, {saved} video frames, "
    f"{len([v for v in vehicles if v.is_alive])} vehicles alive at the end")

chase.stop()
top.stop()
chase.destroy()
top.destroy()
for v in vehicles:
    if v.is_alive:
        v.set_autopilot(False, tm.get_port())
        v.destroy()
tm.set_synchronous_mode(False)
settings.synchronous_mode = False
settings.fixed_delta_seconds = None
world.apply_settings(settings)
tm.shut_down()

mp4 = os.path.join(a.out_dir, "chase_cam.mp4")
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", "10",
                "-i", os.path.join(frames_dir, "chase_%05d.png"),
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", mp4], check=True)
gif = os.path.join(a.out_dir, "chase_cam.gif")
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", mp4, "-t", "12",
                "-vf", "fps=8,scale=640:-1:flags=lanczos", gif], check=True)
for f in glob.glob(os.path.join(frames_dir, "*.png")):
    os.remove(f)
os.rmdir(frames_dir)
log(f"wrote {mp4} and {gif}")
