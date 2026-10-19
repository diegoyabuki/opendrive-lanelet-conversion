#!/usr/bin/env python3
"""Import an .xodr into a running CARLA server and check what CARLA made of it.

    python carla_import_check.py output/r01.xodr [--out-dir output/carla] [--vehicles 10] [--seconds 60]

Needs the ``carla`` Python client matching the server (not part of requirements.txt; e.g.
``uv pip install <CARLA>/PythonAPI/carla/dist/carla-0.9.16-cp310-*.whl``) and a server
listening on --host/--port (``./CarlaUE4.sh -RenderOffScreen``).

Checks (summary line each, FAIL exits 1, WARN does not):
  carla-01  generate_opendrive_world succeeds; roads/lanes/junctions as CARLA counts them
  carla-02  no non-finite waypoint coordinates
  carla-03  geoReference lat_0/lon_0 == CARLA geolocation of the origin
  carla-04  no duplicated topology edges (CARLA's reader duplicates them for connecting
            roads listed in several <connection>s; the Traffic Manager then segfaults)
  carla-05  lane ends without successor == --expected-dead-ends (if given)
  carla-06  Traffic Manager can be created (a crash here kills this process - see docs)
  carla-07  autopilot vehicles drive (WARN if the Traffic Manager removed one away from a dead end)
Also writes a top-down screenshot and a JSON report.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import os
import re
import sys
import time

try:
    import carla
except ImportError:  # pragma: no cover
    sys.exit("the 'carla' client package is not installed in this interpreter")

PASS, FAIL, SKIP, WARN = "PASS", "FAIL", "SKIP", "WARN"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def line(code, status, text):
    print(f"  {code:<9} {status:<4} {text}", flush=True)


def seg(wp):
    return (wp.road_id, wp.section_id, wp.lane_id)


def dead_end_lanes(cmap):
    last = {}
    for wp in cmap.generate_waypoints(1.0):
        k = (wp.road_id, wp.lane_id)
        s = wp.s if wp.lane_id < 0 else -wp.s
        if k not in last or s > last[k][0]:
            last[k] = (s, wp)
    return sorted(k for k, (s, wp) in last.items() if not wp.next(1.5))


def dead_end_ahead(wp, metres):
    frontier = [wp]
    for _ in range(int(metres / 2)):
        nxt = []
        for q in frontier:
            n = q.next(2.0)
            if not n:
                return True
            nxt += n
        frontier = nxt
        if not frontier:
            break
    return False


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("xodr")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=2000)
    ap.add_argument("--out-dir", default="output/carla")
    ap.add_argument("--vehicles", type=int, default=10)
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--expected-dead-ends", type=int, default=None,
                    help="number of source lanelets without successor")
    ap.add_argument("--no-drive", action="store_true", help="skip the Traffic Manager test")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(a.xodr))[0]
    rep = {"xodr": a.xodr}
    results = []

    with open(a.xodr) as f:
        xodr = f.read()
    m = re.search(r"\+lat_0=([-\d.]+).*?\+lon_0=([-\d.]+)", xodr)
    georef = (float(m.group(1)), float(m.group(2))) if m else None

    client = carla.Client(a.host, a.port)
    client.set_timeout(120.0)
    rep["server_version"], rep["client_version"] = client.get_server_version(), client.get_client_version()
    t0 = time.time()
    world = client.generate_opendrive_world(xodr, carla.OpendriveGenerationParameters(
        vertex_distance=2.0, max_road_length=50.0, wall_height=0.0, additional_width=0.6,
        smooth_junctions=True, enable_mesh_visibility=True))
    rep["generate_seconds"] = round(time.time() - t0, 1)
    cmap = world.get_map()
    wps = cmap.generate_waypoints(2.0)
    rep["roads"] = len({w.road_id for w in wps})
    rep["lanes"] = len({seg(w) for w in wps})
    rep["junctions"] = len({w.get_junction().id for w in wps if w.is_junction})
    results.append(("carla-01", PASS if wps else FAIL,
                    f"imported in {rep['generate_seconds']} s: {rep['roads']} roads, {rep['lanes']} lanes, "
                    f"{rep['junctions']} junctions, {len(wps)} waypoints @2 m (server {rep['server_version']})"))

    coords = [v for w in wps for v in (w.transform.location.x, w.transform.location.y, w.transform.location.z)]
    bad = sum(1 for v in coords if not math.isfinite(v))
    results.append(("carla-02", FAIL if bad else PASS, f"non-finite waypoint coordinates: {bad}"))
    xs, ys = [w.transform.location.x for w in wps], [w.transform.location.y for w in wps]
    rep["bbox"] = [min(xs), min(ys), max(xs), max(ys)]

    g0 = cmap.transform_to_geolocation(carla.Location(0, 0, 0))
    rep["geolocation_origin"] = [g0.latitude, g0.longitude]
    if georef:
        err = max(abs(g0.latitude - georef[0]), abs(g0.longitude - georef[1]))
        results.append(("carla-03", PASS if err < 1e-6 else FAIL, f"origin geolocation error {err:.2e} deg"))
    else:
        results.append(("carla-03", SKIP, "no +lat_0/+lon_0 in geoReference"))

    cnt = collections.Counter((seg(x), seg(y)) for x, y in cmap.get_topology() if seg(x) != seg(y))
    dups = [k for k, v in cnt.items() if v > 1]
    rep["topology_edges"], rep["duplicated_edges"] = sum(cnt.values()), len(dups)
    results.append(("carla-04", FAIL if dups else PASS,
                    f"topology edges {rep['topology_edges']}, duplicated {len(dups)} {dups[:3] if dups else ''}"))

    dead = dead_end_lanes(cmap)
    rep["dead_end_lanes"] = dead
    if a.expected_dead_ends is None:
        results.append(("carla-05", SKIP, f"lane ends without successor: {len(dead)} (pass --expected-dead-ends to check)"))
    else:
        results.append(("carla-05", PASS if len(dead) == a.expected_dead_ends else FAIL,
                        f"lane ends without successor: {len(dead)} (expected {a.expected_dead_ends})"))

    # screenshot
    settings = world.get_settings()
    settings.synchronous_mode, settings.fixed_delta_seconds = True, 0.05
    world.apply_settings(settings)
    cx, cy = (rep["bbox"][0] + rep["bbox"][2]) / 2, (rep["bbox"][1] + rep["bbox"][3]) / 2
    ext = max(rep["bbox"][2] - rep["bbox"][0], rep["bbox"][3] - rep["bbox"][1])
    bp = world.get_blueprint_library().find("sensor.camera.rgb")
    bp.set_attribute("image_size_x", "2000"), bp.set_attribute("image_size_y", "2000"), bp.set_attribute("fov", "90")
    cam = world.spawn_actor(bp, carla.Transform(carla.Location(cx, cy, ext * 0.6 + 50), carla.Rotation(pitch=-90)))
    shots = []
    cam.listen(shots.append)
    for _ in range(10):
        world.tick()
    cam.stop()
    if shots:
        rep["topdown_png"] = os.path.join(a.out_dir, f"topdown_{stem}.png")
        shots[-1].save_to_disk(rep["topdown_png"])
    cam.destroy()

    if not a.no_drive:
        tm = client.get_trafficmanager(a.port + 6000)   # a segfault here = Traffic Manager cannot index the map
        tm.set_synchronous_mode(True)
        results.append(("carla-06", PASS, "Traffic Manager created (map indexed)"))
        vbp = world.get_blueprint_library().find("vehicle.tesla.model3")
        cands = [w for w in wps if not w.is_junction and w.s > 6.0 and w.next(20.0)]
        vehicles = []
        for w in cands[::max(1, len(cands) // max(1, a.vehicles))][:a.vehicles]:
            tf = w.transform
            tf.location.z += 0.3
            v = world.try_spawn_actor(vbp, tf)
            if v:
                vehicles.append(v)
                v.set_autopilot(True, tm.get_port())
        info = {v.id: {"dist": 0.0, "last": None, "wp": None, "removed": None, "vmax": 0.0} for v in vehicles}
        for i in range(int(a.seconds / 0.05)):
            world.tick()
            for v in vehicles:
                d = info[v.id]
                if d["removed"] is not None:
                    continue
                if not v.is_alive:
                    d["removed"] = round(i * 0.05, 2)
                    continue
                loc, vel = v.get_location(), v.get_velocity()
                d["vmax"] = max(d["vmax"], math.hypot(vel.x, vel.y))
                if d["last"]:
                    d["dist"] += math.dist((loc.x, loc.y), (d["last"].x, d["last"].y))
                d["last"] = loc
                if i % 10 == 0:
                    d["wp"] = cmap.get_waypoint(loc)
        unexplained = []
        for v in vehicles:
            d = info[v.id]
            if d["removed"] is not None and not (d["wp"] and dead_end_ahead(d["wp"], 80)):
                unexplained.append((d["removed"], seg(d["wp"]) if d["wp"] else None, round(d["dist"], 1)))
        moved = sum(1 for v in vehicles if info[v.id]["dist"] > 5.0)
        rep["drive"] = {"vehicles": len(vehicles), "moved_over_5m": moved,
                        "removed_at_dead_end": sum(1 for v in vehicles if info[v.id]["removed"] is not None) - len(unexplained),
                        "removed_elsewhere": unexplained,
                        "distances_m": [round(info[v.id]["dist"]) for v in vehicles],
                        "max_speed_mps": [round(info[v.id]["vmax"], 1) for v in vehicles]}
        # Removals away from a dead end are usually the Traffic Manager snapping a vehicle to
        # the opposite lane of an overlapping/narrow two-way road (nearest waypoint, no
        # heading), not a map defect -> WARN, not FAIL.
        status = FAIL if not (vehicles and moved) else (WARN if unexplained else PASS)
        results.append(("carla-07", status,
                        f"{moved}/{len(vehicles)} vehicles drove; removed at dead ends "
                        f"{rep['drive']['removed_at_dead_end']}, elsewhere {len(unexplained)} {unexplained[:3] if unexplained else ''}"
                        f" (Traffic Manager removes vehicles reaching a dead end: OSM mode)"))
        for v in vehicles:
            if v.is_alive:
                v.set_autopilot(False, tm.get_port())
                v.destroy()
        tm.set_synchronous_mode(False)
        tm.shut_down()

    settings.synchronous_mode, settings.fixed_delta_seconds = False, None
    world.apply_settings(settings)

    for code, status, text in results:
        line(code, status, text)
    rep["results"] = [{"check": c, "status": s, "detail": t} for c, s, t in results]
    with open(os.path.join(a.out_dir, f"carla_report_{stem}.json"), "w") as f:
        json.dump(rep, f, indent=2)
    failed = [r for r in results if r[1] == FAIL]
    warned = [r for r in results if r[1] == WARN]
    print(f"CARLA check: {len(results) - len(failed) - len(warned)} PASS, {len(warned)} WARN, {len(failed)} FAIL "
          f"of {len(results)}; report {a.out_dir}/carla_report_{stem}.json")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
