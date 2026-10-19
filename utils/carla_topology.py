"""Rebuild OpenDRIVE road links and junctions from the lanelet graph for CARLA.

crdesigner's cr2odr writes junctions the way CommonRoad sees them: one <junction> per
incoming road, with the *successor lanelet roads* reused as connecting roads, several
incoming roads per connecting road, and the movements of opposite-direction lanes
(lane id +1) attached to the road end where the lane *starts* instead of where it exits.
OpenDRIVE tolerates most of that; CARLA does not:

* ``MapBuilder::GetJunctionLanes`` ignores ``<laneLink>`` and resolves junction movements
  from each connecting road's own ``<link>`` (road predecessor / successor) and lane
  ``<predecessor>``/``<successor>`` ids, once per ``<connection>`` entry.  A connecting
  road listed in k connections therefore yields k duplicated successor edges, and a
  connecting road with two incoming roads can only be reached from one of them.
* For lanes with id > 0 CARLA looks at the road *predecessor*, so movements attached to the
  successor side are dead ends.
* The Traffic Manager (``InMemoryMap``) only treats a connecting road as a junction when it
  sits between two ordinary roads and one of them has >= 2 junction paths; chained
  connecting roads and duplicated edges make it index an empty vector (segfault).

``rebuild_junctions`` throws cr2odr's links and junctions away and rebuilds them from the
lanelet succession graph:

* a lanelet edge A -> B that is the only successor of A and the only predecessor of B and
  whose road ends are not shared with any junction becomes a plain road/lane link;
* every other edge becomes its own short one-way *stub* road (the first ``stub_len`` m of
  B's lane, reversed for lanes with id > 0) that is the single connecting road of a single
  connection, with the incoming road as predecessor and B's road as successor;
* stubs sharing a road end are grouped into one <junction>; all other roads stay ordinary.

The stubs overlap the beginning of the lane they lead into (no road is shortened), which
keeps two-way roads intact; the overlap is at most ``stub_len`` metres.
"""

from __future__ import annotations

import copy
import math
from collections import defaultdict

from lxml import etree

LaneRef = tuple[str, int]          # (road id, lane id)
Edge = tuple[LaneRef, LaneRef]     # (from lane, to lane) in driving direction

DEFAULT_STUB_LEN = 2.0
STUB_MAX_FRACTION = 0.5


def _f(e: etree._Element, name: str, default: str = "0") -> float:
    return float(e.get(name, default))


def _fmt(v: float) -> str:
    return repr(float(v))


# --- polynomial / geometry helpers --------------------------------------------------------

def _poly_eval(c, t):
    return c[0] + c[1] * t + c[2] * t * t + c[3] * t ** 3


def _poly_shift(c, ds):
    """Coefficients of p(ds + t)."""
    a, b, cc, d = c
    return (a + b * ds + cc * ds ** 2 + d * ds ** 3, b + 2 * cc * ds + 3 * d * ds ** 2, cc + 3 * d * ds, d)


def _poly_scale(c, k):
    """Coefficients of p(k q)."""
    return tuple(ci * k ** i for i, ci in enumerate(c))


def _reframe(cu, cv):
    """Express the curve (cu, cv) in the frame of its start point and start tangent.

    Returns (cu', cv', theta) with theta the rotation applied (start tangent angle)."""
    u0, v0 = cu[0], cv[0]
    du, dv = cu[1], cv[1]
    if abs(du) + abs(dv) < 1e-12:  # degenerate tangent: use the chord
        du, dv = _poly_eval(cu, 1) - u0, _poly_eval(cv, 1) - v0
    theta = math.atan2(dv, du)
    ct, st = math.cos(theta), math.sin(theta)
    su = (cu[0] - u0, cu[1], cu[2], cu[3])
    sv = (cv[0] - v0, cv[1], cv[2], cv[3])
    cu2 = tuple(ct * a + st * b for a, b in zip(su, sv))
    cv2 = tuple(-st * a + ct * b for a, b in zip(su, sv))
    return cu2, cv2, theta


def _arc_length(cu, cv, n=400):
    total = 0.0
    px, py = cu[0], cv[0]
    for i in range(1, n + 1):
        q = i / n
        x, y = _poly_eval(cu, q), _poly_eval(cv, q)
        total += math.hypot(x - px, y - py)
        px, py = x, y
    return total


def _geom_coeffs(kind):
    cu = tuple(_f(kind, n) for n in ("aU", "bU", "cU", "dU"))
    cv = tuple(_f(kind, n) for n in ("aV", "bV", "cV", "dV"))
    return cu, cv


def _set_coeffs(kind, cu, cv):
    for n, val in zip(("aU", "bU", "cU", "dU"), cu):
        kind.set(n, _fmt(val))
    for n, val in zip(("aV", "bV", "cV", "dV"), cv):
        kind.set(n, _fmt(val))


def _global(x0, y0, hdg, u, v):
    return x0 + u * math.cos(hdg) - v * math.sin(hdg), y0 + u * math.sin(hdg) + v * math.cos(hdg)


def geometry_end(geom: etree._Element):
    """(x, y, hdg) at the end of a <geometry>; None for unsupported types."""
    kind = geom[0]
    x0, y0, hdg, length = _f(geom, "x"), _f(geom, "y"), _f(geom, "hdg"), _f(geom, "length")
    if kind.tag == "line":
        return x0 + math.cos(hdg) * length, y0 + math.sin(hdg) * length, hdg
    if kind.tag == "arc":
        k = _f(kind, "curvature")
        if k == 0:
            return x0 + math.cos(hdg) * length, y0 + math.sin(hdg) * length, hdg
        ang = k * length
        return (x0 + (math.sin(hdg + ang) - math.sin(hdg)) / k,
                y0 - (math.cos(hdg + ang) - math.cos(hdg)) / k, hdg + ang)
    if kind.tag == "paramPoly3" and kind.get("pRange", "normalized") == "normalized":
        cu, cv = _geom_coeffs(kind)
        u1, v1 = _poly_eval(cu, 1.0), _poly_eval(cv, 1.0)
        du, dv = cu[1] + 2 * cu[2] + 3 * cu[3], cv[1] + 2 * cv[2] + 3 * cv[3]
        x, y = _global(x0, y0, hdg, u1, v1)
        return x, y, hdg + math.atan2(dv, du)
    return None


def split_geometry(geom: etree._Element, ds: float):
    """Split one <geometry> at local arc length ds -> (first, second) copies, or None."""
    kind = geom[0]
    x0, y0, hdg, length = _f(geom, "x"), _f(geom, "y"), _f(geom, "hdg"), _f(geom, "length")
    first, second = copy.deepcopy(geom), copy.deepcopy(geom)
    if kind.tag == "line" or (kind.tag == "arc" and _f(kind, "curvature") == 0):
        first.set("length", _fmt(ds))
        second.set("length", _fmt(length - ds))
        second.set("x", _fmt(x0 + math.cos(hdg) * ds))
        second.set("y", _fmt(y0 + math.sin(hdg) * ds))
        return first, second
    if kind.tag == "arc":
        k = _f(kind, "curvature")
        ang = k * ds
        first.set("length", _fmt(ds))
        second.set("length", _fmt(length - ds))
        second.set("x", _fmt(x0 + (math.sin(hdg + ang) - math.sin(hdg)) / k))
        second.set("y", _fmt(y0 - (math.cos(hdg + ang) - math.cos(hdg)) / k))
        second.set("hdg", _fmt(hdg + ang))
        return first, second
    if kind.tag == "paramPoly3" and kind.get("pRange", "normalized") == "normalized":
        cu, cv = _geom_coeffs(kind)
        lo, hi = 0.0, 1.0
        for _ in range(50):
            mid = 0.5 * (lo + hi)
            if _arc_length(_poly_scale(cu, mid), _poly_scale(cv, mid), 200) < ds:
                lo = mid
            else:
                hi = mid
        p0 = 0.5 * (lo + hi)
        cu1, cv1 = _poly_scale(cu, p0), _poly_scale(cv, p0)
        su, sv = _poly_scale(_poly_shift(cu, p0), 1 - p0), _poly_scale(_poly_shift(cv, p0), 1 - p0)
        cu2, cv2, theta = _reframe(su, sv)
        _set_coeffs(first[0], cu1, cv1)
        first.set("length", _fmt(_arc_length(cu1, cv1)))
        _set_coeffs(second[0], cu2, cv2)
        second.set("length", _fmt(_arc_length(cu2, cv2)))
        x, y = _global(x0, y0, hdg, su[0], sv[0])
        second.set("x", _fmt(x)), second.set("y", _fmt(y)), second.set("hdg", _fmt(hdg + theta))
        return first, second
    return None


def reverse_geometry(geom: etree._Element):
    """Copy of a <geometry> traversed in the opposite direction, or None if unsupported."""
    end = geometry_end(geom)
    if end is None:
        return None
    kind = geom[0]
    rev = copy.deepcopy(geom)
    x1, y1, h1 = end
    if kind.tag == "line" or (kind.tag == "arc" and _f(kind, "curvature") == 0):
        rev.set("x", _fmt(x1)), rev.set("y", _fmt(y1)), rev.set("hdg", _fmt(h1 + math.pi))
        return rev
    if kind.tag == "arc":
        rev.set("x", _fmt(x1)), rev.set("y", _fmt(y1)), rev.set("hdg", _fmt(h1 + math.pi))
        rev[0].set("curvature", _fmt(-_f(kind, "curvature")))
        return rev
    if kind.tag == "paramPoly3":
        cu, cv = _geom_coeffs(kind)
        wu, wv = _poly_scale(_poly_shift(cu, 1.0), -1.0), _poly_scale(_poly_shift(cv, 1.0), -1.0)
        cu2, cv2, theta = _reframe(wu, wv)
        _set_coeffs(rev[0], cu2, cv2)
        rev.set("x", _fmt(x1)), rev.set("y", _fmt(y1)), rev.set("hdg", _fmt(_f(geom, "hdg") + theta))
        return rev
    return None


def reference_piece(road: etree._Element, s0: float, s1: float, reverse: bool) -> tuple[list, float]:
    """Geometries of ``road``'s reference line over [s0, s1] (re-based to s=0), optionally reversed."""
    geoms = sorted(road.findall("planView/geometry"), key=lambda g: _f(g, "s"))
    pieces = []
    for g in geoms:
        gs, gl = _f(g, "s"), _f(g, "length")
        ge = gs + gl
        if ge <= s0 + 1e-9 or gs >= s1 - 1e-9:
            continue
        piece = copy.deepcopy(g)
        if gs < s0 - 1e-9:
            cut = split_geometry(piece, s0 - gs)
            if cut is None:
                continue
            piece = cut[1]
            gs = s0
        if gs + _f(piece, "length") > s1 + 1e-9:
            cut = split_geometry(piece, s1 - gs)
            if cut is not None:
                piece = cut[0]
        pieces.append(piece)
    if reverse:
        pieces = [reverse_geometry(p) for p in reversed(pieces)]
        pieces = [p for p in pieces if p is not None]
    s = 0.0
    for p in pieces:
        p.set("s", _fmt(s))
        s += _f(p, "length")
    return pieces, s


# --- lane helpers ---------------------------------------------------------------------------

def _lane_elem(road: etree._Element, lane_id: int):
    for lane in road.iter("lane"):
        if lane.get("id") == str(lane_id):
            return lane
    return None


def _lane_width_at(lane: etree._Element, s: float) -> float:
    recs = sorted(lane.findall("width"), key=lambda e: _f(e, "sOffset"))
    if not recs:
        return 0.0
    cur = recs[0]
    for r in recs:
        if _f(r, "sOffset") <= s + 1e-9:
            cur = r
    ds = s - _f(cur, "sOffset")
    return _poly_eval((_f(cur, "a"), _f(cur, "b"), _f(cur, "c"), _f(cur, "d")), ds)


def _inner_offset_at(road: etree._Element, lane_id: int, s: float) -> float:
    """Signed lateral position (left positive) of the inner edge of ``lane_id`` at s."""
    off = 0.0
    step = 1 if lane_id > 0 else -1
    for lid in range(step, lane_id, step):
        lane = _lane_elem(road, lid)
        if lane is not None:
            off += step * _lane_width_at(lane, s)
    return off


def _clear_children(e: etree._Element):
    for c in list(e):
        e.remove(c)


def _ensure_link(parent: etree._Element) -> etree._Element:
    link = parent.find("link")
    if link is None:
        link = etree.Element("link")
        parent.insert(0, link)
    return link


def _set_road_link(road: etree._Element, side: str, etype: str, eid: str, contact: str | None):
    link = _ensure_link(road)
    tag = "successor" if side == "end" else "predecessor"
    for e in link.findall(tag):
        link.remove(e)
    e = etree.SubElement(link, tag)
    e.set("elementType", etype), e.set("elementId", str(eid))
    if contact and etype == "road":
        e.set("contactPoint", contact)
    # keep predecessor before successor
    preds, succs = link.findall("predecessor"), link.findall("successor")
    _clear_children(link)
    for x in preds + succs:
        link.append(x)


def _set_lane_link(lane: etree._Element, tag: str, lane_id: int):
    link = _ensure_link(lane)
    for e in link.findall(tag):
        link.remove(e)
    etree.SubElement(link, tag).set("id", str(lane_id))


def _exit_side(lane_id: int) -> str:
    return "end" if lane_id < 0 else "start"


def _entry_side(lane_id: int) -> str:
    return "start" if lane_id < 0 else "end"


def _road_end_pose(road: etree._Element, side: str):
    geoms = sorted(road.findall("planView/geometry"), key=lambda g: _f(g, "s"))
    if not geoms:
        return None
    if side == "start":
        g = geoms[0]
        return _f(g, "x"), _f(g, "y"), _f(g, "hdg")
    return geometry_end(geoms[-1])


# --- the rebuild ------------------------------------------------------------------------------

def _find(parent: dict, x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


def rebuild_junctions(root: etree._Element, edges: list[Edge], stub_len: float = DEFAULT_STUB_LEN) -> dict:
    """Rebuild all road/lane links and junctions of ``root`` from lanelet-level ``edges``.

    ``edges`` are ((road, lane), (road, lane)) pairs in driving direction (road ids as in the
    file, lane ids signed OpenDRIVE ids). Returns stats incl. ``stub_roads`` =
    {stub road id: (target road id, target lane id)}.
    """
    roads = {r.get("id"): r for r in root.findall("road")}

    # 0. forget cr2odr's links and junctions
    for j in root.findall("junction"):
        root.remove(j)
    for r in roads.values():
        r.set("junction", "-1")
        _clear_children(_ensure_link(r))
        for lane in r.iter("lane"):
            _clear_children(_ensure_link(lane))

    valid: list[Edge] = []
    dropped = 0
    for a, b in edges:
        a, b = (str(a[0]), int(a[1])), (str(b[0]), int(b[1]))
        if a[0] == b[0] or a[0] not in roads or b[0] not in roads \
                or _lane_elem(roads[a[0]], a[1]) is None or _lane_elem(roads[b[0]], b[1]) is None:
            dropped += 1
            continue
        if (a, b) not in valid:
            valid.append((a, b))
    out, inn = defaultdict(list), defaultdict(list)
    for a, b in valid:
        out[a].append(b)
        inn[b].append(a)

    # 1. classify edges: plain link vs. junction stub.
    # A road end can carry one exit link only (CARLA resolves every movement from the exit
    # side); plain exits that would share an end with a junction exit or with another plain
    # exit to a different road are promoted to stubs.
    stub_edges = {e for e in valid if len(out[e[0]]) > 1 or len(inn[e[1]]) > 1}
    while True:
        exit_slots: dict[tuple, str] = {}
        for (ra, la), _ in stub_edges:
            exit_slots[(ra, _exit_side(la))] = "junction"
        conflict = set()
        for (ra, la), (rb, lb) in valid:
            if ((ra, la), (rb, lb)) in stub_edges:
                continue
            key = (ra, _exit_side(la))
            if key in exit_slots and exit_slots[key] != rb:
                conflict.add(key)
            exit_slots.setdefault(key, rb)
        if not conflict:
            break
        stub_edges |= {e for e in valid if e not in stub_edges and (e[0][0], _exit_side(e[0][1])) in conflict}
    simple = [e for e in valid if e not in stub_edges]
    plain_exit_keys = {(ra, _exit_side(la)) for (ra, la), _ in simple}

    # 2. plain links (entry-side road links only where the end is not already used)
    used_slots = set(exit_slots)
    entry_links_omitted = 0
    for (ra, la), (rb, lb) in simple:
        _set_road_link(roads[ra], _exit_side(la), "road", rb, _entry_side(lb))
        key = (rb, _entry_side(lb))
        if key not in used_slots:
            _set_road_link(roads[rb], key[1], "road", ra, _exit_side(la))
            used_slots.add(key)
        else:
            entry_links_omitted += 1
        _set_lane_link(_lane_elem(roads[ra], la), "successor" if la < 0 else "predecessor", lb)
        _set_lane_link(_lane_elem(roads[rb], lb), "predecessor" if lb < 0 else "successor", la)

    # 3. stubs + junction clusters
    next_id = max([int(i) for i in roads] + [0]) + 1
    parent: dict = {}
    stub_roads: dict[str, tuple[str, int]] = {}
    stub_records = []  # (stub id, incoming lane, target lane, cluster key)
    gap_max = 0.0
    unsplittable = 0
    for (ra, la), (rb, lb) in sorted(stub_edges):
        target = roads[rb]
        length = _f(target, "length")
        d = min(stub_len, STUB_MAX_FRACTION * length)
        if lb < 0:
            pieces, d_eff = reference_piece(target, 0.0, d, reverse=False)
            s_at = lambda s_stub: s_stub  # noqa: E731
        else:
            pieces, d_eff = reference_piece(target, length - d, length, reverse=True)
            s_at = lambda s_stub: length - s_stub  # noqa: E731
        if not pieces or d_eff <= 0:
            unsplittable += 1
            continue
        sid = str(next_id)
        next_id += 1
        lane_t = _lane_elem(target, lb)
        w0, w1 = _lane_width_at(lane_t, s_at(0.0)), _lane_width_at(lane_t, s_at(d_eff))
        o0, o1 = _inner_offset_at(target, lb, s_at(0.0)), _inner_offset_at(target, lb, s_at(d_eff))
        if lb > 0:  # reversed frame: left becomes right
            o0, o1 = -o0, -o1

        stub = etree.Element("road")
        stub.set("name", f"stub {ra}:{la}->{rb}:{lb}")
        stub.set("length", _fmt(d_eff))
        stub.set("id", sid)
        stub.set("junction", "-1")  # set below
        link = etree.SubElement(stub, "link")
        p = etree.SubElement(link, "predecessor")
        p.set("elementType", "road"), p.set("elementId", ra), p.set("contactPoint", _exit_side(la))
        s_el = etree.SubElement(link, "successor")
        s_el.set("elementType", "road"), s_el.set("elementId", rb), s_el.set("contactPoint", _entry_side(lb))
        t = target.find("type")
        if t is not None:
            stub.append(copy.deepcopy(t))
        pv = etree.SubElement(stub, "planView")
        for g in pieces:
            pv.append(g)
        etree.SubElement(stub, "elevationProfile")
        etree.SubElement(stub, "lateralProfile")
        lanes = etree.SubElement(stub, "lanes")
        if abs(o0) > 1e-9 or abs(o1) > 1e-9:
            lo = etree.SubElement(lanes, "laneOffset")
            lo.set("s", "0.0"), lo.set("a", _fmt(o0)), lo.set("b", _fmt((o1 - o0) / d_eff))
            lo.set("c", "0.0"), lo.set("d", "0.0")
        sec = etree.SubElement(lanes, "laneSection")
        sec.set("s", "0.0")
        center = etree.SubElement(sec, "center")
        c_lane = etree.SubElement(center, "lane")
        c_lane.set("id", "0"), c_lane.set("type", "driving"), c_lane.set("level", "false")
        etree.SubElement(c_lane, "link")
        rm = lane_t.find("roadMark")
        if rm is not None:
            c_lane.append(copy.deepcopy(rm))
        right = etree.SubElement(sec, "right")
        r_lane = etree.SubElement(right, "lane")
        r_lane.set("id", "-1"), r_lane.set("type", lane_t.get("type", "driving")), r_lane.set("level", "false")
        r_link = etree.SubElement(r_lane, "link")
        etree.SubElement(r_link, "predecessor").set("id", str(la))
        etree.SubElement(r_link, "successor").set("id", str(lb))
        w = etree.SubElement(r_lane, "width")
        w.set("sOffset", "0.0"), w.set("a", _fmt(w0)), w.set("b", _fmt((w1 - w0) / d_eff))
        w.set("c", "0.0"), w.set("d", "0.0")
        if rm is not None:
            r_lane.append(copy.deepcopy(rm))
        etree.SubElement(stub, "objects")
        etree.SubElement(stub, "signals")
        target.addnext(stub)
        roads[sid] = stub
        stub_roads[sid] = (rb, lb)

        key_in, key_out = (ra, _exit_side(la)), (rb, _entry_side(lb))
        for k in (key_in, key_out):
            parent.setdefault(k, k)
        parent[_find(parent, key_out)] = _find(parent, key_in)
        stub_records.append((sid, (ra, la), (rb, lb), key_in))

        # sanity: the stub must start where the incoming road ends
        pose = _road_end_pose(roads[ra], _exit_side(la))
        if pose is not None:
            gap_max = max(gap_max, math.hypot(pose[0] - _f(pieces[0], "x"), pose[1] - _f(pieces[0], "y")))

    # 4. junction elements
    clusters: dict = defaultdict(list)
    for sid, a, b, key_in in stub_records:
        clusters[_find(parent, key_in)].append((sid, a, b))
    junction_of_end: dict[tuple, str] = {}
    jid = max([int(i) for i in roads] + [0]) + 1
    same_both_ends = []
    for rep, members in sorted(clusters.items(), key=lambda kv: kv[1][0][0]):
        j = etree.SubElement(root, "junction")
        j.set("id", str(jid)), j.set("name", f"junction {jid}")
        for i, (sid, (ra, la), (rb, lb)) in enumerate(sorted(members, key=lambda m: int(m[0]))):
            c = etree.SubElement(j, "connection")
            c.set("id", str(i)), c.set("incomingRoad", ra), c.set("connectingRoad", sid), c.set("contactPoint", "start")
            ll = etree.SubElement(c, "laneLink")
            ll.set("from", str(la)), ll.set("to", "-1")
            roads[sid].set("junction", str(jid))
            for key in ((ra, _exit_side(la)), (rb, _entry_side(lb))):
                junction_of_end[key] = str(jid)
        jid += 1
    for (rid, side), j in junction_of_end.items():
        if (rid, side) not in plain_exit_keys:
            _set_road_link(roads[rid], side, "junction", j, None)
    for rid in roads:
        if junction_of_end.get((rid, "start")) is not None and \
                junction_of_end.get((rid, "start")) == junction_of_end.get((rid, "end")):
            same_both_ends.append(rid)

    return {"edges": len(valid), "edges_dropped": dropped, "plain_links": len(simple),
            "entry_links_omitted": entry_links_omitted,
            "stubs": len(stub_roads), "junctions": len(clusters), "unsplittable": unsplittable,
            "stub_gap_max_m": round(gap_max, 3), "roads_with_same_junction_at_both_ends": same_both_ends,
            "stub_roads": stub_roads}


def lane_graph_from_scenario(scenario, cr_to_road: dict, cr_to_lane: dict) -> list[Edge]:
    """Lanelet succession as ((road, lane), (road, lane)) edges using cr2odr's id tables."""
    edges: list[Edge] = []
    for la in scenario.lanelet_network.lanelets:
        if la.lanelet_id not in cr_to_road or la.lanelet_id not in cr_to_lane:
            continue
        a = (str(cr_to_road[la.lanelet_id]), int(cr_to_lane[la.lanelet_id]))
        for s in la.successor:
            if s in cr_to_road and s in cr_to_lane:
                edges.append((a, (str(cr_to_road[s]), int(cr_to_lane[s]))))
    return edges


def rebuild_file(path, edges: list[Edge], stub_len: float = DEFAULT_STUB_LEN) -> dict:
    tree = etree.parse(str(path))
    stats = rebuild_junctions(tree.getroot(), edges, stub_len)
    tree.write(str(path), xml_declaration=True, encoding="UTF-8", pretty_print=True)
    return stats
