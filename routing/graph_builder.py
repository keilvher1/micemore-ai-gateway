"""부스 그래프 빌더 — networkx 없이 순수 python.

베타 단계: 거리만으로 weighted edge.
Sprint 2+: networkx + 다익스트라로 혼잡 가중 최단 경로.

좌표는 Phase 4-A governance 와 동일한 50m 그리드 셀로 표현 가능.
여기선 (lat, lon) raw 도 허용 — Haversine 으로 거리 계산.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class BoothNode:
    booth_id: str
    name: str
    lat: float
    lon: float
    category: str = ""


@dataclass
class BoothGraph:
    nodes: dict[str, BoothNode] = field(default_factory=dict)
    edges: dict[str, dict[str, float]] = field(default_factory=dict)
    # edges[a][b] = 거리 m (양방향)

    def add_node(self, n: BoothNode) -> None:
        self.nodes[n.booth_id] = n
        self.edges.setdefault(n.booth_id, {})

    def neighbors(self, booth_id: str) -> dict[str, float]:
        return self.edges.get(booth_id, {})

    def distance(self, a: str, b: str) -> float | None:
        return self.edges.get(a, {}).get(b)


def haversine_m(a: BoothNode, b: BoothNode) -> float:
    R = 6_371_000.0
    p1, p2 = math.radians(a.lat), math.radians(b.lat)
    dp = math.radians(b.lat - a.lat)
    dl = math.radians(b.lon - a.lon)
    h = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R * math.asin(math.sqrt(h))


def build_graph(
    nodes: list[BoothNode], *, max_edge_m: float = 200.0
) -> BoothGraph:
    """모든 부스 페어 거리 계산, max_edge_m 이내만 edge 추가."""
    g = BoothGraph()
    for n in nodes:
        g.add_node(n)
    for i, a in enumerate(nodes):
        for b in nodes[i + 1:]:
            d = round(haversine_m(a, b), 1)
            if d <= max_edge_m:
                g.edges[a.booth_id][b.booth_id] = d
                g.edges[b.booth_id][a.booth_id] = d
    return g


# ---------------------------------------------------------------------------
# 다익스트라 (혼잡 가중 옵션)
# ---------------------------------------------------------------------------
def shortest_path(
    graph: BoothGraph,
    *,
    src: str,
    dst: str,
    crowd_weight: dict[str, float] | None = None,
    crowd_factor: float = 30.0,
) -> tuple[list[str], float]:
    """src→dst 최단경로. crowd_weight (booth_id → 0~1) 가 있으면 거리에 가중.

    edge 비용 = base_distance × (1 + crowd_factor × crowd[next_booth])
    """
    if src not in graph.nodes or dst not in graph.nodes:
        return [], math.inf

    import heapq
    crowd_weight = crowd_weight or {}
    dist: dict[str, float] = {src: 0.0}
    prev: dict[str, str | None] = {src: None}
    pq: list[tuple[float, str]] = [(0.0, src)]
    while pq:
        d, u = heapq.heappop(pq)
        if u == dst:
            break
        if d > dist.get(u, math.inf):
            continue
        for v, base in graph.edges.get(u, {}).items():
            cost = base * (1 + crowd_factor * crowd_weight.get(v, 0.0))
            nd = d + cost
            if nd < dist.get(v, math.inf):
                dist[v] = nd
                prev[v] = u
                heapq.heappush(pq, (nd, v))
    if dst not in dist:
        return [], math.inf
    # reconstruct
    path: list[str] = []
    cur: str | None = dst
    while cur is not None:
        path.append(cur)
        cur = prev.get(cur)
    return list(reversed(path)), round(dist[dst], 1)
