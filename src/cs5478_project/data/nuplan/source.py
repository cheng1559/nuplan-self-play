"""Direct read-only queries against a nuPlan log database (ego poses, boxes, tags)."""

import sqlite3
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class FrameRow:
    token: bytes
    token_hex: str
    timestamp_us: int
    x: float
    y: float
    yaw: float
    vx: float
    vy: float
    ax: float
    yaw_rate: float


@dataclass(frozen=True)
class BoxRow:
    track_token_hex: str
    category: str
    x: float
    y: float
    yaw: float
    vx: float
    vy: float
    length: float
    width: float


def connect(database: str | Path) -> sqlite3.Connection:
    conn = sqlite3.connect(Path(database).as_uri() + "?mode=ro&immutable=1", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def yaw_from_quaternion(qw: float, qx: float, qy: float, qz: float) -> float:
    """Yaw exactly as nuPlan reads ego poses (pyquaternion's yaw_pitch_roll).

    The planar atan2 formula differs by up to ~1e-4 rad when the pose has pitch or roll.
    """
    from pyquaternion import Quaternion

    return float(Quaternion(qw, qx, qy, qz).yaw_pitch_roll[0])


def load_frames(conn: sqlite3.Connection) -> list[FrameRow]:
    """All lidar frames with their ego poses, in timestamp order."""
    query = """
        SELECT lp.token, lp.timestamp, ep.x, ep.y, ep.qw, ep.qx, ep.qy, ep.qz,
               ep.vx, ep.vy, ep.acceleration_x AS ax, ep.angular_rate_z AS yaw_rate
        FROM lidar_pc AS lp INNER JOIN ego_pose AS ep ON ep.token = lp.ego_pose_token
        ORDER BY lp.timestamp ASC
    """
    rows = []
    for row in conn.execute(query):
        token = bytes(row["token"])
        rows.append(
            FrameRow(
                token=token,
                token_hex=token.hex(),
                timestamp_us=int(row["timestamp"]),
                x=float(row["x"]),
                y=float(row["y"]),
                yaw=yaw_from_quaternion(row["qw"], row["qx"], row["qy"], row["qz"]),
                vx=float(row["vx"] or 0.0),
                vy=float(row["vy"] or 0.0),
                ax=float(row["ax"] or 0.0),
                yaw_rate=float(row["yaw_rate"] or 0.0),
            )
        )
    return rows


def load_boxes(conn: sqlite3.Connection, tokens: list[bytes]) -> dict[bytes, list[BoxRow]]:
    """Annotated boxes per lidar frame token, with their track category."""
    output: dict[bytes, list[BoxRow]] = {token: [] for token in tokens}
    for start in range(0, len(tokens), 500):
        chunk = tokens[start : start + 500]
        query = f"""
            SELECT lb.lidar_pc_token, lb.track_token, lb.x, lb.y, lb.yaw, lb.vx, lb.vy,
                   lb.length, lb.width, c.name AS category
            FROM lidar_box AS lb
            INNER JOIN track AS tr ON tr.token = lb.track_token
            INNER JOIN category AS c ON c.token = tr.category_token
            WHERE lb.lidar_pc_token IN ({",".join("?" for _ in chunk)})
        """
        for row in conn.execute(query, chunk):
            output.setdefault(bytes(row["lidar_pc_token"]), []).append(
                BoxRow(
                    track_token_hex=bytes(row["track_token"]).hex(),
                    category=str(row["category"] or ""),
                    x=float(row["x"]),
                    y=float(row["y"]),
                    yaw=float(row["yaw"]),
                    vx=float(row["vx"] or 0.0),
                    vy=float(row["vy"] or 0.0),
                    length=float(row["length"] or 0.0),
                    width=float(row["width"] or 0.0),
                )
            )
    return output


def scenario_tags(conn: sqlite3.Connection, token: bytes) -> list[str]:
    rows = conn.execute(
        "SELECT type FROM scenario_tag WHERE lidar_pc_token = ? ORDER BY type ASC", (token,)
    )
    return [str(row["type"]) for row in rows if row["type"] is not None]


def log_map_name(conn: sqlite3.Connection) -> str:
    row = conn.execute("SELECT map_version FROM log LIMIT 1").fetchone()
    if row is None:
        raise ValueError("Log database has no log row")
    return str(row["map_version"])
