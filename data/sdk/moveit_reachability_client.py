from __future__ import annotations

import argparse
import json
from typing import List

import requests


def _parse_list(text: str) -> List[float]:
    parts = [p.strip() for p in text.split(",") if p.strip()]
    return [float(p) for p in parts]


def _post(url: str, payload: dict, timeout: float) -> None:
    resp = requests.post(url, json=payload, timeout=timeout)
    resp.raise_for_status()
    print(json.dumps(resp.json(), indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="HTTP client for MoveIt reachability bridge.")
    parser.add_argument(
        "--url",
        type=str,
        default="http://127.0.0.1:8000",
        help="Server base URL.",
    )
    parser.add_argument("--timeout", type=float, default=5.0, help="Request timeout (s).")

    sub = parser.add_subparsers(dest="mode", required=True)

    joint = sub.add_parser("joint", help="Query by joint positions.")
    joint.add_argument("--positions", required=True, help="Comma-separated joint positions.")
    joint.add_argument("--names", default="", help="Comma-separated joint names.")
    joint.add_argument("--fk-link", default="", help="FK link name.")
    joint.add_argument("--frame-id", default="", help="FK frame id.")
    joint.add_argument("--group", default="", help="MoveIt group name.")

    pose = sub.add_parser("pose", help="Query by target pose.")
    pose.add_argument("--pos", required=True, help="Comma-separated position x,y,z.")
    pose.add_argument("--quat", required=True, help="Comma-separated quaternion x,y,z,w.")
    pose.add_argument("--group", default="", help="MoveIt group name.")
    pose.add_argument("--ik-link", default="", help="IK link name.")
    pose.add_argument("--frame-id", default="", help="Pose frame id.")
    pose.add_argument("--seed", default="", help="Comma-separated seed joint positions.")
    pose.add_argument("--seed-names", default="", help="Comma-separated seed joint names.")
    pose.add_argument("--avoid-collisions", action="store_true", help="IK avoid collisions.")

    args = parser.parse_args()
    base_url = args.url.rstrip("/")

    if args.mode == "joint":
        positions = _parse_list(args.positions)
        names = [n.strip() for n in args.names.split(",") if n.strip()]
        payload = {
            "joint_positions": positions,
            "joint_names": names or None,
            "fk_link": args.fk_link or None,
            "frame_id": args.frame_id or None,
            "group_name": args.group or None,
        }
        _post(f"{base_url}/query/joint", payload, args.timeout)
        return

    if args.mode == "pose":
        pos = _parse_list(args.pos)
        quat = _parse_list(args.quat)
        if len(pos) != 3 or len(quat) != 4:
            raise SystemExit("pos must be x,y,z and quat must be x,y,z,w")
        seed = _parse_list(args.seed) if args.seed else None
        seed_names = [n.strip() for n in args.seed_names.split(",") if n.strip()]
        payload = {
            "position": {"x": pos[0], "y": pos[1], "z": pos[2]},
            "orientation": {"x": quat[0], "y": quat[1], "z": quat[2], "w": quat[3]},
            "group_name": args.group or None,
            "ik_link": args.ik_link or None,
            "frame_id": args.frame_id or None,
            "seed_joint_positions": seed,
            "seed_joint_names": seed_names or None,
            "avoid_collisions": bool(args.avoid_collisions),
        }
        _post(f"{base_url}/query/pose", payload, args.timeout)
        return


if __name__ == "__main__":
    main()
