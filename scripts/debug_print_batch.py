import json
from pathlib import Path
from torch.utils.data import DataLoader

from mcfp.data.datasets import Stage1Dataset, PoseFeatureConfig
from mcfp.data.io import read_jsonl
from mcfp.data.collate import GroupedBatchCollator


def read_split_ids(txt_path: Path):
    ids = []
    with txt_path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                ids.append(s)
    return ids


def main():
    repo_root = Path(".").resolve()
    manifest_path = repo_root / "data/manifests/stage1_manifest.jsonl"
    split_path = repo_root / "data/splits/stage1_train.txt"
    stats_path = repo_root / "data/stats/stage1_train_stats.json"

    label_keys = json.loads(stats_path.read_text(encoding="utf-8"))["label_keys"]
    assert isinstance(label_keys, list) and len(label_keys) > 0
    assert "g_ws" in label_keys

    manifest_records = read_jsonl(manifest_path)
    variant_ids = read_split_ids(split_path)

    pose_cfg = PoseFeatureConfig(
        use_aabb_ratio=True,
        use_aabb_centered=True,
        use_morph_scale=True,
        grid_round_decimals=6,
    )

    ds = Stage1Dataset(
        repo_root=repo_root,
        manifest_records=manifest_records,
        variant_ids=variant_ids,
        label_keys=label_keys,
        pose_cfg=pose_cfg,
        cache_maps=False,
        cache_specs=False,
    )

    collator = GroupedBatchCollator(label_keys=label_keys)

    dl = DataLoader(
        ds,
        batch_size=8,          # 这里先不引入 grouped sampler，先验证 contract
        shuffle=False,
        num_workers=0,
        collate_fn=collator,
    )

    batch = next(iter(dl))
    print("batch keys:", sorted(batch.keys()))
    print("variant_id:", batch["variant_id"])
    print("pose_feats:", batch["pose_feats"].shape, batch["pose_feats"].dtype)
    print("labels:", batch["labels"].shape, batch["labels"].dtype)
    print("ws_mask:", batch["ws_mask"].shape, batch["ws_mask"].dtype)
    print("label_keys len:", len(batch["label_keys"]))

    g = batch["morph_graph"]
    print("morph_graph type:", type(g))
    # 如果 GraphData 里有 x/edge_index/batch，建议也打印
    if hasattr(g, "x"):
        print("graph.x:", getattr(g, "x").shape, getattr(g, "x").dtype)
    if hasattr(g, "edge_index"):
        print("graph.edge_index:", getattr(g, "edge_index").shape, getattr(g, "edge_index").dtype)
    if hasattr(g, "batch"):
        print("graph.batch:", getattr(g, "batch").shape, getattr(g, "batch").dtype)

    b = getattr(g, "batch", None)
    print("graph.batch:", None if b is None else (b.shape, b.dtype))


    # Contract assertions
    assert batch["pose_feats"].shape[1] == 9
    assert batch["labels"].shape[1] == len(label_keys)
    assert batch["ws_mask"].dtype == batch["ws_mask"].dtype  # bool
    print("[OK] Batch contract v1 looks correct (without morph_spec).")


if __name__ == "__main__":
    main()
