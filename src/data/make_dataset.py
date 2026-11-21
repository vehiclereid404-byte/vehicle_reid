"""Build data/processed/reid/crops.csv: clean, ID-disjoint ReID split of RoundaboutHD.
Run via `dvc repro process_data`. Reasoning behind each rule: notebooks/2.0 and 3.0, reports/data_issues.md."""
import numpy as np
import pandas as pd

from src.data.load import ROOT, RAW, CAMS, load_reid_index, add_image_info, link_reid

SEED = 42
TEST_FRAC, VAL_FRAC = 1 / 3, 1 / 6  # of the multi-camera IDs
OUT = ROOT / "data" / "processed" / "reid" / "crops.csv"


def main():
    rng = np.random.default_rng(SEED)
    r = link_reid(add_image_info(load_reid_index())).rename(columns={"split": "official_split"})

    # 1. remove byte-identical duplicates
    r = r.sort_values(["md5", "official_split"]).drop_duplicates("md5").sort_values("path").reset_index(drop=True)

    # 2. file id 673 merges two vehicles: crops linked to global id 664 (box truck) get a new id
    box_truck_cams = set(r[(r.id == 673) & (r.gid == 664)].cam)
    r.loc[(r.id == 673) & r.cam.isin(box_truck_cams), "id"] = int(r.id.max()) + 1

    # 3. drop IDs with < 2 crops
    cnt = r.groupby("id").size()
    r = r[r.id.isin(cnt[cnt >= 2].index)]

    # 4. split by vehicle ID; val/test only from IDs seen in >= 2 cameras
    cams_per_id = r.groupby("id").cam.nunique()
    multi = rng.permutation(cams_per_id[cams_per_id >= 2].index.to_numpy())
    n_test, n_val = round(len(multi) * TEST_FRAC), round(len(multi) * VAL_FRAC)
    split_of = {i: "test" for i in multi[:n_test]} | {i: "val" for i in multi[n_test:n_test + n_val]}
    r["split"] = r.id.map(split_of).fillna("train")

    # 5. val/test: 1 random crop per (id, cam) is a query, the rest is gallery
    r["role"] = np.where(r.split == "train", "train", "gallery")
    q_idx = r[r.split != "train"].groupby(["id", "cam"]).sample(1, random_state=SEED).index
    r.loc[q_idx, "role"] = "query"

    # 6. vehicle attributes (keyed by cam + SCT track_id; only the first 6 columns are data)
    xl = pd.read_excel(RAW / "Vehicle_statistic.xlsx", sheet_name=None)
    attr = pd.concat([xl[f"imagesc{c:03d}"].iloc[:, :6].assign(cam=c) for c in CAMS])
    attr = attr.rename(columns=str.lower).rename(columns={"number": "track_id"})
    r = r.merge(attr[["cam", "track_id", "category", "colour", "make", "model"]], on=["cam", "track_id"], how="left")

    # 7. checks
    ids = {s: set(r[r.split == s].id) for s in ["train", "val", "test"]}
    assert not (ids["train"] & ids["val"] or ids["train"] & ids["test"] or ids["val"] & ids["test"]), "ID leakage"
    for s in ["val", "test"]:
        e = r[r.split == s]
        gal = e[e.role == "gallery"].groupby("id").cam.agg(set)
        assert all(gal.get(i, set()) - {c} for i, c in zip(e[e.role == "query"].id, e[e.role == "query"].cam)), \
            f"{s}: query without cross-camera gallery match"

    cols = ["path", "id", "cam", "frame", "split", "role", "official_split", "track_id", "gid", "w", "h",
            "category", "colour", "make", "model"]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    r[cols].astype({"track_id": "Int64", "gid": "Int64"}).sort_values(["split", "id", "cam", "frame"]).to_csv(OUT, index=False)
    print(r.groupby(["split", "role"]).agg(ids=("id", "nunique"), crops=("path", "size")))


if __name__ == "__main__":
    main()
