from __future__ import annotations

import argparse
import json

from metraq_dip.data.aq_backends import get_aq_backend
from metraq_gnn.data import build_aq_sensor_cache


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build or resume a raw AQ sensor cache for the GNN")
    parser.add_argument("--output", required=True)
    parser.add_argument("--start", required=True, help="Inclusive, hour-aligned timestamp")
    parser.add_argument("--end", required=True, help="Inclusive, hour-aligned timestamp")
    parser.add_argument("--magnitudes", nargs="+", type=int, default=[8])
    parser.add_argument("--dataset", default="metraq")
    parser.add_argument("--backend", default="db")
    parser.add_argument("--cell-size-m", type=int, default=1000)
    parser.add_argument("--margin-m-x", type=int, default=3000)
    parser.add_argument("--margin-m-y", type=int, default=2000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cache = build_aq_sensor_cache(
        path=args.output,
        aq_backend=get_aq_backend(dataset=args.dataset, backend=args.backend),
        start=args.start,
        end=args.end,
        magnitudes=args.magnitudes,
        cell_size_m=args.cell_size_m,
        margin_m_x=args.margin_m_x,
        margin_m_y=args.margin_m_y,
    )
    print(
        json.dumps(
            {
                "path": str(cache.path.resolve()),
                "shape": list(cache.values.shape),
                "grid_shape": list(cache.grid_shape),
                "completed_years": cache.metadata["completed_years"],
                "available_values": int(cache.availability.sum()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
