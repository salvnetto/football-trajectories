from kloppy import sportec
import pandas as pd
from pathlib import Path

match_ids = [
    "000001_DFL-MAT-J03WMX",
    "000001_DFL-MAT-J03WN1",
    "000002_DFL-MAT-J03WOH",
    "000002_DFL-MAT-J03WOY",
    "000002_DFL-MAT-J03WPY",
    "000002_DFL-MAT-J03WQQ",
    "000002_DFL-MAT-J03WR9"
]

data_dir = Path("data/raw")
output_dir = Path("data/")

for match_id in match_ids:
    # Load event data
    events = sportec.load_event(
        event_data=data_dir / f"DFL_03_02_events_raw_DFL-COM-{match_id}.xml",
        meta_data=data_dir / f"DFL_02_01_matchinformation_DFL-COM-{match_id}.xml",
        coordinates = "metrica"
    )

    # Load tracking data
    tracking = sportec.load_tracking(
        raw_data=data_dir / f"DFL_04_03_positions_raw_observed_DFL-COM-{match_id}.xml",
        meta_data=data_dir / f"DFL_02_01_matchinformation_DFL-COM-{match_id}.xml",
        only_alive=False,
        coordinates = "metrica"
    )

    # Convert to DataFrame
    events_df = events.to_df()
    events_df["time_seconds"] = events_df["timestamp"].dt.total_seconds()
    tracking_df = tracking.to_df()
    tracking_df["time_seconds"] = tracking_df["timestamp"].dt.total_seconds()

    # Save
    events_df.to_csv(output_dir / f"{match_id}_events.csv", index=False)
    tracking_df.to_csv(output_dir / f"{match_id}_tracking.csv", index=False)

    print(f"Saved {match_id}")
