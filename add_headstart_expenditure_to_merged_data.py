import argparse
from pathlib import Path

import numpy as np
import pandas as pd


STATE_FIPS_TO_NAME = {
    1: "Alabama",
    2: "Alaska",
    4: "Arizona",
    5: "Arkansas",
    6: "California",
    8: "Colorado",
    9: "Connecticut",
    10: "Delaware",
    11: "District of Columbia",
    12: "Florida",
    13: "Georgia",
    15: "Hawaii",
    16: "Idaho",
    17: "Illinois",
    18: "Indiana",
    19: "Iowa",
    20: "Kansas",
    21: "Kentucky",
    22: "Louisiana",
    23: "Maine",
    24: "Maryland",
    25: "Massachusetts",
    26: "Michigan",
    27: "Minnesota",
    28: "Mississippi",
    29: "Missouri",
    30: "Montana",
    31: "Nebraska",
    32: "Nevada",
    33: "New Hampshire",
    34: "New Jersey",
    35: "New Mexico",
    36: "New York",
    37: "North Carolina",
    38: "North Dakota",
    39: "Ohio",
    40: "Oklahoma",
    41: "Oregon",
    42: "Pennsylvania",
    44: "Rhode Island",
    45: "South Carolina",
    46: "South Dakota",
    47: "Tennessee",
    48: "Texas",
    49: "Utah",
    50: "Vermont",
    51: "Virginia",
    53: "Washington",
    54: "West Virginia",
    55: "Wisconsin",
    56: "Wyoming",
}


REQUIRED_FAKE_COLUMNS = ["FIPS", "CYRB_XRND", "CHILD_EVER_ENROLLED_IN_HEAD"]
RUNNING_TOTAL_COLUMN = "6-Year Running Total Allocations/Student"
NEW_COLUMN = "total real headstart expenditure"


def county_fips_to_state_name(value):
    if pd.isna(value):
        return np.nan
    try:
        digits = str(int(float(value))).zfill(5)
    except (TypeError, ValueError):
        digits = str(value).strip().split(".")[0].zfill(5)
    if not digits[:2].isdigit():
        return np.nan
    return STATE_FIPS_TO_NAME.get(int(digits[:2]), np.nan)


def add_total_real_headstart_expenditure(fake_path, master_path, output_path):
    fake = pd.read_csv(fake_path)
    master = pd.read_csv(master_path)

    missing_fake = [col for col in REQUIRED_FAKE_COLUMNS if col not in fake.columns]
    if missing_fake:
        raise ValueError(f"Missing required fake data columns: {missing_fake}")
    missing_master = [col for col in ["State", "Fiscal Year", RUNNING_TOTAL_COLUMN] if col not in master.columns]
    if missing_master:
        raise ValueError(f"Missing required master data columns: {missing_master}")

    lookup = master[["State", "Fiscal Year", RUNNING_TOTAL_COLUMN]].copy()
    lookup["Fiscal Year"] = pd.to_numeric(lookup["Fiscal Year"], errors="coerce").astype("Int64")
    lookup = lookup.rename(columns={
        "Fiscal Year": "_headstart_lookup_year",
        RUNNING_TOTAL_COLUMN: "_six_year_total",
    })

    work = fake.copy()
    work["_headstart_state"] = work["FIPS"].map(county_fips_to_state_name)
    birth_year = pd.to_numeric(work["CYRB_XRND"], errors="coerce")
    # The master running total is stored on the terminal year, so start year Y maps to Y+5.
    work["_headstart_lookup_year"] = (birth_year + 5).round().astype("Int64")

    work = work.merge(
        lookup,
        how="left",
        left_on=["_headstart_state", "_headstart_lookup_year"],
        right_on=["State", "_headstart_lookup_year"],
        suffixes=("", "_headstart_master"),
    )

    enrolled = pd.to_numeric(work["CHILD_EVER_ENROLLED_IN_HEAD"], errors="coerce")
    work[NEW_COLUMN] = np.nan
    work.loc[enrolled.eq(0), NEW_COLUMN] = 0
    work.loc[enrolled.gt(0), NEW_COLUMN] = work.loc[enrolled.gt(0), "_six_year_total"]

    drop_cols = [
        "_headstart_state",
        "_headstart_lookup_year",
        "_six_year_total",
        "State",
    ]
    output = work.drop(columns=[col for col in drop_cols if col in work.columns])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)

    return {
        "input_rows": len(fake),
        "output_rows": len(output),
        "zero_values": int(output[NEW_COLUMN].eq(0).sum()),
        "positive_enrolled_rows": int(enrolled.gt(0).sum()),
        "positive_enrolled_matched": int(output.loc[enrolled.gt(0), NEW_COLUMN].notna().sum()),
        "positive_enrolled_blank": int(output.loc[enrolled.gt(0), NEW_COLUMN].isna().sum()),
        "output_path": str(output_path),
    }


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Append total real Head Start expenditure to Fake_Merged_Data.csv."
    )
    parser.add_argument("--fake-data", type=Path, default=root / "Fake_Merged_Data.csv")
    parser.add_argument(
        "--master-data",
        type=Path,
        default=root / "outputs" / "headstart_funding_per_student_state_year_1988_2010.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "outputs" / "Fake_Merged_Data_with_headstart_expenditure.csv",
    )
    args = parser.parse_args()

    summary = add_total_real_headstart_expenditure(args.fake_data, args.master_data, args.output)
    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
