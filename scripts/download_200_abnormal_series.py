import re
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from datetime import datetime

from dicomweb_client.api import DICOMwebClient
from requests.auth import HTTPBasicAuth


CSV_PATH = r"D:\CT_Brain_StudyWise_Project\metadata\selected_300_studywise.csv"

PACS_DICOMWEB_URL = "http://pacs.uniquewellness.co.in:8080/dcm4chee-arc/aets/DCM4CHEE/rs"

OUTPUT_DICOM_ROOT = r"D:\CT_Brain_StudyWise_Project\dataset_raw\ABNORMAL"

PACS_USERNAME = ""
PACS_PASSWORD = ""

OVERWRITE_EXISTING_SERIES_FOLDER = True

LOG_CSV = r"D:\CT_Brain_StudyWise_Project\results\abnormal_200_download_log.csv"

MAX_ABNORMAL_STUDIES = 200


def normalize_uid(value):
    return str(value).strip().replace("_", ".")


def safe_folder_name(value):
    value = str(value).strip()
    value = re.sub(r'[<>:"/\\|?*]', "_", value)
    return value


def make_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def clear_folder(folder: Path):
    if folder.exists():
        for file in folder.rglob("*"):
            if file.is_file():
                try:
                    file.unlink()
                except Exception:
                    pass
    folder.mkdir(parents=True, exist_ok=True)


def load_abnormal_series(csv_path):
    df = pd.read_csv(csv_path)

    required_cols = ["study_uid", "selected_series_uid", "label"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required CSV columns: {missing}")

    df["label"] = df["label"].astype(str).str.upper().str.strip()
    df["study_uid"] = df["study_uid"].astype(str).str.strip()
    df["selected_series_uid"] = df["selected_series_uid"].astype(str).str.strip()

    abnormal_df = df[df["label"] == "ABNORMAL"].copy()

    abnormal_df = abnormal_df[
        (abnormal_df["study_uid"] != "") &
        (abnormal_df["study_uid"].str.lower() != "nan") &
        (abnormal_df["selected_series_uid"] != "") &
        (abnormal_df["selected_series_uid"].str.lower() != "nan")
    ]

    abnormal_df["study_uid"] = abnormal_df["study_uid"].apply(normalize_uid)
    abnormal_df["selected_series_uid"] = abnormal_df["selected_series_uid"].apply(normalize_uid)

    abnormal_df = abnormal_df.drop_duplicates(
        subset=["study_uid", "selected_series_uid"]
    ).reset_index(drop=True)

    abnormal_df = abnormal_df.head(MAX_ABNORMAL_STUDIES)

    return abnormal_df


def get_instance_sort_key(ds):
    try:
        instance_number = int(getattr(ds, "InstanceNumber", 999999))
    except Exception:
        instance_number = 999999

    try:
        z_pos = float(ds.ImagePositionPatient[2])
    except Exception:
        z_pos = 0.0

    sop_uid = str(getattr(ds, "SOPInstanceUID", ""))

    return instance_number, z_pos, sop_uid


def save_series_datasets(datasets, output_series_folder: Path):
    datasets = sorted(datasets, key=get_instance_sort_key)

    saved_count = 0

    for idx, ds in enumerate(datasets, start=1):
        sop_uid = str(getattr(ds, "SOPInstanceUID", "")).strip()

        if sop_uid:
            filename = f"{idx:04d}_{safe_folder_name(sop_uid)}.dcm"
        else:
            filename = f"IMG_{idx:04d}.dcm"

        output_file = output_series_folder / filename
        ds.save_as(str(output_file), write_like_original=False)
        saved_count += 1

    return saved_count


def main():
    start_time = datetime.now()

    output_root = Path(OUTPUT_DICOM_ROOT)
    make_dir(output_root)

    abnormal_df = load_abnormal_series(CSV_PATH)

    print("=" * 90)
    print("200 ABNORMAL STUDYWISE SERIES DOWNLOADER")
    print("=" * 90)
    print(f"CSV path              : {CSV_PATH}")
    print(f"PACS DICOMweb URL     : {PACS_DICOMWEB_URL}")
    print(f"Output DICOM root     : {OUTPUT_DICOM_ROOT}")
    print(f"Abnormal series count : {len(abnormal_df)}")
    print("=" * 90)

    if PACS_USERNAME and PACS_PASSWORD:
        client = DICOMwebClient(url=PACS_DICOMWEB_URL)
        client.session.auth = HTTPBasicAuth(PACS_USERNAME, PACS_PASSWORD)
    else:
        client = DICOMwebClient(url=PACS_DICOMWEB_URL)

    logs = []

    for _, row in tqdm(abnormal_df.iterrows(), total=len(abnormal_df), desc="Downloading abnormal series"):
        study_uid = row["study_uid"]
        series_uid = row["selected_series_uid"]

        safe_study = safe_folder_name(study_uid)
        safe_series = safe_folder_name(series_uid)

        output_series_folder = output_root / safe_study / safe_series

        log_row = {
            "study_uid": study_uid,
            "selected_series_uid": series_uid,
            "label": "ABNORMAL",
            "output_series_folder": str(output_series_folder),
            "status": "",
            "downloaded_files": 0,
            "error": ""
        }

        try:
            if OVERWRITE_EXISTING_SERIES_FOLDER:
                clear_folder(output_series_folder)
            else:
                make_dir(output_series_folder)

            datasets = client.retrieve_series(
                study_instance_uid=study_uid,
                series_instance_uid=series_uid
            )

            if datasets is None or len(datasets) == 0:
                log_row["status"] = "NO_INSTANCES_FOUND"
                logs.append(log_row)
                continue

            saved_count = save_series_datasets(datasets, output_series_folder)

            log_row["downloaded_files"] = saved_count
            log_row["status"] = "SUCCESS" if saved_count > 0 else "NO_FILES_SAVED"

            logs.append(log_row)

        except Exception as e:
            log_row["status"] = "FAILED"
            log_row["error"] = str(e)
            logs.append(log_row)

    log_df = pd.DataFrame(logs)
    log_df.to_csv(LOG_CSV, index=False)

    print("\n" + "=" * 90)
    print("DOWNLOAD COMPLETED")
    print("=" * 90)
    print(f"Total abnormal series : {len(abnormal_df)}")
    print(f"Success               : {(log_df['status'] == 'SUCCESS').sum()}")
    print(f"No instances found    : {(log_df['status'] == 'NO_INSTANCES_FOUND').sum()}")
    print(f"No files saved        : {(log_df['status'] == 'NO_FILES_SAVED').sum()}")
    print(f"Failed                : {(log_df['status'] == 'FAILED').sum()}")
    print(f"Log saved             : {LOG_CSV}")
    print(f"Started               : {start_time}")
    print(f"Finished              : {datetime.now()}")
    print("=" * 90)


if __name__ == "__main__":
    main()