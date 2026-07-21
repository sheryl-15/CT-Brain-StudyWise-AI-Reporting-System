from ftplib import FTP
from pathlib import Path
import pandas as pd
from tqdm import tqdm

CSV_PATH = r"D:\CT_Brain_StudyWise_Project\metadata\selected_300_studywise.csv"

FTP_HOST = "106.51.0.53"
FTP_USER = "Administrator"
FTP_PASS = "mtpl$123"   # add password if needed
FTP_ROOT = "/dicoms"

OUTPUT_ROOT = Path(r"D:\CT_Brain_StudyWise_Project\dataset_raw\NORMAL")
LOG_CSV = r"D:\CT_Brain_StudyWise_Project\results\normal_100_ftp_fixed_log.csv"

MAX_NORMAL = 150


def connect_ftp():
    ftp = FTP(FTP_HOST, timeout=120)
    ftp.login(FTP_USER, FTP_PASS)
    return ftp


def is_folder(ftp, path):
    current = ftp.pwd()
    try:
        ftp.cwd(path)
        ftp.cwd(current)
        return True
    except Exception:
        try:
            ftp.cwd(current)
        except Exception:
            pass
        return False


def download_recursive(ftp, remote_dir, local_dir):
    local_dir.mkdir(parents=True, exist_ok=True)
    downloaded = 0

    try:
        items = ftp.nlst(remote_dir)
    except Exception as e:
        print("Cannot list:", remote_dir, e)
        return 0

    for item in items:
        name = item.split("/")[-1]
        local_path = local_dir / name

        if is_folder(ftp, item):
            downloaded += download_recursive(ftp, item, local_path)
        else:
            try:
                # Only download DICOM-like files
                if local_path.exists() and local_path.stat().st_size > 0:
                    downloaded += 1
                    continue

                with open(local_path, "wb") as f:
                    ftp.retrbinary(f"RETR {item}", f.write)

                downloaded += 1

            except Exception as e:
                print("Failed file:", item, e)

    return downloaded


def main():
    df = pd.read_csv(CSV_PATH)
    df["label"] = df["label"].astype(str).str.upper().str.strip()
    df["study_uid"] = df["study_uid"].astype(str).str.strip()

    normal_df = df[df["label"] == "NORMAL"].drop_duplicates(
        subset=["study_uid"]
    ).head(MAX_NORMAL)

    print("NORMAL studies selected:", len(normal_df))

    ftp = connect_ftp()

    logs = []

    for _, row in tqdm(normal_df.iterrows(), total=len(normal_df), desc="Downloading NORMAL from FTP"):
        study_uid = row["study_uid"]

        remote_study = f"{FTP_ROOT}/{study_uid}"
        local_study = OUTPUT_ROOT / study_uid

        try:
            count = download_recursive(ftp, remote_study, local_study)
            status = "SUCCESS" if count > 0 else "NO_FILES"

            logs.append({
                "study_uid": study_uid,
                "label": "NORMAL",
                "remote_path": remote_study,
                "local_path": str(local_study),
                "downloaded_files": count,
                "status": status,
                "error": ""
            })

        except Exception as e:
            logs.append({
                "study_uid": study_uid,
                "label": "NORMAL",
                "remote_path": remote_study,
                "local_path": str(local_study),
                "downloaded_files": 0,
                "status": "FAILED",
                "error": str(e)
            })

    ftp.quit()

    log_df = pd.DataFrame(logs)
    log_df.to_csv(LOG_CSV, index=False)

    print("\nNORMAL FTP download completed.")
    print("Success:", (log_df["status"] == "SUCCESS").sum())
    print("No files:", (log_df["status"] == "NO_FILES").sum())
    print("Failed:", (log_df["status"] == "FAILED").sum())
    print("Log saved:", LOG_CSV)


if __name__ == "__main__":
    main()