from pathlib import Path
import kagglehub
import shutil

def main():
    target_dir = Path("data/raw")
    target_dir.mkdir(parents=True, exist_ok=True)

    print("Downloading UK-DALE dataset...")
    path = Path(kagglehub.dataset_download("abdelmdz/uk-dale"))

    print(f"Dataset downloaded to: {path}")


    for item in path.iterdir():
        dest = target_dir / item.name
        if item.is_dir():
            shutil.copytree(item, dest, dirs_exist_ok=True)
        else:
            shutil.copy2(item, dest)
    




if __name__ == "__main__":
    main()
