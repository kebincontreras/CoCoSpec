import os

def check_specim_iq_files(data_dir):
    print("Checking files for specim_iq in all scenes...\n")
    for i in range(1, 20):
        folder = os.path.join(data_dir, "scenes", f"scene_{i:02d}", "specim_iq")
        files_needed = [
            "hsi_closed.dat", "hsi_open.dat",
            "annotations_closed.txt", "annotations_open.txt"
        ]
        print(f"\nScene {i:02d}:")
        if not os.path.exists(folder):
            print(f"  ❌ Folder does not exist: {folder}")
            continue
        for fname in files_needed:
            fpath = os.path.join(folder, fname)
            if os.path.exists(fpath):
                print(f"  ✅ Found: {fname}")
            else:
                print(f"  ❌ Missing: {fname}")

if __name__ == "__main__":
    # Adjust this path if your data folder is elsewhere
    base_data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    check_specim_iq_files(os.path.abspath(base_data_dir))
