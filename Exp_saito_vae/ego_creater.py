import os
import re
import pandas as pd
import cv2

file_list = ['video20220722']#,'video20220729', 'video20220801', 'video20221110', 'video20230315']  # 

for f in file_list:
    print(f"\n🔍 Processing video: {f}")
    path = rf'/baksv/CIGIT/GXN_Liuxy/LenSe/{f}/screenshots'

    ego_dir = os.path.join(path, 'ego')
    os.makedirs(ego_dir, exist_ok=True)

    numeric_id = re.sub(r'\D', '', f)
    csv_path = f'total_data_{numeric_id}.csv'
    
    if not os.path.exists(csv_path):
        print(f"❌ CSV file not found: {csv_path}")
        continue

    df = pd.read_csv(csv_path)

    success_count = 0
    fail_count = 0

    for i in range(len(df)):
        try:
            frame_id = int(df.iloc[i]['time'])
            label = int(df.iloc[i]['label'])

            if label <= 0 or label > 6:
                print(f"⚠️ Invalid label {label} at frame {frame_id} in file {f}")
                fail_count += 1
                continue

            ego_path = os.path.join(path, 'ego', f"ego_screenshot_{frame_id:04d}.jpg")
            cam_path = os.path.join(path, f"{label}", f"len{label}_screenshot_{frame_id:04d}.jpg")

            if not os.path.exists(cam_path):
                print(f"❌ Camera image not found: {cam_path}")
                fail_count += 1
                continue

            img = cv2.imread(cam_path)
            if img is None:
                print(f"❌ Failed to read image: {cam_path}")
                fail_count += 1
                continue

            cv2.imwrite(ego_path, img)
            success_count += 1

        except Exception as e:
            print(f"❌ Exception at index {i} in {f}: {e}")
            fail_count += 1

    print(f"✅ Done: {success_count} saved, {fail_count} failed")
