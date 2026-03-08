import cv2
import torch
import einops
import torch.nn as nn
import torchvision.models as models
from clock_utils import warp
import os

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    image_folder = r"D:\Bai tap\Visual Studio for Python\watch_photos_output\Clock_1"
    output_folder = r"D:\Bai tap\Visual Studio for Python\watch_photos_output\Clock_1\output"

    resume_path = r"C:\Users\VBK computer\Downloads\itsabouttime-main\models\full.pth"
    stn_resume_path = r"C:\Users\VBK computer\Downloads\itsabouttime-main\models\full_st.pth"

    os.makedirs(output_folder, exist_ok=True)

    model_stn = models.resnet50(pretrained=False)
    model_stn.fc = nn.Linear(2048, 8)

    model = models.resnet50(pretrained=False)
    model.fc = nn.Linear(2048, 720)

    model.load_state_dict(torch.load(resume_path, map_location=device))
    model_stn.load_state_dict(torch.load(stn_resume_path, map_location=device))

    model.to(device).eval()
    model_stn.to(device).eval()

    image_paths = [
        os.path.join(image_folder, f)
        for f in os.listdir(image_folder)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ]

    if len(image_paths) == 0:
        print("No images found!")
        return

    for img_path in image_paths:
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            print("Failed to read:", img_path)
            continue

        h0, w0 = img_bgr.shape[:2]

        img = cv2.resize(img_bgr, (224, 224)) / 255.0
        img = einops.rearrange(img, 'h w c -> c h w')
        img = torch.tensor(img, dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            pred_st = model_stn(img)
            pred_st = torch.cat(
                [pred_st, torch.ones(1, 1, device=device)], dim=1
            )
            Minv_pred = pred_st.view(-1, 3, 3)
            img_warped = warp(img, Minv_pred)

            pred = model(img_warped)
            idx = torch.argmax(pred, dim=1)[0]

            hour = (idx // 60).item()
            minute = (idx % 60).item()
            
        text = f"{hour:02d}:{minute:02d}"
        cv2.putText(
            img_bgr,
            text,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )

        out_path = os.path.join(output_folder, os.path.basename(img_path))
        cv2.imwrite(out_path, img_bgr)

        print(f"Saved: {out_path} -> {text}")

    print("\nDone.")

if __name__ == "__main__":
    main()

