import os
import cv2
import numpy as np
import torch
import lpips
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

# ==========================================
# 【設定】比較するフォルダの住所や！
# ========================================== 
REAL_DIR = "data_mns/data_3_p1/data/data_3_p1"
# 本物の写真
AI_DIR = "renders"                    
# NGPが作った写真

print("カァーッ！ 採点官を叩き起こしています...（VGGモデル読み込み中）")
# GPUが使えるならGPUのフルパワーで計算させるで！
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
loss_fn_vgg = lpips.LPIPS(net='vgg').to(device)

# フォルダの中の画像リストを取得して、名前順にピシッと並び替える
real_imgs = sorted([f for f in os.listdir(REAL_DIR) if f.endswith(('.png', '.jpg', '.JPG'))])
ai_imgs = sorted([f for f in os.listdir(AI_DIR) if f.endswith(('.png', '.jpg'))])

if len(real_imgs) == 0 or len(ai_imgs) == 0:
    print("アホか！ 画像が見つからんぞ！ フォルダのパスをもう一回確認せえ！")
    exit()

compare_count = min(len(real_imgs), len(ai_imgs))
print(f"🔥 評価開始！ {compare_count}枚の画像をガチンコ比較するで！\n")

total_psnr = 0.0
total_ssim = 0.0
total_lpips = 0.0

# 1枚ずつ順番に呼び出して採点していくループや！
for i in range(compare_count):
    real_path = os.path.join(REAL_DIR, real_imgs[i])
    ai_path = os.path.join(AI_DIR, ai_imgs[i])
    
    # 1. OpenCVで画像を読み込む（PSNR/SSIM計算用）
    img_real_cv = cv2.imread(real_path)
    img_ai_cv = cv2.imread(ai_path)
    
    # サイズが違うと計算できんからチェックや！
    if img_real_cv.shape != img_ai_cv.shape:
        print(f"⚠️ エラー: {real_imgs[i]} と {ai_imgs[i]} の解像度がちゃうぞ！ スキップや！")
        continue
        
    # 2. PSNR と SSIM を計算！
    # （SSIMはカラー画像の場合、channel_axis=2を指定せなアカン！）
    current_psnr = psnr(img_real_cv, img_ai_cv)
    current_ssim = ssim(img_real_cv, img_ai_cv, channel_axis=2)
    
    # 3. LPIPS用に画像をPyTorchのテンソル（AIのデータ形式）に変換！
    # OpenCVは「青・緑・赤」の順やから「赤・緑・青」に直して、数値を [-1, 1] にするんやで
    img_real_rgb = cv2.cvtColor(img_real_cv, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img_ai_rgb = cv2.cvtColor(img_ai_cv, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    
    t_real = torch.from_numpy(img_real_rgb).permute(2, 0, 1).unsqueeze(0) * 2.0 - 1.0
    t_ai = torch.from_numpy(img_ai_rgb).permute(2, 0, 1).unsqueeze(0) * 2.0 - 1.0
    
    # 採点官（LPIPS）による評価！
    current_lpips = loss_fn_vgg(t_real.to(device), t_ai.to(device)).item()
    
    # 各スコアを加算していく
    total_psnr += current_psnr
    total_ssim += current_ssim
    total_lpips += current_lpips
    
    print(f"[{i+1:02d}/{compare_count:02d}] {ai_imgs[i]} | PSNR: {current_psnr:.2f}, SSIM: {current_ssim:.4f}, LPIPS: {current_lpips:.4f}")

# 最終結果発表！
print("\n=========================================")
print("🎉 最終結果（平均スコア） 🎉")
print(f"📈 PSNR  : {total_psnr / compare_count:.2f} dB (高いほど画質ヨシ！ 30以上で優秀)")
print(f"👁️ SSIM  : {total_ssim / compare_count:.4f} (1に近いほど構造が本物そっくり！)")
print(f"🧠 LPIPS : {total_lpips / compare_count:.4f} (0に近いほど人間の感覚で本物そっくり！)")
print("=========================================")