import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
from unet_segmentation_pipeline import UNet, DEVICE, IMAGE_SIZE, COLOR_MAP, NUM_CLASSES, map_to_rgb

CHECKPOINT_PATH = 'dataset/checkpoints/unet_checkpoint.pth'
INPUT_DIR = 'input_ujicoba'
HASIL_DIR = 'input_ujicoba'

def predict_large_image(model, image_pil, patch_size=(256, 256)):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    img_width, img_height = image_pil.size
    stride_x, stride_y = patch_size[0], patch_size[1]
    
    full_prediction = np.zeros((img_height, img_width), dtype=np.uint8)
    
    model.eval()
    with torch.no_grad():
        for y in range(0, img_height, stride_y):
            for x in range(0, img_width, stride_x):
                y_end = min(y + stride_y, img_height)
                x_end = min(x + stride_x, img_width)
                
                patch_pil = image_pil.crop((x, y, x_end, y_end))
                
                patch_w, patch_h = patch_pil.size
                if patch_w != patch_size[0] or patch_h != patch_size[1]:
                    patch_pil = patch_pil.resize(patch_size)
                    
                input_tensor = transform(patch_pil).unsqueeze(0).to(DEVICE)
                output = model(input_tensor)
                
                predicted_patch = torch.argmax(output.squeeze(), dim=0).cpu().numpy()
                
                if patch_w != patch_size[0] or patch_h != patch_size[1]:
                    predicted_patch_img = Image.fromarray(predicted_patch.astype(np.uint8))
                    predicted_patch_img = predicted_patch_img.resize((patch_w, patch_h), resample=Image.NEAREST)
                    predicted_patch = np.array(predicted_patch_img)
                    
                full_prediction[y:y_end, x:x_end] = predicted_patch
                
    return full_prediction

def run_inference():
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"File checkpoint belum tersedia di: {CHECKPOINT_PATH}")
        print("Tunggu minimal model menyelesaikan 1 Epoch pada skrip training utama.")
        return
        
    valid_extensions = ('.png', '.jpg', '.jpeg', '.tif', '.tiff')
    
    if not os.path.exists(INPUT_DIR):
        os.makedirs(INPUT_DIR)
        
    test_images = [os.path.join(INPUT_DIR, f) for f in os.listdir(INPUT_DIR) 
                   if f.lower().endswith(valid_extensions)]
                   
    if len(test_images) == 0:
        print(f"Silakan masukkan gambar-gambar baru yang ingin diuji ke dalam folder: {INPUT_DIR}")
        return
        
    print("Memuat arsitektur model...")
    model = UNet(in_channels=3, out_classes=NUM_CLASSES).to(DEVICE)
    
    print(f"Membaca otak model dari checkpoint: {CHECKPOINT_PATH}")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Model berhasil dimuat! Berasal dari hasil belajar sampai Epoch: {checkpoint['epoch'] + 1}")

    print(f"Ditemukan {len(test_images)} gambar untuk diuji coba. Memulai inference presisi tinggi...\n")

    model.eval()
    
    for image_path in test_images:
        print(f"Menebak gambar (Bisa memakan waktu untuk citra sangat besar): {os.path.basename(image_path)}")
        image_pil = Image.open(image_path).convert("RGB")
        
        predicted_classes = predict_large_image(model, image_pil, patch_size=IMAGE_SIZE)
        
        rgb_segmentation = map_to_rgb(predicted_classes)
        
        import matplotlib.patches as mpatches
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 7))
        
        axes[0].imshow(image_pil)
        axes[0].set_title(f"Satelit Asli: {os.path.basename(image_path)}")
        axes[0].axis('off')
        
        axes[1].imshow(image_pil)
        axes[1].imshow(rgb_segmentation, alpha=0.45) 
        axes[1].set_title(f"Prediksi Model (Epoch {checkpoint['epoch'] + 1})")
        axes[1].axis('off')

        labels = ["Pasir/Bg/Jalan (0)", "Bangunan (1)", "Vegetasi (2)", "Laut (3)"]
        legend_elements = [mpatches.Patch(color=np.array(COLOR_MAP[i])/255., label=labels[i]) for i in range(4)]
        axes[1].legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.01, 1), title="Warna Label")
        
        plt.tight_layout()
        
        plt.show()  
        plt.close(fig) 

    print("\nSemua perhitungan uji coba presisi tinggi selesai diproses!")

if __name__ == '__main__':
    run_inference()
