import os, cv2, numpy as np, pandas as pd, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import albumentations as A
import segmentation_models_pytorch as smp
from torchvision import models, transforms
from tqdm import tqdm

# === 1. CONFIG ===
TRAIN_CSV_PATH = r'C:\Users\rajas\OneDrive\Desktop\airbus ship detection DRDO\train_ship_segmentations_v2.csv'
TRAIN_IMG_DIR = r'C:\Users\rajas\OneDrive\Desktop\airbus ship detection DRDO\train_v2'
TEST_IMG_DIR = r'C:\Users\rajas\OneDrive\Desktop\airbus ship detection DRDO\test_v2'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# === 2. RLE DECODE ===
def rle_decode(rle, shape=(768,768)):
    if pd.isna(rle): return np.zeros(shape, np.uint8)
    s = list(map(int, rle.split()))
    img = np.zeros(shape[0]*shape[1], np.uint8)
    for i in range(0, len(s), 2):
        img[s[i]-1:s[i]-1+s[i+1]] = 1
    return img.reshape(shape).T

# === 3. CLASSIFIER DATASET ===
class ShipClassifierDataset(Dataset):
    def __init__(self, df, img_dir, transform):
        self.df = df
        self.ids = df.ImageId.unique()
        self.labels = df.groupby('ImageId')['EncodedPixels'].apply(lambda x: x.notna().any())
        self.img_dir = img_dir
        self.transform = transform
    
    def __len__(self): return len(self.ids)
    
    def __getitem__(self, idx):
        img_id = self.ids[idx]
        img = cv2.imread(os.path.join(self.img_dir, img_id))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (128,128))
        img = self.transform(img)
        label = float(self.labels[img_id])
        return img, torch.tensor(label)

# === 4. SEGMENTATION DATASET ===
class ShipSegDataset(Dataset):
    def __init__(self, image_list, img_dir, df, size, transform):
        self.ids = image_list
        self.img_dir = img_dir
        self.df = df
        self.size = size
        self.transform = transform
    
    def __len__(self): return len(self.ids)
    
    def __getitem__(self, idx):
        img_id = self.ids[idx]
        img = cv2.cvtColor(cv2.imread(os.path.join(self.img_dir, img_id)), cv2.COLOR_BGR2RGB)
        mask = np.zeros((768,768), np.uint8)
        for rle in self.df.loc[self.df.ImageId == img_id, 'EncodedPixels']:
            mask = np.clip(mask + rle_decode(rle), 0, 1)
        
        ys, xs = np.where(mask)
        if len(xs):
            x0, x1 = xs.min(), xs.max()
            y0, y1 = ys.min(), ys.max()
            cx, cy = (x0+x1)//2, (y0+y1)//2
        else: cx,cy = 384,384
        x0, y0 = max(0, cx-self.size//2), max(0, cy-self.size//2)
        crop_img = img[y0:y0+self.size, x0:x0+self.size]
        crop_m = mask[y0:y0+self.size, x0:x0+self.size]
        crop_img = cv2.resize(crop_img, (self.size, self.size))
        crop_m = cv2.resize(crop_m, (self.size, self.size))
        
        if self.transform:
            aug = self.transform(image=crop_img, mask=crop_m)
            crop_img, crop_m = aug['image'], aug['mask']
        return (torch.tensor(crop_img.transpose(2,0,1)/255., dtype=torch.float),
                torch.tensor(crop_m[None,:,:], dtype=torch.float))

# === 5. METRICS & LOSS ===
dice = smp.losses.DiceLoss(mode='binary')
def seg_loss(pred, true):
    bce_pos = torch.tensor([5.0], device=pred.device)
    bce = nn.BCEWithLogitsLoss(pos_weight=bce_pos)(pred, true)
    return 0.5*bce + 0.5*dice(pred, true)

# === 6. MAIN ===
if __name__ == "__main__":
    print("Device:", DEVICE)
    
    df = pd.read_csv(TRAIN_CSV_PATH)
    df['has_ship'] = df['EncodedPixels'].notna()
    
    # === 6A. Classifier Stage ===
    clf_tx = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    clf_ds = ShipClassifierDataset(df, TRAIN_IMG_DIR, clf_tx)
    clf_loader = DataLoader(clf_ds, batch_size=64, shuffle=True, num_workers=4)
    clf_model = models.resnet34(pretrained=True)
    clf_model.fc = nn.Linear(clf_model.fc.in_features, 1)
    clf_model = clf_model.to(DEVICE)
    opt = torch.optim.Adam(clf_model.parameters(), lr=1e-3)
    crit = nn.BCEWithLogitsLoss()
    
    print("Starting classifier training...")
    clf_model.train()
    for epoch in range(3):
        epoch_loss = 0
        for x,y in tqdm(clf_loader, desc=f"Classifier Epoch {epoch+1}/3"):
            x,y = x.to(DEVICE), y.unsqueeze(1).to(DEVICE)
            loss = crit(clf_model(x), y)
            opt.zero_grad(); loss.backward(); opt.step()
            epoch_loss += loss.item()
        print(f"Classifier Epoch {epoch+1} Loss: {epoch_loss/len(clf_loader):.4f}")
    
    clf_model.eval()
    has_ship, img_ids = [], clf_ds.ids
    with torch.no_grad():
        for x,_ in DataLoader(clf_ds, batch_size=64, num_workers=4):
            pr = torch.sigmoid(clf_model(x.to(DEVICE))).cpu().squeeze()
            has_ship.extend(pr>0.5)
    ship_images = [i for i,keep in zip(img_ids,has_ship) if keep]
    print("Ship images:", len(ship_images), " / ", len(img_ids))
    
    # === 6B. Segmentation Stage ===
    sizes = [256,384,512]
    best_loss = float('inf')
    try:
        for stage, size in enumerate(sizes):
            tx = A.Compose([A.HorizontalFlip(),A.VerticalFlip(),A.RandomRotate90()])
            ds = ShipSegDataset(ship_images, TRAIN_IMG_DIR, df, size, tx)
            loader = DataLoader(ds, batch_size=16, shuffle=True, num_workers=4)
            
            model = smp.Unet('resnet34', encoder_weights='imagenet', classes=1).to(DEVICE)
            params = [
                {'params': model.encoder.parameters(), 'lr':1e-4},
                {'params': model.decoder.parameters(), 'lr':1e-3},
            ]
            opt = torch.optim.Adam(params)
            sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, 'min', patience=2)
            
            for e in range(stage*4+1, stage*4+5):
                model.train()
                tot_loss = 0
                for img,mask in tqdm(loader, desc=f"Seg Stage {stage+1}, Size {size}, Epoch {e}"):
                    img,mask = img.to(DEVICE),mask.to(DEVICE)
                    pr = model(img)
                    loss = seg_loss(pr, mask)
                    opt.zero_grad(); loss.backward(); opt.step()
                    tot_loss += loss.item()
                avg = tot_loss/len(loader)
                print(f"Seg Stage {stage+1} Epoch {e}: Loss={avg:.4f}")
                sched.step(avg)
                
                if avg < best_loss:
                    best_loss = avg
                    torch.save(model.state_dict(), 'best_model.pth')
                    print(f"✅ Best model saved at Stage {stage+1} Epoch {e}")
                    
    except KeyboardInterrupt:
        print("Training interrupted. Saving last model state...")
        torch.save(model.state_dict(), 'interrupted_model.pth')
    
    print("✅ Training complete!")
