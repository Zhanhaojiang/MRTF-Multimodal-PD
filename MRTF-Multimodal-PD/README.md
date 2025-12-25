# MRTF — Multimodal Reinforcement-Assisted Transformer Framework (PD Detection)

این ریپازیتوری یک **پیاده‌سازی مرجع (Reference Implementation)** از چارچوب پیشنهادی MRTF بر اساس دیاگرام/روش ارائه‌شده است.

## ایده اصلی
- سه مدالیته: **Voice** (اسپکتروگرام)، **MRI** (تصویر)، **Sensor** (سری‌زمانی پوشیدنی)
- استخراج ویژگی:
  - Voice Encoder: CNN + BiLSTM
  - MRI Encoder: ViT (با fallback به CNN)
  - Sensor Encoder: TCN + Temporal Transformer
- همجوشی: **CAFT** (Tri-directional Cross Attention) + Transformer Encoder
- XAI: attribution (IG fallback برای SHAP) + Counterfactual + ECS
- RASL: بهینه‌سازی کمک‌گرفته از RL برای آستانه/پاداش و بهبود همجوشی

## ساختار پروژه
```
MRTF-Multimodal-PD/
  notebooks/              # نوت‌بوک‌های آموزشی/دمو
  src/mrtf/               # کد ماژولار پایتون
  scripts/                # اسکریپت‌های اجرا/آموزش
  data/raw/               # داده خام (در گیت ignore شده)
  data/processed/         # داده پردازش‌شده (در گیت ignore شده)
  configs/                # تنظیمات و کانفیگ‌ها
  docs/                   # مستندات تکمیلی
```

## شروع سریع
### 1) نصب
```bash
python -m venv .venv
source .venv/bin/activate   # روی ویندوز: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

### 2) اجرای نوت‌بوک
- `notebooks/MRTF_implementation.ipynb`

### 3) اجرای آموزش دمو (Dummy)
```bash
python scripts/train_dummy.py --epochs 3 --batch_size 8
```

## استفاده با دیتای واقعی
در نوت‌بوک، توابع زیر را با pipeline واقعی خودتان جایگزین کنید:
- `preprocess_voice` (STFT/MFCC, normalization, augmentation, ...)
- `preprocess_mri` (skull stripping, bias correction, resize, intensity norm, ...)
- `preprocess_sensor` (windowing, wavelet denoise, FFT features, ...)

## License
MIT — فایل `LICENSE` را ببینید.
