"""
Script de teste local para diagnosticar problema do cv2.resize
"""
import os
os.environ['HF_HOME'] = 'F:/huggingface_cache'  # Cache no disco F
os.environ['HF_DATASETS_CACHE'] = 'F:/huggingface_cache/datasets'

import numpy as np
import cv2
import datasets
from PIL import Image

print("=" * 80)
print("🧪 TESTE LOCAL - Diagnóstico cv2.resize")
print("=" * 80)

# Teste 1: Conversão PIL → numpy → cv2.resize diretamente
print("\n📋 Teste 1: PIL Image → numpy → cv2.resize (sem dataset)")
print("-" * 80)

# Criar uma imagem PIL de teste
pil_img = Image.new('RGB', (640, 480), color=(73, 109, 137))
print(f"✅ PIL Image criada: {type(pil_img)}, mode={pil_img.mode}, size={pil_img.size}")

# Converter para numpy
np_img = np.array(pil_img, dtype=np.uint8, copy=True)
print(f"✅ Numpy array: {type(np_img)}, shape={np_img.shape}, dtype={np_img.dtype}")
print(f"   Flags: C_CONTIGUOUS={np_img.flags['C_CONTIGUOUS']}, OWNDATA={np_img.flags['OWNDATA']}")

# Tentar resize
try:
    resized = cv2.resize(np_img, (320, 240))
    print(f"✅ cv2.resize OK: shape={resized.shape}")
except Exception as e:
    print(f"❌ cv2.resize FALHOU: {e}")

# Teste 2: Carregar dataset real
print("\n📋 Teste 2: Dataset PRAIG/grandstaff (apenas 5 amostras)")
print("-" * 80)

try:
    print("📥 Carregando dataset...")
    test_ds = datasets.load_dataset(
        "PRAIG/grandstaff",
        split="train[:5]",
        trust_remote_code=False,
        cache_dir="F:/huggingface_cache/datasets"
    )
    print(f"✅ Dataset carregado: {len(test_ds)} amostras")
    
    # Testar primeira imagem
    img = test_ds[0]['image']
    print(f"\n🖼️  Imagem do dataset:")
    print(f"   Tipo: {type(img)}")
    print(f"   Mode: {img.mode if hasattr(img, 'mode') else 'N/A'}")
    print(f"   Size: {img.size if hasattr(img, 'size') else 'N/A'}")
    
    # Converter
    print("\n🔄 Convertendo para numpy...")
    if hasattr(img, 'mode'):
        if img.mode != 'RGB':
            img = img.convert('RGB')
        np_img2 = np.array(img, dtype=np.uint8, copy=True)
    else:
        np_img2 = np.array(img, dtype=np.uint8, copy=True)
    
    print(f"✅ Numpy array: shape={np_img2.shape}, dtype={np_img2.dtype}")
    print(f"   Flags: C_CONTIGUOUS={np_img2.flags['C_CONTIGUOUS']}, OWNDATA={np_img2.flags['OWNDATA']}")
    
    # Tentar resize
    print("\n🔄 Testando cv2.resize...")
    width = int(np_img2.shape[1] * 0.5)
    height = int(np_img2.shape[0] * 0.5)
    print(f"   Tamanho original: {np_img2.shape[:2]}")
    print(f"   Tamanho alvo: ({height}, {width})")
    
    resized2 = cv2.resize(np_img2, (width, height))
    print(f"✅ cv2.resize OK: shape={resized2.shape}")
    
except Exception as e:
    print(f"❌ ERRO: {e}")
    import traceback
    traceback.print_exc()

# Teste 3: Testar com a função prepare_fp_data
print("\n📋 Teste 3: Usando função prepare_fp_data")
print("-" * 80)

try:
    from data import prepare_fp_data
    
    # Testar com uma amostra
    sample = {'image': test_ds[0]['image'], 'transcription': test_ds[0]['transcription']}
    print(f"📥 Amostra: imagem tipo={type(sample['image'])}")
    
    result = prepare_fp_data(sample, reduce_ratio=0.5, krn_format='bekern')
    
    print(f"✅ prepare_fp_data OK!")
    print(f"   Imagem processada: type={type(result['image'])}, shape={result['image'].shape}")
    
except Exception as e:
    print(f"❌ ERRO em prepare_fp_data: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
print("🏁 Teste concluído!")
print("=" * 80)

# Verificar versões
print("\n📦 Versões:")
print(f"   NumPy: {np.__version__}")
print(f"   OpenCV: {cv2.__version__}")
print(f"   Datasets: {datasets.__version__}")
