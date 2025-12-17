# 🚀 Guia Rápido: Rodando SMT no Kaggle

## 📋 Pré-requisitos
- Conta no Kaggle
- Notebook com **2x GPUs** (T4 ou P100)
- **30GB RAM**

## 🎯 Passos para Usar

### 1️⃣ Criar Novo Notebook no Kaggle
1. Acesse [kaggle.com/code](https://www.kaggle.com/code)
2. Clique em **"New Notebook"**
3. Configure:
   - **Accelerator**: GPU T4 x2 ou GPU P100 x2
   - **Internet**: ON (para clonar repositório)

### 2️⃣ Upload do Notebook
1. No Kaggle, clique em **File → Upload Notebook**
2. Selecione `kaggle_training_notebook.ipynb` deste repositório
3. Ou copie manualmente o conteúdo das células

### 3️⃣ Executar Células Sequencialmente

#### Célula 1: Instalar Dependências (⏱️ ~2 min)
```python
# Instala todas as bibliotecas necessárias
# Já otimizado para usar PyTorch nativo do Kaggle
```

#### Célula 2: Clonar Repositório (⏱️ ~30 seg)
```python
# Clona este repositório com código otimizado
# Detecta automaticamente pasta SMT existente
```

#### Célula 3: Verificar Configurações (⏱️ ~5 seg)
```python
# Verifica se configs existem
# Se não, cria automaticamente com valores otimizados
```

#### Célula 4: Baixar Dataset (⏱️ ~3-5 min)
```python
# Baixa PRAIG/grandstaff do HuggingFace
# ~455MB de dados
```

#### Célula 5: TESTE - Processar 5 Amostras (⏱️ ~30 seg) ⚠️ IMPORTANTE
```python
# TESTE CRÍTICO antes do treinamento completo
# Se falhar aqui, não prossiga!
```

#### Célula 6: Treinamento Completo (⏱️ ~2-6 horas)
```python
# Treina o modelo SMT com:
# - DDP (2 GPUs)
# - Mixed precision
# - Gradient accumulation
# - Early stopping
```

### 4️⃣ Monitoramento

Durante o treinamento, você verá:
```
Epoch X/10: 100%|██████████| steps/steps [XX:XX<XX:XX, X.XXit/s]
Train Loss: X.XXX | Val Loss: X.XXX | Val CER: XX.XX%
```

Checkpoints salvos em `/kaggle/working/logs/version_X/checkpoints/`

## ⚠️ Problemas Comuns

### ❌ Erro: "Map operation frozen at 9%"
**Causa:** `num_proc` > 0 ativa multiprocessing com serialização defeituosa

**Solução:** Já aplicada! Usamos `num_proc=None` em todas as configs

### ❌ Erro: "CUDA out of memory"
**Solução:** 
- Reduzir `batch_size` de 2 para 1 em `kaggle_config.json`
- Ou aumentar `reduce_ratio` para 0.3 (reduz tamanho das imagens)

### ❌ Erro: "numpy/scipy version mismatch"
**Solução:** Já aplicada! Forçamos numpy==1.26.4 e scipy==1.11.4 na célula 1

## 📊 Resultados Esperados

Após o treinamento completo:
- **CER (Character Error Rate)**: ~5-15% (menor é melhor)
- **Train Loss**: ~0.5-1.5
- **Val Loss**: ~0.8-2.0

Checkpoints salvos:
- `best-checkpoint.ckpt`: Melhor modelo (menor val_loss)
- `last-checkpoint.ckpt`: Último checkpoint

## 🎓 Próximos Passos

1. **Baixar Checkpoints**: 
   - No Kaggle, vá para Output → Download checkpoint
   
2. **Fazer Inferência**:
   - Use o modelo salvo para transcrever novas partituras
   
3. **Fine-tuning**:
   - Ajuste `config/FP-GrandStaff/kaggle_finetuning.json`
   - Carregue checkpoint do pré-treinamento

## 🐛 Debug Local (Opcional)

Se quiser testar localmente antes de subir no Kaggle:

```bash
# Configure cache para disco F: (se tiver espaço)
set HF_HOME=F:/huggingface_cache

# Execute teste local
python test_local.py
```

Isso testa:
1. Conversão PIL → numpy
2. cv2.resize funciona
3. Dataset PRAIG/grandstaff carrega
4. Processamento de amostra funciona

## 📚 Documentação Adicional

- **README.md**: Visão geral do projeto SMT
- **CONTRIBUTING.md**: Guia para contribuir
- **config/**: Arquivos de configuração com parâmetros detalhados

## 🆘 Suporte

Se encontrar problemas:
1. Verifique a seção "Diagnóstico" no notebook
2. Leia os comentários nas células de código
3. Abra uma issue no GitHub com logs completos

---

**Última atualização**: 2024
**Testado em**: Kaggle (GPU T4 x2, 30GB RAM)
