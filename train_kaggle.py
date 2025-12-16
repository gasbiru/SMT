import fire
import json
import torch
import gc
from data import SyntheticCLGrandStaffDataset, GrandStaffFullPageCurriculumLearning
from smt_trainer import SMT_Trainer

from ExperimentConfig import experiment_config_from_dict
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.strategies import DDPStrategy

torch.set_float32_matmul_precision('high')

def main(config_path, use_wandb=False, max_epochs=100):
    """
    Script otimizado para treinamento no Kaggle com 2 GPUs e 30GB RAM
    
    Args:
        config_path: Caminho para arquivo de configuração JSON
        use_wandb: Se True, usa Weights & Biases para logging
        max_epochs: Número máximo de épocas
    """
    
    # Limpar cache
    torch.cuda.empty_cache()
    gc.collect()
    
    with open(config_path, "r") as f:
        config = experiment_config_from_dict(json.load(f))

    print(f"🚀 Configuração carregada:")
    print(f"   - Dataset: {config.data.data_path}")
    print(f"   - Batch size: {config.data.batch_size}")
    print(f"   - Num workers: {config.data.num_workers}")
    print(f"   - Reduce ratio: {config.data.reduce_ratio}")

    datamodule = SyntheticCLGrandStaffDataset(config=config.data, skip_steps=0)

    max_height = datamodule.get_max_height()
    max_width = datamodule.get_max_width()
    max_len = datamodule.get_max_length()

    print(f"📏 Dimensões do modelo:")
    print(f"   - Max height: {max_height}")
    print(f"   - Max width: {max_width}")
    print(f"   - Max length: {max_len}")

    model_wrapper = SMT_Trainer(
        maxh=int(max_height), 
        maxw=int(max_width), 
        maxlen=int(max_len),
        out_categories=len(datamodule.train_set.w2i), 
        padding_token=datamodule.train_set.w2i["<pad>"],
        in_channels=1, 
        w2i=datamodule.train_set.w2i, 
        i2w=datamodule.train_set.i2w,
        d_model=256, 
        dim_ff=256, 
        num_dec_layers=8
    )

    # Configurar logger
    if use_wandb:
        group = config.checkpoint.dirpath.split("/")[-1]
        wandb_logger = WandbLogger(
            project='SMT-Kaggle', 
            group=group, 
            name="SMT-FP-Kaggle-2xGPU", 
            log_model=False
        )
    else:
        wandb_logger = None
        print("⚠️ WandB desabilitado - use use_wandb=True para habilitar")

    early_stopping = EarlyStopping(
        monitor="val_SER", 
        min_delta=0.01, 
        patience=10,  # Aumentado para dar mais chances
        mode="min", 
        verbose=True
    )

    checkpointer = ModelCheckpoint(
        dirpath=config.checkpoint.dirpath, 
        filename=config.checkpoint.filename,
        monitor=config.checkpoint.monitor, 
        mode=config.checkpoint.mode,
        save_top_k=config.checkpoint.save_top_k, 
        verbose=config.checkpoint.verbose,
        save_last=True  # Salvar último checkpoint também
    )

    # Estratégia DDP para 2 GPUs
    strategy = DDPStrategy(
        find_unused_parameters=False,
        gradient_as_bucket_view=True  # Otimização de memória
    )

    # Configurar Trainer otimizado para Kaggle
    trainer = Trainer(
        max_epochs=max_epochs,
        check_val_every_n_epoch=2,  # Validar a cada 2 épocas
        logger=wandb_logger, 
        callbacks=[checkpointer, early_stopping], 
        precision='16-mixed',  # Mixed precision para economia de memória
        accelerator='gpu',
        devices=2,  # 2 GPUs no Kaggle
        strategy=strategy,
        accumulate_grad_batches=4,  # Gradient accumulation (batch efetivo = 2*2*4 = 16)
        gradient_clip_val=1.0,  # Clip gradientes
        log_every_n_steps=50,
        enable_progress_bar=True,
        enable_model_summary=True,
        # Otimizações de memória
        enable_checkpointing=True,
        deterministic=False,  # Mais rápido
    )

    print("\n🎯 Iniciando treinamento...")
    print(f"   - Dispositivos: {trainer.num_devices} GPUs")
    print(f"   - Batch efetivo: {config.data.batch_size * trainer.num_devices * trainer.accumulate_grad_batches}")
    print(f"   - Precisão: {trainer.precision}")
    
    try:
        trainer.fit(model_wrapper, datamodule=datamodule)
        
        print("\n✅ Treinamento concluído!")
        print(f"   - Melhor checkpoint: {checkpointer.best_model_path}")
        
        # Testar modelo
        if checkpointer.best_model_path:
            print("\n🧪 Testando melhor modelo...")
            model = SMT_Trainer.load_from_checkpoint(checkpointer.best_model_path)
            trainer.test(model, datamodule=datamodule)
            
    except KeyboardInterrupt:
        print("\n⚠️ Treinamento interrompido pelo usuário")
    except Exception as e:
        print(f"\n❌ Erro durante treinamento: {e}")
        raise
    finally:
        # Limpar memória
        torch.cuda.empty_cache()
        gc.collect()

def launch(config_path, use_wandb=False, max_epochs=100):
    main(config_path, use_wandb, max_epochs)

if __name__ == "__main__":
    fire.Fire(launch)
