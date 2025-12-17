#!/usr/bin/env python3
"""
Script para verificar se o código está atualizado com as otimizações.
Execute no Kaggle para confirmar que está usando a versão correta.
"""

def check_num_proc():
    """Verifica se num_proc está configurado como None"""
    import data
    import SynthGenerator
    
    issues = []
    
    # Verificar data.py
    try:
        import inspect
        
        # Verificar load_set
        source = inspect.getsource(data.load_set)
        if 'num_proc=None' not in source:
            issues.append("❌ data.load_set ainda usa num_proc != None")
        else:
            print("✅ data.load_set: num_proc=None")
            
        # Verificar load_from_files_list
        source = inspect.getsource(data.load_from_files_list)
        if 'num_proc' in source and 'None' in source:
            print("✅ data.load_from_files_list: num_proc padrão = None")
        else:
            issues.append("❌ data.load_from_files_list: problema com num_proc")
            
    except Exception as e:
        issues.append(f"⚠️ Erro verificando data.py: {e}")
    
    # Verificar SynthGenerator.py
    try:
        source = inspect.getsource(SynthGenerator.load_from_files_list)
        if 'num_proc=None' not in source:
            issues.append("❌ SynthGenerator.load_from_files_list ainda usa num_proc != None")
        else:
            print("✅ SynthGenerator.load_from_files_list: num_proc=None")
    except Exception as e:
        issues.append(f"⚠️ Erro verificando SynthGenerator.py: {e}")
    
    return issues

def check_max_functions():
    """Verifica se get_max_* usam valores fixos"""
    import data
    import inspect
    
    issues = []
    
    try:
        # Verificar get_max_height
        source = inspect.getsource(data.GrandStaffFullPage.get_max_height)
        if 'for s in self.data' in source:
            issues.append("❌ get_max_height ainda itera o dataset")
        else:
            print("✅ get_max_height: otimizado (valor fixo)")
            
        # Verificar get_max_width  
        source = inspect.getsource(data.GrandStaffFullPage.get_max_width)
        if 'for s in self.data' in source:
            issues.append("❌ get_max_width ainda itera o dataset")
        else:
            print("✅ get_max_width: otimizado (valor fixo)")
            
        # Verificar get_max_seqlen
        source = inspect.getsource(data.GrandStaffFullPage.get_max_seqlen)
        if 'for s in self.data' in source or 'for seq in' in source:
            issues.append("❌ get_max_seqlen ainda itera o dataset")
        else:
            print("✅ get_max_seqlen: otimizado (valor fixo)")
            
    except Exception as e:
        issues.append(f"⚠️ Erro verificando get_max_*: {e}")
    
    return issues

def main():
    print("=" * 80)
    print("🔍 VERIFICAÇÃO DE VERSÃO DO CÓDIGO SMT")
    print("=" * 80)
    print()
    
    all_issues = []
    
    print("📋 Verificando configurações num_proc...")
    all_issues.extend(check_num_proc())
    print()
    
    print("📋 Verificando funções get_max_*...")
    all_issues.extend(check_max_functions())
    print()
    
    print("=" * 80)
    if all_issues:
        print("❌ PROBLEMAS ENCONTRADOS:")
        for issue in all_issues:
            print(f"   {issue}")
        print()
        print("⚠️ AÇÃO NECESSÁRIA:")
        print("   1. Faça git pull para atualizar o código")
        print("   2. Reinicie o kernel Python")
        print("   3. Execute este script novamente")
    else:
        print("✅ TUDO OK! Código está atualizado com todas as otimizações.")
        print("   Pode prosseguir com o treinamento!")
    print("=" * 80)

if __name__ == "__main__":
    main()
