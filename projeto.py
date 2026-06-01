#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# # Two-Stage Wrapper Feature Selection: Estágio 1 (Pruning)
# Implementação de três abordagens de poda de meta-features para redução de dimensionalidade antes da aplicação de um Algoritmo Genético.
# 1. **V1:** Baseline (Força Bruta)
# 2. **V2:** Hard Pruning (Random Forest Padrão)
# 3. **V3:** Enriched Random Forest (Amostragem Ponderada)

# In[ ]:


import os
import sys
import time
import csv
import warnings
import openml
import socket

os.environ["PYTHONWARNINGS"] = "ignore"
# Forçar output UTF-8 para evitar erros de charmap no Windows com emojis
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import concurrent.futures
import threading

from pymfe.mfe import MFE
from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold, LeaveOneOut
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import f_classif
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from sklearn.utils.multiclass import type_of_target

# In[]:
lock_csv = threading.Lock()
socket.setdefaulttimeout(180)

def processar_um_dataset(dataset_name, modelos_base, cv_base, arquivo_log):
    t_id = threading.get_ident() % 10000 
    print(f"▶️ [Th-{t_id}] Iniciando: {dataset_name}...", flush=True)
    try:
        # 1. Download acontece AQUI DENTRO da thread
        dataset = fetch_openml(name=dataset_name, as_frame=True, parser='auto')
        X, y = dataset.data.copy(), dataset.target.copy()
        
        # 2. Validação inicial
        if y is None or type_of_target(y) in ['continuous', 'continuous-multioutput']:
            print(f"⏩ [Th-{t_id}] {dataset_name} ignorado (Sem target ou Regressão).")
            return False

        # Prevenção de warnings: remove colunas 100% vazias
        X = X.dropna(axis=1, how='all')

        # 3. Tratamento inicial de strings
        str_cols = X.select_dtypes(include=['object', 'string', 'category']).columns
        if len(str_cols) > 0:
            X[str_cols] = X[str_cols].astype(str).replace({'<NA>': np.nan, 'nan': np.nan})
        
        if pd.api.types.is_string_dtype(y) or pd.api.types.is_object_dtype(y):
            y = LabelEncoder().fit_transform(y)

        num_cols = X.select_dtypes(include=['int', 'float', 'number']).columns.tolist()
        cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

        # 4. Pipeline de Pré-processamento
        preprocessor = ColumnTransformer(transformers=[
            ('num', Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='mean')),
                ('scaler', StandardScaler())
            ]), num_cols),
            ('cat', Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
            ]), cat_cols)
        ])

        # 5. Subamostragem para eleição do melhor modelo
        TAMANHO_MAXIMO = 2000 
        if X.shape[0] > TAMANHO_MAXIMO:
            X_eval, y_eval = resample(X, y, n_samples=TAMANHO_MAXIMO, random_state=42, stratify=y)
        else:
            X_eval, y_eval = X, y

        # 6. Eleição do Meta-Target (Sem Data Leakage)
        best_acc, best_model = -1, None
        for nome_modelo, modelo in modelos_base.items():
            pipeline_avaliacao = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('classifier', modelo)
            ])
            acc = np.mean(cross_val_score(pipeline_avaliacao, X_eval, y_eval, cv=cv_base, scoring='accuracy', n_jobs=1)) # n_jobs=1 aqui para não brigar com as Threads
            if acc > best_acc:
                best_acc, best_model = acc, nome_modelo

        # 7. Extração de Meta-Features (Pymfe)
        X_mfe = preprocessor.fit_transform(X_eval) 
        mfe = MFE(
            groups=["general", "statistical", "info-theory", "landmarking", "model-based"],
            summary=["mean", "sd", "min", "max", "skewness", "kurtosis"]
        )
        mfe.fit(X_mfe, y_eval) 
        ft_names, ft_values = mfe.extract()

        # 8. Salvar no Log com Lock de segurança
        row = dict(zip(ft_names, ft_values))
        row['Dataset'] = dataset_name
        row['MetaTarget'] = best_model
        
        with lock_csv:
            arquivo_existe = os.path.exists(arquivo_log)
            df_row = pd.DataFrame([row])
            df_row.to_csv(arquivo_log, index=False, mode='a' if arquivo_existe else 'w', header=not arquivo_existe)

        print(f"✅ [Th-{t_id}] Sucesso: {dataset_name} | Vencedor: {best_model} | MFs: {len(ft_names)}")
        return True

    except Exception as e:
        print(f"❌ [Th-{t_id}] Erro em {dataset_name}: {str(e)}")
        return False

# In[ ]:


def gerar_meta_base_paralela(dataset_names, arquivo_cache="meta_base_cache.csv"):
    if os.path.exists(arquivo_cache):
        print(f"✅ Meta-base carregada do cache: {arquivo_cache}")
        return pd.read_csv(arquivo_cache)

    print(f"--- INICIANDO DOWNLOAD E EXTRAÇÃO PARALELA DE {len(dataset_names)} DATASETS ---")
    
    modelos_base = {
        'RF': RandomForestClassifier(n_estimators=50, random_state=42), 
        'SVM': SVC(random_state=42, max_iter=1000, cache_size=1000, tol=1e-2), 
        'kNN': KNeighborsClassifier(n_jobs=1),
        'LR': LogisticRegression(random_state=42, max_iter=500, solver='saga', tol=1e-2),
        'GB': GradientBoostingClassifier(n_estimators=50, random_state=42),
        'NB': GaussianNB(),
        'DT': DecisionTreeClassifier(random_state=42)
    }
    cv_base = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    arquivo_log = "checkpoint_meta_base.csv"

    # LIMITES DE CONCORRÊNCIA
    MAX_THREADS = 4
    TIMEOUT_SEGUNDOS = 180 # 3 Minutos por dataset no máximo

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
        futuros = {
            executor.submit(processar_um_dataset, nome, modelos_base, cv_base, arquivo_log): nome 
            for nome in dataset_names
        }
        
        for futuro in concurrent.futures.as_completed(futuros):
            nome = futuros[futuro]
            try:
                sucesso = futuro.result(timeout=TIMEOUT_SEGUNDOS)
            except concurrent.futures.TimeoutError:
                print(f"⚠️ [Timeout] {nome} demorou mais de {TIMEOUT_SEGUNDOS}s e foi cancelado.")
            except Exception as e:
                print(f"⚠️ [Falha] Erro crítico na execução do {nome}: {e}")

    # Montagem final do CSV após todas as threads terminarem
    if not os.path.exists(arquivo_log):
        return pd.DataFrame()
        
    df_meta = pd.read_csv(arquivo_log)
    df_meta.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    limite = int(0.8 * len(df_meta))
    df_meta = df_meta.dropna(axis=1, thresh=limite)
    
    num_cols = df_meta.select_dtypes(include=[np.number]).columns
    if len(num_cols) > 0:
        df_meta[num_cols] = SimpleImputer(strategy='median').fit_transform(df_meta[num_cols])

    df_meta.to_csv(arquivo_cache, index=False)
    print(f"✅ Meta-base final salva em {arquivo_cache}")
    return df_meta

# Defina seus datasets aqui
names = ['diabetes', 'blood-transfusion-service-center',
         'monks-problems-2', 'tic-tac-toe', 'titanic', 'pc1',
         'kr-vs-kp', 'phoneme', 'wdbc', 'semeion', 'isolet',
         'cnae-9', 'ilpd-numeric', 'students_scores',
         'usps', 'ibm-employee-performance','mushroom',
         'segment',  'autoUniv-au1-1000', 'pizzacutter3',
         'qsar', 'solar-flare']
blacklist = {
    'sick-numeric', 'telco-custumer-churn',
    'credit-g', 'anneal', 'kits'
}

MIN_EXTRA = 60
MAX_AREA = 300000
MAX_ROWS = 10000
MAX_COLS = 50

print("🔍 Buscando datasets candidatos no OpenML...")
try:
    # 1. Busca rápida no servidor (Evita o travamento de 10 minutos)
    df = openml.datasets.list_datasets(
        number_instances=f"200..{MAX_ROWS}",
        number_features=f"5..{MAX_COLS}",
        output_format="dataframe"
    )
except Exception as e:
    print(f"⚠️ O servidor do OpenML falhou: {e}")
    # Fallback seguro para não estourar o código
    df = pd.DataFrame() 

if not df.empty:
    # 2. A sua lógica de filtragem local rigorosa
    filtered = df[
        (df["NumberOfClasses"] >= 2) &
        (df["NumberOfMissingValues"] < 5000)
    ].copy()

    filtered["area"] = filtered["NumberOfInstances"] * filtered["NumberOfFeatures"]
    filtered = filtered[filtered["area"] <= MAX_AREA]
    filtered = filtered.sort_values(by=["area", "NumberOfMissingValues"])

    # 3. Pega nomes extras automaticamente
    candidate_names = filtered["name"].dropna().unique().tolist()

    # 4. Remove os já existentes + blacklist
    extra_names = [
        name for name in candidate_names
        if name not in names and name not in blacklist
    ]

    # Pega os 60 melhores
    extra_names = extra_names[:MIN_EXTRA]

    # Junta tudo
    nomes_finais = names + extra_names

    print(f"Datasets originais: {len(names)}")
    print(f"Extras adicionados: {len(extra_names)}")
    print(f"Total: {len(nomes_finais)}")

    # 5. Roda a extração paralela em cima da lista final
    meta_base = gerar_meta_base_paralela(nomes_finais)

    if meta_base.empty:
        print("❌ Operação abortada: A meta-base foi gerada vazia. Verifique sua conexão com o OpenML.")
        sys.exit(1) # <--- ADICIONE ISTO AQUI


else:
    print("❌ Não foi possível carregar a lista base do OpenML. Tente novamente mais tarde.")
    sys.exit(1)


# In[ ]:


class MetaFeaturePruner:
    def v1_brute_force(self, X):
        return np.arange(X.shape[1]), 0.0

    def v2_hard_pruning(self, X, y):
        start = time.time()
        rf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
        rf.fit(X, y)
        idx = np.where(rf.feature_importances_ > np.mean(rf.feature_importances_))[0]
        return idx, time.time() - start

    def v3_erf(self, X, y):
        start = time.time()
        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X, y)
        idx = np.where(rf.feature_importances_ >= (np.mean(rf.feature_importances_) * 0.5))[0]
        return idx, time.time() - start

class GeneticFeatureSelector:
    def __init__(self, n_generations=50, crossover_prob=0.8, mutation_prob=0.05, random_state=42):
        self.n_generations = n_generations
        self.crossover_prob = crossover_prob
        self.base_mutation_prob = mutation_prob
        self.random_state = random_state
        
        self.evaluator = RandomForestClassifier(n_estimators=20, random_state=self.random_state, n_jobs=-1)
        self.cv_strategy = KFold(n_splits=3, shuffle=True, random_state=self.random_state)
        self.history = []
        self.fitness_cache = {}

    def _calculate_fitness(self, X, y, chromosome):
        chrom_key = chromosome.tobytes()
        if chrom_key in self.fitness_cache:
            return self.fitness_cache[chrom_key]
            
        n_selected = np.sum(chromosome)
        if n_selected == 0: 
            return 1e-4, n_selected
            
        X_subset = X[:, chromosome == 1]
        acc = np.mean(cross_val_score(self.evaluator, X_subset, y, cv=self.cv_strategy, scoring='balanced_accuracy'))
        
        sparsity_bonus = 1.0 - (n_selected / X.shape[1])
        fitness = max(1e-4, acc + (0.001 * sparsity_bonus))
        
        # Salva no cache antes de retornar
        self.fitness_cache[chrom_key] = (fitness, n_selected)
        return fitness, n_selected

    def _tournament_selection(self, population, fitness_scores, k=3):
        pop_size = len(population)
        
        # Sorteia uma matriz de índices (pop_size, k) com os lutadores
        torneios = np.random.randint(0, pop_size, size=(pop_size, k))
        
        # Resgata o fitness de cada lutador na matriz
        scores_torneios = fitness_scores[torneios]
        
        # Encontra quem ganhou cada torneio (índice da coluna com maior score)
        vencedores_idx_relativo = np.argmax(scores_torneios, axis=1)
        
        # Mapeia de volta para o índice real da população
        vencedores_idx_real = torneios[np.arange(pop_size), vencedores_idx_relativo]
        
        # Retorna os pais selecionados em uma única operação de slice
        return population[vencedores_idx_real]

    def fit_transform(self, X, y):
        start_time = time.time()
        np.random.seed(self.random_state)
        n_features = X.shape[1]
        
        pop_size = max(20, min(100, n_features // 2))
        prob_1 = min(0.5, 175 / n_features) if n_features > 0 else 0.5
        population = np.random.choice([0, 1], size=(pop_size, n_features), p=[1-prob_1, prob_1])

        best_global_chromosome = None
        best_global_fitness = -1
        best_global_count = 0

        print(f"        [GA] Iniciando evolução... População: {pop_size} | Gerações: {self.n_generations}")

        for generation in range(self.n_generations):
            fitness_results = [self._calculate_fitness(X, y, ind) for ind in population]
            fitness_scores = np.array([f[0] for f in fitness_results])
            counts = np.array([f[1] for f in fitness_results])
            
            best_idx = np.argmax(fitness_scores)
            
            if fitness_scores[best_idx] > best_global_fitness:
                best_global_fitness = fitness_scores[best_idx]
                best_global_count = counts[best_idx]
                best_global_chromosome = population[best_idx].copy()

            self.history.append({
                'generation': generation,
                'best_fitness': best_global_fitness,
                'num_features': best_global_count
            })

            # -----------------------------------------------------------------
            # CRITÉRIO DE PARADA: COLAPSO DE DIVERSIDADE GENÉTICA
            # -----------------------------------------------------------------
            # Calcula a variância de cada gene (feature) na população e tira a média.
            # Se for muito baixa, todos os indivíduos são praticamente clones.
            diversidade = np.mean(np.var(population, axis=0))
            if generation > 20 and diversidade < 0.01: 
                print(f"        [GA] Parada antecipada: Convergência genética atingida na Geração {generation:03d} (Diversidade: {diversidade:.4f}).")
                break

            if generation > 0 and generation % 10 == 0:
                current_mutation = min(0.3, self.base_mutation_prob * 4)
                print(f"        [GA] Gen {generation:03d}/{self.n_generations} | Melhor: {best_global_fitness:.4f} | Ativas: {best_global_count:03d} | Div: {diversidade:.3f} ⚠️ CAOS")
            else:
                current_mutation = self.base_mutation_prob
                if generation % 10 == 0 or generation == self.n_generations - 1:
                    print(f"        [GA] Gen {generation:03d}/{self.n_generations} | Melhor: {best_global_fitness:.4f} | Ativas: {best_global_count:03d} | Div: {diversidade:.3f}")

            parents = self._tournament_selection(population, fitness_scores, k=3)
            
            # -----------------------------------------------------------------
            # ELITISMO CORRIGIDO: O melhor indivíduo pula a mutação/cruzamento
            # -----------------------------------------------------------------
            next_gen = [best_global_chromosome.copy()]
            
            # O loop agora preenche apenas o RESTANTE da população
            for i in range(0, pop_size - 1, 2):
                p1 = parents[i]
                p2 = parents[(i+1) % len(parents)] # Evita index out of bounds
                
                if np.random.rand() < self.crossover_prob:
                    mask = np.random.rand(n_features) > 0.5
                    c1 = np.where(mask, p1, p2)
                    c2 = np.where(mask, p2, p1)
                else:
                    c1, c2 = p1.copy(), p2.copy()
                
                # Mutação vetorizada (já estava boa, mantida)
                c1 ^= (np.random.rand(n_features) < current_mutation)
                c2 ^= (np.random.rand(n_features) < current_mutation)
                
                next_gen.extend([c1, c2])

            # Garante o tamanho exato cortando possíveis excessos do loop par
            population = np.array(next_gen)[:pop_size]

        final_indices = np.where(best_global_chromosome == 1)[0]
        return final_indices, time.time() - start_time, self.history


# In[ ]:


def executar_pipeline_wrapper(df_meta, arq_resultados="resultados_tabela.csv", arq_features="meta_features_v4.csv"):
    
    X_meta = df_meta.drop(columns=['Dataset', 'MetaTarget']).values
    y_meta = LabelEncoder().fit_transform(df_meta['MetaTarget'])
    meta_cols = df_meta.drop(columns=['Dataset', 'MetaTarget']).columns.tolist()
    
    cv_eval = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    final_clf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
    
    metricas = ['balanced_accuracy', 'f1_macro']
    resultados = []
    pruner = MetaFeaturePruner()
    
    # --- V1 ---
    print("-> 1/4. Analisando V1 (Nenhuma Poda / Força Bruta)...")
    idx_v1, t_v1 = pruner.v1_brute_force(X_meta)
    scores_v1 = cross_validate(final_clf, X_meta[:, idx_v1], y_meta, cv=cv_eval, scoring=metricas)
    acc_v1, f1_v1 = np.mean(scores_v1['test_balanced_accuracy']), np.mean(scores_v1['test_f1_macro'])
    print(f"        Meta-features retidas: {len(idx_v1)}\n")
    
    # --- V2 ---
    print("-> 2/4. Aplicando V2 (Hard Pruning Padrão)...")
    idx_v2, t_v2 = pruner.v2_hard_pruning(X_meta, y_meta)
    scores_v2 = cross_validate(final_clf, X_meta[:, idx_v2], y_meta, cv=cv_eval, scoring=metricas)
    acc_v2, f1_v2 = np.mean(scores_v2['test_balanced_accuracy']), np.mean(scores_v2['test_f1_macro'])
    print(f"        Meta-features retidas: {len(idx_v2)}\n")
    
    # --- V3 ---
    print("-> 3/4. Aplicando V3 (Enriched Random Forest)...")
    idx_v3, t_v3 = pruner.v3_erf(X_meta, y_meta)
    scores_v3 = cross_validate(final_clf, X_meta[:, idx_v3], y_meta, cv=cv_eval, scoring=metricas)
    acc_v3, f1_v3 = np.mean(scores_v3['test_balanced_accuracy']), np.mean(scores_v3['test_f1_macro'])
    print(f"        Meta-features retidas: {len(idx_v3)}\n")
    
    # --- V4 ---
    print("-> 4/4. Aplicando V4 (Otimização via Algoritmo Genético)...")
    X_v3 = X_meta[:, idx_v3] 
    ga = GeneticFeatureSelector(n_generations=300, random_state=42)
    idx_ga_relativo, t_ga, ga_history = ga.fit_transform(X_v3, y_meta)
    
    idx_v4_absoluto = idx_v3[idx_ga_relativo]
    scores_v4 = cross_validate(final_clf, X_meta[:, idx_v4_absoluto], y_meta, cv=cv_eval, scoring=metricas)
    acc_v4, f1_v4 = np.mean(scores_v4['test_balanced_accuracy']), np.mean(scores_v4['test_f1_macro'])
    print(f"        Meta-features retidas: {len(idx_v4_absoluto)}\n")
    
    # --- SALVAR O NOME DAS FEATURES VENCEDORAS (V4) EM CSV ---
    features_v4_nomes = [meta_cols[i] for i in idx_v4_absoluto]
    pd.DataFrame({"Meta_Features_Sobreviventes_V4": features_v4_nomes}).to_csv(arq_features, index=False)
    print(f"✅ Nomes das Meta-Features vencedoras salvos em: '{arq_features}'")
    
    # --- MONTAR RELATÓRIO ---
    modelos_info = [
        ('V1 (Força Bruta)', len(idx_v1), t_v1, t_v1, acc_v1, f1_v1),
        ('V2 (Hard Pruning)', len(idx_v2), t_v2, t_v2, acc_v2, f1_v2),
        ('V3 (ERF)', len(idx_v3), t_v3, t_v3, acc_v3, f1_v3),
        ('V4 (ERF + GA)', len(idx_v4_absoluto), t_v3, t_v3 + t_ga, acc_v4, f1_v4)
    ]
    
    for nome, qtd, tp, tt, acc, f1 in modelos_info:
        resultados.append({
            'Modelo': nome, 
            'Qtd_MFs_Sobreviventes': qtd,
            'Tempo_Pruning(s)': round(tp, 4), 
            'Tempo_Total(s)': round(tt, 4),
            'Acc_Balanceada': round(acc, 4),
            'F1_Macro': round(f1, 4)
        })
        
    df_resultados = pd.DataFrame(resultados)
    df_resultados.to_csv(arq_resultados, index=False)
    print(f"✅ Tabela de comparação salva em: '{arq_resultados}'\n")
    
    # Plot GA history
    history_df = pd.DataFrame(ga_history)
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(history_df['generation'], history_df['best_fitness'], 'b-')
    ax1.set_xlabel('Geração')
    ax1.set_ylabel('Fitness (Balanced Accuracy + Sparsity)', color='b')
    ax1.tick_params('y', colors='b')
    
    ax2 = ax1.twinx()
    ax2.plot(history_df['generation'], history_df['num_features'], 'r--')
    ax2.set_ylabel('Número de Features', color='r')
    ax2.tick_params('y', colors='r')
    
    plt.title('Evolução do Algoritmo Genético')
    plt.savefig('grafico_evolucao_ga.png', dpi=300, bbox_inches='tight') # SALVA O GRÁFICO 1
    plt.show()
    
    return df_resultados

# Executa e faz o display visual
df_final = executar_pipeline_wrapper(meta_base)
print(df_final.to_string())


# ## Análise das Meta-Features Vencedoras
# Vamos verificar quais features o GA achou essenciais e tentar interpretar o porquê de elas terem sido selecionadas. A análise avalia a frequência com que determinados grupos (ex: estatísticas, landmarks, info-theory) sobreviveram.

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

features_v4 = pd.read_csv('meta_features_v4.csv')
print("Total de features selecionadas:", len(features_v4))

def extrair_grupo(nome):
    if 'mean' in nome or 'sd' in nome or 'min' in nome or 'max' in nome:
        return 'Statistical / Summary'
    elif 'attrEnt' in nome or 'mutInf' in nome or 'classEnt' in nome or 'jointEnt' in nome:
        return 'Info-Theory'
    elif 'bestNode' in nome or 'eliteNN' in nome or 'linearDiscr' in nome or 'naiveBayes' in nome or 'oneNN' in nome or 'randomNode' in nome or 'worstNode' in nome:
        return 'Landmarking'
    elif 'leaves' in nome or 'tree' in nome or 'nodes' in nome or 'varImportance' in nome:
        return 'Model-Based'
    else:
        return 'General / Other'

features_v4['Grupo'] = features_v4['Meta_Features_Sobreviventes_V4'].apply(extrair_grupo)

plt.figure(figsize=(10, 6))
sns.countplot(data=features_v4, y='Grupo', order=features_v4['Grupo'].value_counts().index, palette='viridis')
plt.title('Distribuição das Famílias de Meta-features Selecionadas (GA)')
plt.xlabel('Quantidade')
plt.ylabel('Família da Meta-Feature')
plt.savefig('grafico_familias_mf.png', dpi=300, bbox_inches='tight') # SALVA O GRÁFICO 2
plt.show()

print("As features selecionadas são as que mais impactam na detecção do melhor algoritmo.")
print("Features baseadas em modelo (como tamanho de árvore) e Landmarking normalmente sobrevivem mais porque")
print("carregam em si uma visão muito rica de quão linear ou complexa é a fronteira de decisão do dataset.")

# =====================================================================
# ANÁLISE: RANDOM FOREST IMPORTANCE VS GA SELECTION
# =====================================================================
print("\n--- Analisando Importância RF vs Seleção GA ---")

# Preparar dados novamente para a análise final
X_meta_final = meta_base.drop(columns=['Dataset', 'MetaTarget']).values
y_meta_final = LabelEncoder().fit_transform(meta_base['MetaTarget'])
meta_cols_final = meta_base.drop(columns=['Dataset', 'MetaTarget']).columns.tolist()

# 1. Treinar RF com todas as features originais (V1)
rf_analise = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_analise.fit(X_meta_final, y_meta_final)

# 2. Criar DataFrame com as importâncias
df_importances = pd.DataFrame({
    'Meta_Feature': meta_cols_final,
    'Importancia_RF': rf_analise.feature_importances_
})

# 3. Marcar quais sobreviveram ao GA (V4)
features_v4_list = features_v4['Meta_Features_Sobreviventes_V4'].tolist()
df_importances['Mantida_pelo_GA'] = df_importances['Meta_Feature'].isin(features_v4_list)

# 4. Pegar as Top 30 mais importantes segundo a Random Forest original
top_30_rf = df_importances.sort_values(by='Importancia_RF', ascending=False).head(30)

# 5. Plotar e Salvar
plt.figure(figsize=(12, 8))
sns.barplot(
    data=top_30_rf, 
    x='Importancia_RF', 
    y='Meta_Feature', 
    hue='Mantida_pelo_GA', 
    dodge=False,
    palette={True: '#2ecc71', False: '#e74c3c'} # Verde: GA manteve, Vermelho: GA cortou
)
plt.title('Top 30 Meta-Features (Importância RF) vs. Sobrevivência no GA')
plt.xlabel('Importância (Gini) na Random Forest')
plt.ylabel('Meta-Feature')
plt.legend(title='Mantida pelo GA (V4)?', loc='lower right')
plt.tight_layout()
plt.savefig('grafico_rf_vs_ga.png', dpi=300, bbox_inches='tight') # SALVA O GRÁFICO 3
plt.show()

print("\nPipeline 100% finalizado! Verifique a sua pasta raiz para encontrar os 3 arquivos .png salvos.")