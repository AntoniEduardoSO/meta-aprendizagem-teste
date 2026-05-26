# %pip install pymfe openml matplotlib seaborn

import os
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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


def gerar_meta_base(dataset_ids, arquivo_cache="meta_base_cache.csv"):
    if os.path.exists(arquivo_cache):
        print(f"✅ Meta-base carregada do cache: {arquivo_cache}")
        return pd.read_csv(arquivo_cache)

    print("--- INICIANDO DOWNLOAD E EXTRAÇÃO DE META-FEATURES ---")
    meta_features_list = []
    
    modelos_base = {
        'RF': RandomForestClassifier(random_state=42),
        'SVM': SVC(random_state=42, probability=True),
        'kNN': KNeighborsClassifier(),
        'LR': LogisticRegression(random_state=42, max_iter=1000),
        'GB': GradientBoostingClassifier(random_state=42),
        'NB': GaussianNB(),
        'DT': DecisionTreeClassifier(random_state=42)
    }
    cv_base = KFold(n_splits=3, shuffle=True, random_state=42)

    for data_id in dataset_ids:
        print(f"Processando OpenML ID {data_id}...", end=" ")
        try:
            dataset = fetch_openml(data_id=data_id, as_frame=True, parser='auto')
            X, y = dataset.data, dataset.target
            dataset_name = dataset.details.get('name', f'ID_{data_id}')

            num_cols = X.select_dtypes(include=['int', 'float', 'number']).columns
            cat_cols = X.select_dtypes(include=['object', 'category']).columns

            preprocessor = ColumnTransformer(transformers=[
                ('num', SimpleImputer(strategy='mean'), num_cols),
                ('cat', Pipeline(steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
                ]), cat_cols)
            ])
            X_clean = preprocessor.fit_transform(X)
            y_clean = LabelEncoder().fit_transform(y) if y.dtype in ['object', 'category'] else np.array(y)

            best_acc, best_model = -1, None
            for nome_modelo, modelo in modelos_base.items():
                acc = np.mean(cross_val_score(modelo, X_clean, y_clean, cv=cv_base, scoring='accuracy', n_jobs=-1))
                if acc > best_acc:
                    best_acc, best_model = acc, nome_modelo

            # Extração massiva
            mfe = MFE(
                groups=["general", "statistical", "info-theory", "landmarking", "model-based"],
                summary=["mean", "sd", "min", "max", "skewness", "kurtosis"]
            )
            mfe.fit(X_clean, y_clean)
            ft_names, ft_values = mfe.extract()

            row = dict(zip(ft_names, ft_values))
            row['Dataset'] = dataset_name
            row['MetaTarget'] = best_model
            meta_features_list.append(row)
            print(f"OK! Vencedor: {best_model}. MFs extraídas: {len(ft_names)}")

        except Exception as e:
            print(f"ERRO: {e}")

    df_meta = pd.DataFrame(meta_features_list)
    if df_meta.empty: return df_meta
        
    # Limpeza e tratamento de dados
    colunas_mf = df_meta.columns.drop(['Dataset', 'MetaTarget'], errors='ignore')
    df_meta[colunas_mf] = df_meta[colunas_mf].apply(pd.to_numeric, errors='coerce')
    df_meta.replace([np.inf, -np.inf], np.nan, inplace=True)

    limite = int(0.8 * len(df_meta))
    df_meta = df_meta.dropna(axis=1, thresh=limite)
    
    num_cols = df_meta.select_dtypes(include=[np.number]).columns
    if len(num_cols) > 0:
        df_meta[num_cols] = SimpleImputer(strategy='median').fit_transform(df_meta[num_cols])

    df_meta.to_csv(arquivo_cache, index=False)
    print(f"✅ Meta-base salva em {arquivo_cache}")
    return df_meta

# Defina seus datasets aqui
ids_teste = [3, 11, 15, 29, 31] 
meta_base = gerar_meta_base(ids_teste)

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

    def _calculate_fitness(self, X, y, chromosome):
        n_selected = np.sum(chromosome)
        if n_selected == 0: return 1e-4, n_selected
            
        X_subset = X[:, chromosome == 1]
        acc = np.mean(cross_val_score(self.evaluator, X_subset, y, cv=self.cv_strategy, scoring='balanced_accuracy'))
        
        # Penalidade leve para incentivar menos features, mas sem hard threshold
        sparsity_bonus = 1.0 - (n_selected / X.shape[1])
        fitness = acc + (0.001 * sparsity_bonus)
            
        return max(1e-4, fitness), n_selected

    def _tournament_selection(self, population, fitness_scores, k=3):
        parents = []
        for _ in range(len(population)):
            fighters = np.random.choice(len(population), size=k, replace=False)
            winner = fighters[np.argmax(fitness_scores[fighters])]
            parents.append(population[winner])
        return np.array(parents)

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

            if generation > 0 and generation % 10 == 0:
                current_mutation = min(0.3, self.base_mutation_prob * 4)
                print(f"        [GA] Gen {generation:03d}/{self.n_generations} | Melhor: {best_global_fitness:.4f} | Ativas: {best_global_count:03d} ⚠️ CAOS (Mutação {int(current_mutation*100)}%)")
            else:
                current_mutation = self.base_mutation_prob
                if generation % 10 == 0 or generation == self.n_generations - 1:
                    print(f"        [GA] Gen {generation:03d}/{self.n_generations} | Melhor: {best_global_fitness:.4f} | Ativas: {best_global_count:03d}")

            parents = self._tournament_selection(population, fitness_scores, k=3)
            next_gen = []
            
            for i in range(0, pop_size, 2):
                p1 = parents[i]
                p2 = parents[(i+1) % pop_size]
                
                if np.random.rand() < self.crossover_prob:
                    num_cortes = np.random.randint(1, 6)
                    max_cortes = min(5, max(1, n_features - 2))
                    if max_cortes > 1:
                        num_cortes = np.random.randint(1, max_cortes)
                        pontos_corte = sorted(np.random.choice(range(1, n_features - 1), num_cortes, replace=False))
                    else:
                        pontos_corte = [n_features // 2] if n_features > 1 else []
                    
                    c1, c2 = [], []
                    inverter = False
                    ultimo_ponto = 0
                    
                    for ponto in pontos_corte + [n_features]:
                        if not inverter:
                            c1.extend(p1[ultimo_ponto:ponto])
                            c2.extend(p2[ultimo_ponto:ponto])
                        else:
                            c1.extend(p2[ultimo_ponto:ponto])
                            c2.extend(p1[ultimo_ponto:ponto])
                        inverter = not inverter
                        ultimo_ponto = ponto
                        
                    c1, c2 = np.array(c1), np.array(c2)
                else:
                    c1, c2 = p1.copy(), p2.copy()
                
                mask1 = np.random.rand(n_features) < current_mutation
                c1[mask1] = 1 - c1[mask1]
                mask2 = np.random.rand(n_features) < current_mutation
                c2[mask2] = 1 - c2[mask2]
                
                next_gen.extend([c1, c2])

            population = np.array(next_gen)[:pop_size]
            population[0] = best_global_chromosome 

        final_indices = np.where(best_global_chromosome == 1)[0]
        return final_indices, time.time() - start_time, self.history

def executar_pipeline_wrapper(df_meta, arq_resultados="resultados_tabela.csv", arq_features="meta_features_v4.csv"):
    print("\n=======================================================")
    print(" INICIANDO PIPELINE DE SELEÇÃO E REDUÇÃO")
    print("=======================================================\n")
    
    X_meta = df_meta.drop(columns=['Dataset', 'MetaTarget']).values
    y_meta = LabelEncoder().fit_transform(df_meta['MetaTarget'])
    meta_cols = df_meta.drop(columns=['Dataset', 'MetaTarget']).columns.tolist()
    
    cv_eval = KFold(n_splits=3, shuffle=True, random_state=42)
    final_clf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
    
    resultados = []
    pruner = MetaFeaturePruner()
    
    # --- V1 ---
    print("-> 1/4. Analisando V1 (Nenhuma Poda / Força Bruta)...")
    idx_v1, t_v1 = pruner.v1_brute_force(X_meta)
    acc_v1 = np.mean(cross_val_score(final_clf, X_meta[:, idx_v1], y_meta, cv=cv_eval, scoring='balanced_accuracy'))
    print(f"        Meta-features retidas: {len(idx_v1)}\n")
    
    # --- V2 ---
    print("-> 2/4. Aplicando V2 (Hard Pruning Padrão)...")
    idx_v2, t_v2 = pruner.v2_hard_pruning(X_meta, y_meta)
    acc_v2 = np.mean(cross_val_score(final_clf, X_meta[:, idx_v2], y_meta, cv=cv_eval, scoring='balanced_accuracy'))
    print(f"        Meta-features retidas: {len(idx_v2)}\n")
    
    # --- V3 ---
    print("-> 3/4. Aplicando V3 (Enriched Random Forest)...")
    idx_v3, t_v3 = pruner.v3_erf(X_meta, y_meta)
    acc_v3 = np.mean(cross_val_score(final_clf, X_meta[:, idx_v3], y_meta, cv=cv_eval, scoring='balanced_accuracy'))
    print(f"        Meta-features retidas: {len(idx_v3)}\n")
    
    # --- V4 ---
    print("-> 4/4. Aplicando V4 (Otimização via Algoritmo Genético)...")
    X_v3 = X_meta[:, idx_v3] 
    ga = GeneticFeatureSelector(n_generations=50, random_state=42)
    idx_ga_relativo, t_ga, ga_history = ga.fit_transform(X_v3, y_meta)
    
    idx_v4_absoluto = idx_v3[idx_ga_relativo]
    acc_v4 = np.mean(cross_val_score(final_clf, X_meta[:, idx_v4_absoluto], y_meta, cv=cv_eval, scoring='balanced_accuracy'))
    print(f"        Meta-features retidas: {len(idx_v4_absoluto)}\n")
    
    # --- SALVAR O NOME DAS FEATURES VENCEDORAS (V4) EM CSV ---
    features_v4_nomes = [meta_cols[i] for i in idx_v4_absoluto]
    pd.DataFrame({"Meta_Features_Sobreviventes_V4": features_v4_nomes}).to_csv(arq_features, index=False)
    print(f"✅ Nomes das Meta-Features vencedoras salvos em: '{arq_features}'")
    
    # --- MONTAR RELATÓRIO ---
    modelos_info = [
        ('V1 (Força Bruta)', len(idx_v1), t_v1, t_v1, acc_v1),
        ('V2 (Hard Pruning)', len(idx_v2), t_v2, t_v2, acc_v2),
        ('V3 (ERF)', len(idx_v3), t_v3, t_v3, acc_v3),
        ('V4 (ERF + GA)', len(idx_v4_absoluto), t_v3, t_v3 + t_ga, acc_v4)
    ]
    
    for nome, qtd, tp, tt, acc in modelos_info:
        resultados.append({
            'Modelo': nome, 
            'Qtd_MFs_Sobreviventes': qtd,
            'Tempo_Pruning(s)': round(tp, 4), 
            'Tempo_Total(s)': round(tt, 4),
            'Acc_Balanceada': round(acc, 4)
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
    plt.show()
    
    return df_resultados

# Executa e faz o display visual
df_final = executar_pipeline_wrapper(meta_base)
display(df_final)

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
plt.show()

print("As features selecionadas são as que mais impactam na detecção do melhor algoritmo.")
print("Features baseadas em modelo (como tamanho de árvore) e Landmarking normalmente sobrevivem mais porque")
print("carregam em si uma visão muito rica de quão linear ou complexa é a fronteira de decisão do dataset.")


