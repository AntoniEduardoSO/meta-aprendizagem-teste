import json

with open('projeto.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        
        if "from sklearn.ensemble import RandomForestClassifier" in source and "SVC" in source:
            source = source.replace("from sklearn.ensemble import RandomForestClassifier", "from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier\nfrom sklearn.linear_model import LogisticRegression\nfrom sklearn.naive_bayes import GaussianNB")
            cell['source'] = [line + "\n" if not line.endswith("\n") else line for line in source.split("\n")][:-1]
            if len(cell['source']) > 0 and cell['source'][-1].endswith('\n') and not source.endswith('\n'):
                cell['source'][-1] = cell['source'][-1][:-1]

        if "modelos_base = {" in source and "'RF': RandomForestClassifier" in source:
            new_modelos_base = """    modelos_base = {
        'RF': RandomForestClassifier(random_state=42),
        'SVM': SVC(random_state=42, probability=True),
        'kNN': KNeighborsClassifier(),
        'LR': LogisticRegression(random_state=42, max_iter=1000),
        'GB': GradientBoostingClassifier(random_state=42),
        'NB': GaussianNB(),
        'DT': DecisionTreeClassifier(random_state=42)
    }"""
            import re
            source = re.sub(r"    modelos_base = \{.*?'kNN': KNeighborsClassifier\(\)\n    \}", new_modelos_base, source, flags=re.DOTALL)
            
            # Update datasets
            source = re.sub(r"ids_teste = \[.*?\]", "ids_teste = [3, 11, 15, 29, 31]", source)
            
            cell['source'] = [line + "\n" for line in source.split('\n')]
            cell['source'][-1] = cell['source'][-1][:-1] # remove last newline

        if "class GeneticFeatureSelector:" in source:
            new_ga_class = """class GeneticFeatureSelector:
    def __init__(self, n_generations=50, crossover_prob=0.8, mutation_prob=0.05, random_state=42):
        self.n_generations = n_generations
        self.crossover_prob = crossover_prob
        self.base_mutation_prob = mutation_prob
        self.random_state = random_state
        
        self.evaluator = RandomForestClassifier(n_estimators=20, random_state=self.random_state, n_jobs=-1)
        self.cv_strategy = StratifiedKFold(n_splits=3, shuffle=True, random_state=self.random_state)
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
        return final_indices, time.time() - start_time, self.history"""
            
            import re
            source = re.sub(r"class GeneticFeatureSelector:.*?return final_indices, time\.time\(\) - start_time", new_ga_class, source, flags=re.DOTALL)
            cell['source'] = [line + "\n" for line in source.split('\n')]
            cell['source'][-1] = cell['source'][-1][:-1]

        if "def executar_pipeline_wrapper" in source:
            source = source.replace("idx_ga_relativo, t_ga = ga.fit_transform(X_v3, y_meta)", "idx_ga_relativo, t_ga, ga_history = ga.fit_transform(X_v3, y_meta)")
            source = source.replace("ga = GeneticFeatureSelector(target_min=150, target_max=200, random_state=42)", "ga = GeneticFeatureSelector(n_generations=50, random_state=42)")
            
            # Plotting logic for GA history
            plot_logic = """
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
    
    return df_resultados"""
            source = source.replace("return df_resultados", plot_logic)
            cell['source'] = [line + "\n" for line in source.split('\n')]
            cell['source'][-1] = cell['source'][-1][:-1]

with open('projeto.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Notebook updated successfully.")
