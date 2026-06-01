# Histórico de Mudanças - Projeto Meta-Aprendizagem

## [2026-05-25] Melhorias no Pipeline e Base de Modelos
- **Expansão de Algoritmos Base**: O pool de meta-targets foi expandido. Além de RF, SVM e kNN, agora conta também com Regressão Logística, Gradient Boosting, Naive Bayes e Decision Tree, aproximando a meta-base a cenários de Machine Learning do mundo real.
- **Redução nos Datasets do Experimento**: Diminuição da carga inicial de datasets de teste para permitir ciclos de experimentação mais rápidos durante o desenvolvimento e ajuste do GA.
- **Evolução do Algoritmo Genético (Remoção da Punição Rígida)**: O GA não pune mais estritamente o tamanho usando `target_min` e `target_max` que estavam espremendo a seleção. No lugar, utiliza apenas um bônus numérico proporcional à esparsidade (quantidade de features removidas), permitindo reduções drásticas, maximizando a acurácia.
- **Critério de Parada e Plotagem**: Substituído o gatilho de "paciência" por um limite de 50 gerações preestabelecidas, capturando a capacidade de fuga de ótimos locais a longo prazo através de mutações caóticas. Adicionado gráfico de linha para acompanhamento visual do fitness e redução de dimensionalidade por geração.
- **Análise Final das Features Vencedoras**: Adicionada célula final de análise com gráfico de barras para exibir as famílias teóricas de meta-features que conseguiram sobreviver após a filtragem pelo GA (Landmarking, Model-based, etc.).

## [2024-05-25] Correção de Erros e Reversão de Domínio (GA)

- **Correção Crítica (NameError)**: Inclusão da classe `MetaFeaturePruner` que não havia sido salva na célula e impedia a execução do pipeline.
- **Domínio do Cromossomo**: Garantido o uso do padrão onde `1 = feature presente` e `0 = feature ausente`.
- **Prevenção de Erro no Crossover**: Adicionada proteção matemática (`max_cortes`) para impedir a quebra do código (`ValueError`) caso o número de features disponíveis seja menor que o número de pontos de corte estipulados no crossover dinâmico.

## [2024-05-22] Refatoração e Melhoria do Algoritmo Genético (GA)

- **Nova Lógica de Domínio**: Alterado o GA para que `0` represente feature presente e `1` represente feature ausente, conforme requisito de domínio específico.
- **Crossover Dinâmico**: Implementado crossover de múltiplos pontos aleatórios para criar "filetes" genéticos, permitindo que blocos dinâmicos de genes sejam herdados.
- **Classe MetaFeaturePruner**: Adicionada a classe ausente com três níveis de poda:
    - V1: Força bruta (sem poda).
    - V2: Hard Pruning (baseado na média de importância do Random Forest).
    - V3: Enriched RF (poda mais permissiva para alimentar o GA).
- **Refatoração do Notebook**: Células consolidadas para maior concisão e legibilidade.
- **Tracker**: Criação deste arquivo para controle de versões e mudanças.

## Atualizações Recentes (Algoritmo Genético e Correção de Erros)
- **Correção do GA (Fitness):** A métrica de *fitness* passou a usar alanced_accuracy em vez de ccuracy, resolvendo a discrepância nos resultados onde o GA convergia mas entregava modelos desbalanceados.
- **Crossover Dinâmico:** Implementado cruzamento dinâmico no Algoritmo Genético (selecionando $ cortes de tamanhos diferentes, gerando partições complementares).
- **Aumento de Testes:** Ampliada a lista de datasets de avaliação de 20 para 40 datasets (OpenML).
- **Gerações do GA:** Ampliadas de 50 para 150 para permitir maior tempo de exploração no pool de features.
- **Correção de Erro de Tipo (Pandas):** Adicionada conversão explícita de colunas com StringDtype para object, prevenindo falhas de interpretação do scikit-learn ao iterar pipelines.
- **Tratamento de Warnings:** Reforço no silenciamento de FutureWarning e UserWarning emitidos por PyMFE e Scikit-Learn para manter a limpeza do terminal.
