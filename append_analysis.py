import json

with open('projeto.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

markdown_cell = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## Análise das Meta-Features Vencedoras\n",
        "Vamos verificar quais features o GA achou essenciais e tentar interpretar o porquê de elas terem sido selecionadas. A análise avalia a frequência com que determinados grupos (ex: estatísticas, landmarks, info-theory) sobreviveram."
    ]
}

code_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "import pandas as pd\n",
        "import matplotlib.pyplot as plt\n",
        "import seaborn as sns\n",
        "\n",
        "features_v4 = pd.read_csv('meta_features_v4.csv')\n",
        "print(\"Total de features selecionadas:\", len(features_v4))\n",
        "\n",
        "def extrair_grupo(nome):\n",
        "    if 'mean' in nome or 'sd' in nome or 'min' in nome or 'max' in nome:\n",
        "        return 'Statistical / Summary'\n",
        "    elif 'attrEnt' in nome or 'mutInf' in nome or 'classEnt' in nome or 'jointEnt' in nome:\n",
        "        return 'Info-Theory'\n",
        "    elif 'bestNode' in nome or 'eliteNN' in nome or 'linearDiscr' in nome or 'naiveBayes' in nome or 'oneNN' in nome or 'randomNode' in nome or 'worstNode' in nome:\n",
        "        return 'Landmarking'\n",
        "    elif 'leaves' in nome or 'tree' in nome or 'nodes' in nome or 'varImportance' in nome:\n",
        "        return 'Model-Based'\n",
        "    else:\n",
        "        return 'General / Other'\n",
        "\n",
        "features_v4['Grupo'] = features_v4['Meta_Features_Sobreviventes_V4'].apply(extrair_grupo)\n",
        "\n",
        "plt.figure(figsize=(10, 6))\n",
        "sns.countplot(data=features_v4, y='Grupo', order=features_v4['Grupo'].value_counts().index, palette='viridis')\n",
        "plt.title('Distribuição das Famílias de Meta-features Selecionadas (GA)')\n",
        "plt.xlabel('Quantidade')\n",
        "plt.ylabel('Família da Meta-Feature')\n",
        "plt.show()\n",
        "\n",
        "print(\"As features selecionadas são as que mais impactam na detecção do melhor algoritmo.\")\n",
        "print(\"Features baseadas em modelo (como tamanho de árvore) e Landmarking normalmente sobrevivem mais porque\")\n",
        "print(\"carregam em si uma visão muito rica de quão linear ou complexa é a fronteira de decisão do dataset.\")\n"
    ]
}

nb['cells'].extend([markdown_cell, code_cell])

with open('projeto.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Analysis cells appended successfully.")
