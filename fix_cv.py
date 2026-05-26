import json

with open('projeto.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        
        if "from sklearn.model_selection import cross_val_score, StratifiedKFold" in source:
            source = source.replace(
                "from sklearn.model_selection import cross_val_score, StratifiedKFold", 
                "from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold, LeaveOneOut"
            )
            
        if "StratifiedKFold(" in source:
            # We replace StratifiedKFold with KFold in the meta-evaluator because n=5 is too small for stratification
            # We also change it for cv_eval in executar_pipeline_wrapper and cv_strategy in GeneticFeatureSelector
            source = source.replace("StratifiedKFold(n_splits=3", "KFold(n_splits=3")
            
        cell['source'] = [line + "\n" if not line.endswith("\n") else line for line in source.split("\n")]
        # clean trailing empty element if exists and remove last newline if original didn't have it
        if cell['source'][-1] == "\n":
            cell['source'] = cell['source'][:-1]
        if len(cell['source']) > 0 and cell['source'][-1].endswith('\n') and not source.endswith('\n'):
            cell['source'][-1] = cell['source'][-1][:-1]

with open('projeto.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("CV fixed.")
