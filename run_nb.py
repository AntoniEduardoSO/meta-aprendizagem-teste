import json

with open('projeto.ipynb', 'r') as f:
    nb = json.load(f)

code = ""
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        code += "".join(cell['source']) + "\n\n"

with open('projeto_extracted.py', 'w') as f:
    f.write(code)
