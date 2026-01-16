import os

# Definição das regras (convertendo seus caminhos para o padrão do Git)
# O Git trabalha com caminhos relativos à raiz do projeto.
conteudo_gitignore = """
# --- Pastas Pesadas/Locais (Sua solicitação) ---
data/raw/
notebooks/

# --- Padrões comuns para Python e Análise de Dados (Recomendado) ---
# Ignorar cache do Jupyter Notebook (sempre aparece quando se usa notebooks)
.ipynb_checkpoints/

# Ignorar arquivos compilados do Python
__pycache__/
*.pyc

# Ignorar arquivos de ambiente virtual (se houver)
.venv/
env/

# Ignorar arquivos de configuração local/senhas
.env
"""

# Cria (ou sobrescreve) o arquivo .gitignore na pasta atual
nome_arquivo = ".gitignore"

with open(nome_arquivo, "w") as f:
    f.write(conteudo_gitignore.strip())

print(f"Arquivo {nome_arquivo} gerado com sucesso na pasta atual!")
print("Lembre-se: Se a pasta 'notebooks' já foi commitada antes, rode: git rm -r --cached notebooks/")