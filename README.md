# Análise Comparativa FALL vs CONTROLE

App Streamlit com análise estatística comparativa entre grupos FALL e CONTROLE.

## Como fazer deploy no Streamlit Cloud

### 1. Crie um repositório no GitHub

1. Acesse [github.com](https://github.com) e clique em **New repository**
2. Dê um nome (ex: `fall-controle-analise`)
3. Deixe **Public** (obrigatório para o plano gratuito do Streamlit Cloud)
4. Clique em **Create repository**

### 2. Faça upload dos arquivos

Na página do repositório, clique em **Add file → Upload files** e envie:
- `app.py`
- `requirements.txt`
- A pasta `.streamlit/` com `config.toml`
- A pasta `data/` com os 4 arquivos `.xlsx`

Ou, se tiver Git instalado:
```bash
git clone https://github.com/SEU_USUARIO/fall-controle-analise
cd fall-controle-analise
# copie todos os arquivos desta pasta para cá
git add .
git commit -m "first commit"
git push
```

### 3. Deploy no Streamlit Cloud

1. Acesse [share.streamlit.io](https://share.streamlit.io)
2. Faça login com sua conta GitHub
3. Clique em **New app**
4. Selecione o repositório e o arquivo `app.py`
5. Clique em **Deploy!**

Pronto — em ~2 minutos seu app estará online com URL pública.

## Estrutura do projeto

```
fall_app/
├── app.py                    # App principal
├── requirements.txt          # Dependências Python
├── .streamlit/
│   └── config.toml          # Tema claro
└── data/
    ├── metricas_individuaisFALL.xlsx
    ├── resultante_grupoFALL.xlsx
    ├── metricas_individuaisCONTROLE.xlsx
    └── resultante_grupoCONTROLE.xlsx
```

## Funcionalidades

- **📈 Curvas Resultantes** — sobreposição com bandas ±1DP, curva de diferença, RMSE, correlação, área entre curvas e análise por fase (P1/P2/P3)
- **📊 Estatísticas por Métrica** — Mann-Whitney U, d de Cohen, correção Benjamini-Hochberg, filtros por significância e tamanho de efeito
- **🔬 Análise Temporal (SPM)** — z-test ponto-a-ponto nos 300 pontos da curva, identificação automática de regiões significativas
- **🧪 Análises Adicionais** — boxplot com jitter por métrica, radar normalizado, coeficiente de variação intragrupal
