# 🌡️ Termômetro da Notícia — Detector Inteligente de Fake News

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-NLP-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-Serverless-000000?style=for-the-badge&logo=flask&logoColor=white)
![TailwindCSS](https://img.shields.io/badge/Tailwind_CSS-Mobile_First-38B2AC?style=for-the-badge&logo=tailwind-css&logoColor=white)
![Vercel](https://img.shields.io/badge/Vercel-Deployed-000000?style=for-the-badge&logo=vercel&logoColor=white)

> **MVP Acadêmico de Ciência de Dados e Processamento de Linguagem Natural (PLN)** projetado em arquitetura 100% Serverless para avaliar o índice de confiabilidade de notícias, links jornalísticos e prints/imagens virais de grupos de WhatsApp.

🔗 ** Repositório:** [github.com/WellyngtonMolinari/termometro-da-noticia](https://github.com/WellyngtonMolinari/termometro-da-noticia)

---

## 🎯 Sobre o Projeto

A disseminação de desinformação ocorre em múltiplos formatos: correntes de texto com apelo emocional, links compartilhados fora de contexto e capturas de tela (*prints*) encaminhadas com frequência em aplicativos de mensagens.

O **Termômetro da Notícia** reúne três frentes de análise em **um único campo inteligente (Omnibox)** que detecta automaticamente o tipo de entrada enviada pelo usuário:

1. **📝 Texto Direto:** Analisa manchetes, parágrafos ou mensagens coladas.
2. **🔗 Link de Site de Notícia (Web Scraping + Domínio):** Acessa a página informada, extrai automaticamente o título e os parágrafos da matéria com `BeautifulSoup`, verifica a reputação do domínio e submete o conteúdo ao modelo.
3. **📷 Print / Imagem Viral de WhatsApp (Visão Computacional / OCR):** Permite arrastar uma imagem, anexar da galeria ou pressionar `Ctrl + V` direto na página. O motor `Tesseract.js` extrai o texto em Português diretamente no navegador e envia para classificação.

---

## 🧠 Como Funciona o Diagnóstico (Data Science & Heurísticas)

O cálculo do **Índice de Confiabilidade (0% a 100%)** combina um pipeline supervisionado de Machine Learning com heurísticas linguísticas de verificação estilística:

1. **Vetorização TF-IDF (`TfidfVectorizer`):**
   - Extração de unigramas e bigramas (`ngram_range=(1, 2)`), remoção de *stopwords* em Português e suavização logarítmica (`sublinear_tf=True`).
2. **Classificador Probabilístico (`MultinomialNB`):**
   - Calcula a probabilidade base `predict_proba` de o vocabulário pertencer à classe jornalística/factual (`1`) ou desinformação/sensacionalismo (`0`).
3. **Camada Heurística e Estilística:**
   - **Sinais de Clickbait e Alarme:** Penaliza excesso de caixa alta (`> 25%`), múltiplos pontos de exclamação (`!`) e termos de urgência artificial (*"URGENTE"*, *"COMPARTILHE ANTES QUE APAGUEM"*, *"A MÍDIA ESCONDE"*).
   - **Marcadores de Corrente de WhatsApp:** Detecta selos extraídos via OCR como *"Encaminhada com frequência"* / *"Mensagem encaminhada"*.
   - **Reputação de Domínio:** Verifica se o link analisado pertence a veículos jornalísticos ou instituições oficiais reconhecidas (`euronews.com`, `g1.globo.com`, `bbc.com`, `reuters.com`, `gov.br`, `fiocruz.br`, etc.).

### Escala do Termômetro

| Faixa de Score | Classificação | Significado |
| :--- | :--- | :--- |
| **>= 65%** | 🟢 **CONFIÁVEL** | Estrutura jornalística informativa e vocabulário compatível com notícias factuais. |
| **40% a 64%** | 🟡 **SUSPEITO** | Incerteza estatística ou mistura de termos informativos com elementos opinativos/alarmistas. |
| **< 40%** | 🔴 **FALSO / SENSACIONALISTA** | Forte presença de padrões de *clickbait*, apelo emocional ou estrutura típica de boatos virais. |

---

## 🏗️ Arquitetura e Stack Tecnológica

O projeto foi desenhado especificamente para respeitar os limites de execução e armazenamento das **Serverless Functions da Vercel** (`< 250 MB`):

- **Frontend:** HTML5 Semântico, Tailwind CSS (via CDN), Vanilla JavaScript (`Fetch API`) e `Tesseract.js v5` (WebAssembly OCR Client-Side).
- **Backend Serverless:** Python + Flask exposto via WSGI na rota `/api/index.py`.
- **Machine Learning & Scraping:** `scikit-learn`, `joblib`, `pandas` (treinamento local), `requests` e `beautifulsoup4`.
- **Resiliência Serverless:** Caso o artefato `api/model.pkl` não esteja presente no container, o backend executa um *fallback* em memória durante o *cold start* em menos de `30ms`.

### Estrutura de Pastas

```text
termometro-da-noticia/
├── api/
│   ├── index.py          # Serverless Function (Flask + ML Inference + Web Scraping)
│   └── model.pkl         # Pipeline serializado (gerado opcionalmente via train.py)
├── public/
│   └── index.html        # Interface Mobile-First com Omnibox (Texto, Link e OCR)
├── train.py              # Script offline para treinar e exportar o modelo (.pkl) com Pandas
├── requirements.txt      # Dependências enxutas de produção para a Vercel
├── vercel.json           # Regras de roteamento Serverless (/api/* e /*)
└── README.md             # Documentação do projeto
```

---

## 🚀 Como Executar Localmente

### 1. Clonar o repositório

```bash
git clone https://github.com/WellyngtonMolinari/termometro-da-noticia.git
cd termometro-da-noticia
```

### 2. Criar um ambiente virtual e instalar as dependências

```bash
python -m venv .venv
# Windows (PowerShell):
.\.venv\Scripts\Activate.ps1
# Linux / macOS:
source .venv/bin/activate

pip install -r requirements.txt pandas
```

### 3. (Opcional) Treinar e gerar o artefato `api/model.pkl`

```bash
python train.py
```

### 4. Iniciar o servidor local

Você pode testar a API diretamente com o Flask ou usando a Vercel CLI:

```bash
python api/index.py
```

---

## 📡 Contrato da API (`POST /api/index`)

### Exemplo 1: Analisar Texto ou Transcrição de Print (OCR)

**Request:**
```json
{
  "text": "URGENTE! Vazou áudio secreto que a grande mídia não quer mostrar! Compartilhe rápido!",
  "from_ocr": true
}
```

### Exemplo 2: Analisar Link de Notícia (Web Scraping)

**Request:**
```json
{
  "url": "https://pt.euronews.com/2026/10/31/franca-mobiliza-70-mil-policias-para-fim-de-semana-de-manifestacoes-e-halloween"
}
```

**Response (`200 OK`):**
```json
{
  "trust_score": 90.7,
  "fake_probability": 9.3,
  "raw_model_score": 72.7,
  "status": "CONFIÁVEL",
  "color": "green",
  "emoji": "🟢",
  "message": "O conteúdo apresenta estrutura jornalística informativa e vocabulário compatível com notícias factuais.",
  "signals": [
    "✅ Domínio jornalístico/institucional reconhecido (pt.euronews.com)"
  ],
  "scraped_article": {
    "domain": "pt.euronews.com",
    "title": "Manifestações em França voltam a degenerar em violência",
    "url": "https://pt.euronews.com/2026/10/31/franca-mobiliza-70-mil-policias-para-fim-de-semana-de-manifestacoes-e-halloween",
    "extracted_text": "..."
  }
}
```

---

## 👨‍💻 Autor

Desenvolvido por **Wellyngton Molinari** como projeto prático de **Ciência de Dados e Processamento de Linguagem Natural**.
