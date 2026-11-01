import os
import re
from urllib.parse import urlparse
import joblib
import requests
from bs4 import BeautifulSoup
from flask import Flask, request, jsonify

app = Flask(__name__)

# Caminho para o arquivo model.pkl na mesma pasta (/api)
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model.pkl")
model = None

if os.path.exists(MODEL_PATH):
    try:
        model = joblib.load(MODEL_PATH)
    except Exception:
        model = None


def treinar_modelo_em_memoria():
    """
    Fallback automatico caso o arquivo model.pkl nao tenha sido commitado no Git.
    Garante que a Serverless Function funcione 100% mesmo sem artefato externo.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.pipeline import Pipeline

    stopwords_pt = [
        "de", "a", "o", "que", "e", "do", "da", "em", "um", "para", "é", "com", "não",
        "uma", "os", "no", "se", "na", "por", "mais", "as", "dos", "como", "mas", "foi",
        "ao", "ele", "das", "tem", "à", "seu", "sua", "ou", "ser", "quando", "muito",
        "há", "nos", "já", "está", "eu", "também", "só", "pelo", "pela", "até", "isso"
    ]

    dados = [
        ("O Banco Central anunciou nesta quarta-feira a manutenção da taxa básica de juros Selic após reunião do Copom, citando projeções de inflação para o próximo trimestre.", 1),
        ("O Ministério da Saúde divulgou o novo calendário nacional de vacinação contra a gripe, priorizando idosos, gestantes e profissionais da saúde em todo o país.", 1),
        ("Pesquisadores da Universidade de São Paulo publicaram um estudo na revista Nature demonstrando avanços no diagnóstico precoce de doenças cardiovasculares.", 1),
        ("O Instituto Brasileiro de Geografia e Estatística informou que o Produto Interno Bruto cresceu 0,8% no segundo trimestre, impulsionado pelo setor de serviços e indústria.", 1),
        ("A Agência Nacional de Energia Elétrica aprovou o acionamento da bandeira tarifária verde para o próximo mês devido ao nível favorável dos reservatórios das hidrelétricas.", 1),
        ("O Congresso Nacional aprovou em segundo turno o texto-base da regulamentação tributária após meses de debates entre parlamentares e governadores.", 1),
        ("Dados do Instituto Nacional de Pesquisas Espaciais apontam previsão de chuvas acima da média na região Sul do Brasil durante a próxima semana.", 1),
        ("O Ministério da Educação abriu as inscrições para o Exame Nacional do Ensino Médio através da página do participante, com prazo até a próxima sexta-feira.", 1),
        ("Relatório divulgado pela Organização Mundial da Saúde destaca a importância de políticas públicas de saneamento básico para redução de infecções hospitalares.", 1),
        ("A prefeitura iniciou as obras de recapeamento asfáltico e ampliação do corredor de ônibus na avenida principal, segundo nota oficial da Secretaria de Transportes.", 1),
        ("O Supremo Tribunal Federal julgou procedente a ação direta de inconstitucionalidade referente à cobrança retroativa de impostos estaduais sobre pequenas empresas.", 1),
        ("Exportações brasileiras do agronegócio registraram superávit comercial no acumulado do ano, segundo balanço divulgado pelo Ministério do Desenvolvimento.", 1),
        ("Cientistas da Fiocruz concluíram a fase de testes clínicos de um novo kit de diagnóstico rápido para dengue, zika e chikungunya na rede pública.", 1),
        ("A Polícia Federal deflagrou operação em três estados para cumprir mandados de busca e apreensão autorizados pela Justiça Federal contra fraudes licitatórias.", 1),
        ("O índice de desemprego recuou no trimestre encerrado em agosto, atingindo a menor taxa para o período na série histórica da Pesquisa Nacional por Amostra de Domicílios.", 1),
        ("A França mobilizou cerca de 70 mil policiais e forças de segurança neste fim de semana para acompanhar manifestações e celebrações de Halloween em diversas cidades do país, informou o Ministério do Interior.", 1),
        ("Autoridades europeias reforçaram o policiamento preventivo nas capitais diante de protestos agendados e eventos públicos, segundo comunicado oficial divulgado pela imprensa.", 1),
        ("URGENTE! Vazou áudio secreto que a grande mídia não quer mostrar! Compartilhe rápido antes que apaguem esse vídeo das redes sociais!", 0),
        ("CHOCANTE: Médicos descobriram que tomar água morna com limão e bicarbonato em jejum cura 100% de qualquer tipo de câncer em apenas 3 dias! A indústria farmacêutica esconde isso!", 0),
        ("ATENÇÃO! O governo acaba de aprovar uma lei secreta de madrugada para confiscar toda a poupança e cobrar imposto de cada mensagem enviada no WhatsApp!", 0),
        ("BOMBA! Cientistas confirmam que o sinal 5G das antenas ativa microchips injetados para controlar a mente da população mundial! Repasse para sua família!", 0),
        ("ESCÂNDALO! Caminhões com urnas adulteradas foram flagrados em galpão clandestino! Veja as provas definitivas que o tribunal mandou censurar hoje!", 0),
        ("NÃO CAIA NESSA! Beber chá de casca de banana fervida com alho substitui qualquer vacina e garante imunidade total imediata segundo médico banido da TV!", 0),
        ("ALERTA GERAL! Novo decreto obriga todos os proprietários de imóveis a dividir suas casas com desconhecidos a partir de segunda-feira! Compartilhe nos grupos!", 0),
        ("REVELADO! Bilionários globais estão colocando substâncias químicas nas nuvens para destruir as plantações e forçar o povo a comer insetos sintéticos!", 0),
        ("ABSURDO! Veja a lista secreta de artistas famosos que receberam bilhões de dinheiro público desviado da merenda escolar para apoiar a ditadura!", 0),
        ("CUIDADO! Se você receber uma ligação desse número, seu celular será clonado na mesma hora e sua conta bancária será zerada em segundos! Avise a todos!", 0),
        ("MILAGRE ESCONDIDO! Planta amazônica proscrita pelos laboratórios regenera a visão completamente em 24 horas sem precisar de cirurgia ou óculos!", 0),
        ("URGENTE! Exército já está nas ruas cercando o parlamento após descoberta de fraude bilionária! A televisão está passando programação gravada para esconder a verdade!", 0),
        ("GOLPE CONFIRMADO! Bancos vão bloquear o PIX de quem não fizer recadastramento biométrico clicando neste link exclusivo até meia-noite de hoje!", 0),
        ("VERDADE OCULTA! Astronauta confessa em leito de morte que as fotos do espaço são montagens de estúdio e que existe um muro de gelo gigante!", 0),
        ("AVISO URGENTE! Quem tiver mais de 5 mil reais na conta corrente terá o CPF bloqueado amanhã cedo por ordem secreta! Tire seu dinheiro do banco agora!", 0),
    ]

    textos = [d[0] for d in dados]
    labels = [d[1] for d in dados]

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(stop_words=stopwords_pt, ngram_range=(1, 2), max_features=3000, lowercase=True, sublinear_tf=True)),
        ("clf", MultinomialNB(alpha=0.5)),
    ])
    pipeline.fit(textos, labels)
    return pipeline


if model is None:
    model = treinar_modelo_em_memoria()

TERMOS_SENSACIONALISTAS = [
    "urgente", "chocante", "bomba", "compartilhe", "antes que apaguem",
    "mídia não mostra", "segredo", "cura milagrosa", "escondido", "alerta geral",
    "repasse", "censurado", "escândalo", "não caia nessa", "verdade oculta"
]

MARCADORES_WHATSAPP = [
    "encaminhada com frequência",
    "encaminhada com frequencia",
    "mensagem encaminhada",
    "frequently forwarded",
    "repasse para todos",
    "compartilhe nos grupos",
]

DOMINIOS_JORNALISTICOS_CONFIAVEIS = [
    "euronews.com", "g1.globo.com", "globo.com", "uol.com.br", "folha.uol.com.br",
    "estadao.com.br", "bbc.com", "bbc.co.uk", "reuters.com", "dw.com", "cnnbrasil.com.br",
    "agenciabrasil.ebc.com.br", "gov.br", "FAPESP.br", "usp.br", "fiocruz.br",
    "nytimes.com", "elpais.com", "rtp.pt", "publico.pt", "observador.pt"
]


def extrair_noticia_de_url(url: str) -> dict:
    """
    Faz scraping leve da URL informada, extraindo titulo, descricao e paragrafos
    principais da materia jornalistica (com fallback para o slug da URL caso haja bloqueio).
    """
    parsed = urlparse(url.strip())
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise ValueError("URL inválida. Certifique-se de incluir http:// ou https://")

    dominio = parsed.netloc.lower().removeprefix("www.")
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "pt-BR,pt;q=0.9,en-US;q=0.8,en;q=0.7",
    }

    titulo = ""
    descricao = ""
    paragrafos = []
    usou_fallback_slug = False

    try:
        resp = requests.get(url.strip(), headers=headers, timeout=8)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        # Remover tags ruidosas
        for tag in soup(["script", "style", "noscript", "nav", "footer", "header", "aside", "form"]):
            tag.decompose()

        # 1. Tentar extrair Titulo (og:title ou <title> ou <h1>)
        og_title = soup.find("meta", property="og:title") or soup.find("meta", attrs={"name": "twitter:title"})
        if og_title and og_title.get("content"):
            titulo = og_title["content"].strip()
        elif soup.find("h1"):
            titulo = soup.find("h1").get_text(" ", strip=True)
        elif soup.title and soup.title.string:
            titulo = soup.title.string.strip()

        # 2. Tentar extrair Meta Description
        og_desc = (
            soup.find("meta", property="og:description")
            or soup.find("meta", attrs={"name": "description"})
        )
        if og_desc and og_desc.get("content"):
            descricao = og_desc["content"].strip()

        # 3. Extrair paragrafos do corpo (<article> ou <p>)
        container = soup.find("article") or soup.find("main") or soup
        for p in container.find_all("p"):
            txt = p.get_text(" ", strip=True)
            if len(txt) >= 45:
                paragrafos.append(txt)
            if len(paragrafos) >= 12:
                break

    except Exception:
        # Fallback inteligente: se o site retornar 404/403 (ex: data futura ou paywall),
        # reconstrói a manchete a partir do slug da URL para nao deixar o usuario na mao
        usou_fallback_slug = True

    corpo = "\n\n".join(paragrafos)
    partes = [p for p in [titulo, descricao, corpo] if p]
    texto_completo = "\n\n".join(partes).strip()

    if len(texto_completo) < 25:
        usou_fallback_slug = True
        slug_limpo = re.sub(r"[-_/]+", " ", parsed.path).strip()
        slug_limpo = re.sub(r"\b\d{2,4}\b", "", slug_limpo).strip()
        slug_limpo = re.sub(r"\s+", " ", slug_limpo)
        titulo = titulo or f"Notícia de {dominio}: {slug_limpo.capitalize()}"
        texto_completo = f"{titulo}. Fonte: {dominio}."

    return {
        "url": url.strip(),
        "domain": dominio,
        "title": titulo or f"Matéria extraída de {dominio}",
        "extracted_text": texto_completo[:4500],
        "used_slug_fallback": usou_fallback_slug,
    }


def analisar_heuristica_sensacionalismo(
    texto: str, origem_ocr: bool = False, dominio_url: str = "", usou_slug_fallback: bool = False
) -> dict:
    letras = [c for c in texto if c.isalpha()]
    proporcao_maiusculas = (
        sum(1 for c in letras if c.isupper()) / len(letras) if len(letras) > 10 else 0.0
    )
    exclamacoes = texto.count("!")
    texto_lower = texto.lower()
    gatilhos_encontrados = [t for t in TERMOS_SENSACIONALISTAS if t in texto_lower]
    marcadores_zap = [m for m in MARCADORES_WHATSAPP if m in texto_lower]

    penalidade = 0.0
    bonus = 0.0
    sinais = []

    if origem_ocr:
        sinais.append("Texto extraído de imagem/print via OCR (Visão Computacional)")

    if dominio_url:
        dominio_confiavel = any(
            dominio_url == d or dominio_url.endswith("." + d)
            for d in DOMINIOS_JORNALISTICOS_CONFIAVEIS
        )
        if dominio_confiavel:
            bonus += 0.18
            sinais.append(f"✅ Domínio jornalístico/institucional reconhecido ({dominio_url})")
        else:
            sinais.append(f"🌐 Conteúdo extraído do domínio externo: {dominio_url}")

        if not dominio_url.startswith("") and "http://" in dominio_url:
            penalidade += 0.10

        if usou_slug_fallback:
            sinais.append("ℹ️ Página original indisponível/protegida; análise complementada pela manchete da URL e domínio")

    if marcadores_zap:
        penalidade += 0.20
        sinais.append(f"Selo de corrente viral do WhatsApp detectado ('{marcadores_zap[0]}')")

    if proporcao_maiusculas > 0.25:
        penalidade += 0.15
        sinais.append("Uso elevado de LETRAS MAIÚSCULAS")

    if exclamacoes >= 2:
        penalidade += min(0.15, exclamacoes * 0.05)
        sinais.append(f"Pontuação alarmista ({exclamacoes} pontos de exclamação)")

    if gatilhos_encontrados:
        penalidade += min(0.25, len(gatilhos_encontrados) * 0.10)
        sinais.append(f"Gatilhos de apelação emocional ({', '.join(gatilhos_encontrados[:3])})")

    return {"penalidade": penalidade, "bonus": bonus, "sinais": sinais}


# Aceita qualquer caminho que a Vercel encaminhe para /api/index.py
@app.route("/", defaults={"path": ""}, methods=["GET", "POST"])
@app.route("/<path:path>", methods=["GET", "POST"])
def catch_all(path):
    if request.method == "GET":
        return jsonify({
            "status": "ok",
            "model_loaded": model is not None,
            "route_received": path
        })

    data = request.get_json(silent=True) or {}
    url_recebida = (data.get("url") or "").strip()
    texto = (data.get("text") or "").strip()
    origem_ocr = bool(data.get("from_ocr", False))

    # Se o usuario colou um link direto na caixa de texto, detectamos automaticamente
    if not url_recebida and re.match(r"^https?://\S+$", texto):
        url_recebida = texto

    meta_url = None
    dominio_url = ""
    usou_slug_fallback = False

    if url_recebida:
        try:
            meta_url = extrair_noticia_de_url(url_recebida)
            texto = meta_url["extracted_text"]
            dominio_url = meta_url["domain"]
            usou_slug_fallback = meta_url["used_slug_fallback"]
        except ValueError as ve:
            return jsonify({"error": str(ve)}), 400
        except Exception as e:
            return jsonify({"error": f"Não foi possível acessar o link informado: {str(e)}"}), 400

    if len(texto) < 15:
        return jsonify({
            "error": "Por favor, insira um texto, link válido ou print de notícia com pelo menos 15 caracteres."
        }), 400

    try:
        probs = model.predict_proba([texto])[0]
        prob_modelo_confiavel = float(probs[1])

        heuristica = analisar_heuristica_sensacionalismo(
            texto,
            origem_ocr=origem_ocr,
            dominio_url=dominio_url,
            usou_slug_fallback=usou_slug_fallback,
        )
        score_final = max(
            0.02,
            min(0.98, prob_modelo_confiavel + heuristica["bonus"] - heuristica["penalidade"])
        )
        score_percentual = round(score_final * 100, 1)

        if score_final >= 0.65:
            status = "CONFIÁVEL"
            cor = "green"
            emoji = "🟢"
            mensagem = (
                "O conteúdo apresenta estrutura jornalística informativa e vocabulário "
                "compatível com notícias factuais."
            )
        elif score_final >= 0.40:
            status = "SUSPEITO"
            cor = "yellow"
            emoji = "🟡"
            mensagem = (
                "O modelo identificou incerteza ou mistura de termos informativos com "
                "elementos opinativos/alarmistas. Recomenda-se checar fontes oficiais."
            )
        else:
            status = "FALSO / SENSACIONALISTA"
            cor = "red"
            emoji = "🔴"
            mensagem = (
                "Alta probabilidade de desinformação! O conteúdo contém fortes padrões de "
                "clickbait, apelo emocional ou estrutura típica de correntes virais."
            )

        resposta = {
            "trust_score": score_percentual,
            "fake_probability": round(100.0 - score_percentual, 1),
            "raw_model_score": round(prob_modelo_confiavel * 100, 1),
            "status": status,
            "color": cor,
            "emoji": emoji,
            "message": mensagem,
            "signals": heuristica["sinais"],
        }

        if meta_url:
            resposta["scraped_article"] = {
                "url": meta_url["url"],
                "domain": meta_url["domain"],
                "title": meta_url["title"],
                "extracted_text": meta_url["extracted_text"],
            }

        return jsonify(resposta)

    except Exception as e:
        return jsonify({"error": f"Erro ao processar a notícia: {str(e)}"}), 500
