import os
import joblib
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


def analisar_heuristica_sensacionalismo(texto: str, origem_ocr: bool = False) -> dict:
    letras = [c for c in texto if c.isalpha()]
    proporcao_maiusculas = (
        sum(1 for c in letras if c.isupper()) / len(letras) if len(letras) > 10 else 0.0
    )
    exclamacoes = texto.count("!")
    texto_lower = texto.lower()
    gatilhos_encontrados = [t for t in TERMOS_SENSACIONALISTAS if t in texto_lower]
    marcadores_zap = [m for m in MARCADORES_WHATSAPP if m in texto_lower]

    penalidade = 0.0
    sinais = []

    if origem_ocr:
        sinais.append("Texto extraído de imagem/print via OCR (Visão Computacional)")

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

    return {"penalidade": penalidade, "sinais": sinais}


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
    texto = (data.get("text") or "").strip()
    origem_ocr = bool(data.get("from_ocr", False))

    if len(texto) < 15:
        return jsonify({
            "error": "Por favor, insira ou extraia um texto de notícia com pelo menos 15 caracteres para análise."
        }), 400

    try:
        probs = model.predict_proba([texto])[0]
        prob_modelo_confiavel = float(probs[1])

        heuristica = analisar_heuristica_sensacionalismo(texto, origem_ocr=origem_ocr)
        score_final = max(0.02, min(0.98, prob_modelo_confiavel - heuristica["penalidade"]))
        score_percentual = round(score_final * 100, 1)

        if score_final >= 0.65:
            status = "CONFIÁVEL"
            cor = "green"
            emoji = "🟢"
            mensagem = (
                "O texto apresenta estrutura jornalística informativa e vocabulário "
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
                "clickbait, apelo emocional ou estrutura típica de correntes virais de WhatsApp."
            )

        return jsonify({
            "trust_score": score_percentual,
            "fake_probability": round(100.0 - score_percentual, 1),
            "raw_model_score": round(prob_modelo_confiavel * 100, 1),
            "status": status,
            "color": cor,
            "emoji": emoji,
            "message": mensagem,
            "signals": heuristica["sinais"]
        })

    except Exception as e:
        return jsonify({"error": f"Erro ao processar a notícia: {str(e)}"}), 500
