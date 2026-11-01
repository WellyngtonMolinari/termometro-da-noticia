import os
import re
import joblib
from flask import Flask, request, jsonify

app = Flask(__name__)

# Caminho absoluto para o arquivo model.pkl na mesma pasta (/api)
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model.pkl")
model = None

if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)

# Termos tipicos de clickbait / alarmismo para enriquecer a heuristica em conjunto com o TF-IDF
TERMOS_SENSACIONALISTAS = [
    "urgente", "chocante", "bomba", "compartilhe", "antes que apaguem",
    "mídia não mostra", "segredo", "cura milagrosa", "escondido", "alerta geral",
    "repasse", "censurado", "escândalo", "não caia nessa", "verdade oculta"
]


def analisar_heuristica_sensacionalismo(texto: str) -> dict:
    """
    Extrai sinais estilisticos comuns em desinformacao (excesso de caixa alta,
    exclamacoes repetidas e jargoes de corrente) para calibrar o termometro.
    """
    letras = [c for c in texto if c.isalpha()]
    proporcao_maiusculas = (
        sum(1 for c in letras if c.isupper()) / len(letras) if len(letras) > 10 else 0.0
    )
    exclamacoes = texto.count("!")
    texto_lower = texto.lower()
    gatilhos_encontrados = [t for t in TERMOS_SENSACIONALISTAS if t in texto_lower]

    penalidade = 0.0
    sinais = []

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


@app.route("/api/predict", methods=["POST"])
@app.route("/api/index", methods=["POST"])
@app.route("/api", methods=["POST"])
def predict():
    if model is None:
        return jsonify({
            "error": "Modelo não encontrado no servidor. Execute 'python train.py' antes do deploy."
        }), 500

    data = request.get_json(silent=True) or {}
    texto = (data.get("text") or "").strip()

    if len(texto) < 15:
        return jsonify({
            "error": "Por favor, insira um trecho de notícia com pelo menos 15 caracteres para análise."
        }), 400

    try:
        # 1. Probabilidade do classificador Naive Bayes + TF-IDF (classe 1 = Confiavel, 0 = Falso)
        probs = model.predict_proba([texto])[0]
        prob_modelo_confiavel = float(probs[1])

        # 2. Ajuste leve com sinais estilisticos de clickbait
        heuristica = analisar_heuristica_sensacionalismo(texto)
        score_final = max(0.02, min(0.98, prob_modelo_confiavel - heuristica["penalidade"]))
        score_percentual = round(score_final * 100, 1)

        # 3. Regra de Negocio: Termometro de Confiabilidade
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
                "Alta probabilidade de desinformação! O texto contém fortes padrões de "
                "clickbait, apelo emocional ou estrutura típica de boatos virais."
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


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model_loaded": model is not None})


if __name__ == "__main__":
    app.run(debug=True, port=5000)

