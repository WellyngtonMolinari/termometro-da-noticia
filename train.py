"""
Script de Treinamento Offline - Detector de Fake News (MVP)
Executa localmente para gerar o pipeline serializado (.pkl) que sera consumido
pela Serverless Function na Vercel.
"""

import os
import re
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score

# Stopwords basicas em Portugues para evitar dependencia de download do NLTK no serverless
STOPWORDS_PT = [
    "de", "a", "o", "que", "e", "do", "da", "em", "um", "para", "é", "com", "não",
    "uma", "os", "no", "se", "na", "por", "mais", "as", "dos", "como", "mas", "foi",
    "ao", "ele", "das", "tem", "à", "seu", "sua", "ou", "ser", "quando", "muito",
    "há", "nos", "já", "está", "eu", "também", "só", "pelo", "pela", "até", "isso",
    "ela", "entre", "era", "depois", "sem", "mesmo", "aos", "ter", "seus", "quem",
    "nas", "me", "esse", "eles", "estão", "você", "tinha", "foram", "essa", "num",
    "nem", "suas", "meu", "às", "minha", "têm", "numa", "pelos", "elas", "havia",
    "seja", "qual", "será", "nós", "tenho", "lhe", "deles", "essas", "esses",
    "pelas", "este", "fosse", "dele", "tu", "te", "vocês", "vos", "lhes", "meus",
    "minhas", "teu", "tua", "teus", "tuas", "nosso", "nossa", "nossos", "nossas"
]


def get_training_data() -> pd.DataFrame:
    """
    Cria um dataset sintetico balanceado (0 = Falso/Fake, 1 = Verdadeiro/Confiavel)
    com padroes linguisticos reais do jornalismo vs desinformacao/sensacionalismo.
    """
    dados = [
        # --- NOTICIAS VERDADEIRAS (Label = 1) ---
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
        ("A comissão organizadora confirmou as datas do festival literário municipal, que contará com palestras gratuitas de autores nacionais e oficinas para estudantes.", 1),
        ("A Agência Nacional de Vigilância Sanitária autorizou o registro de um novo medicamento genérico indicado para o controle de hipertensão arterial crônica.", 1),
        ("Segundo boletim epidemiológico da secretaria estadual, houve queda de 15% nas internações por síndromes respiratórias agudas em comparação ao ano anterior.", 1),
        ("O conselho universitário aprovou a criação de novas bolsas de iniciação científica voltadas para projetos de transição energética e sustentabilidade.", 1),
        ("Acordo bilateral assinado entre países do Mercosul e União Europeia prevê redução gradual de tarifas alfandegárias para produtos industriais ao longo de dez anos.", 1),

        # --- NOTICIAS FALSAS / SENSACIONALISTAS (Label = 0) ---
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
        ("PERIGO NAS ESCOLAS! Novo material didático distribuído em segredo ensina crianças a abandonarem suas famílias! Veja o vídeo antes que seja deletado!", 0),
        ("EXCLUSIVO! Juiz internacional anula todas as dívidas de cartão de crédito no país após descobrir esquema dos banqueiros! Saiba como resgatar seu dinheiro!", 0),
        ("PÂNICO! Lote de alimentos contaminados com bactéria mortal está sendo vendido nos supermercados e as autoridades proibiram os médicos de falar!", 0),
        ("INACREDITÁVEL! Café misturado com óleo de coco derrete 15 quilos de gordura abdominal em uma semana sem dieta! Nutricionistas estão furiosos com o segredo!", 0),
        ("DENÚNCIA GRAVE! Satélites estrangeiros estão disparando raios laser para causar secas e incêndios propositais nas fazendas! A prova está aqui!", 0),
        ("AVISO URGENTE! Quem tiver mais de 5 mil reais na conta corrente terá o CPF bloqueado amanhã cedo por ordem secreta! Tire seu dinheiro do banco agora!", 0),
    ]

    return pd.DataFrame(dados, columns=["texto", "label"])


def train_and_export():
    print("1. Carregando dataset de treinamento com Pandas...")
    df = get_training_data()
    print(f"   Total de amostras: {len(df)} ({df['label'].sum()} verdadeiras, {len(df) - df['label'].sum()} falsas)")

    print("2. Construindo Pipeline Scikit-Learn (TF-IDF + Multinomial Naive Bayes)...")
    pipeline = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                stop_words=STOPWORDS_PT,
                ngram_range=(1, 2),
                max_features=3000,
                lowercase=True,
                sublinear_tf=True,
            ),
        ),
        ("clf", MultinomialNB(alpha=0.5)),
    ])

    # Avaliacao rapida com validacao cruzada
    scores = cross_val_score(pipeline, df["texto"], df["label"], cv=5, scoring="accuracy")
    print(f"   Acuracia media (5-Fold CV): {scores.mean():.2%}")

    print("3. Treinando modelo final com todo o corpus...")
    pipeline.fit(df["texto"], df["label"])

    # Garantir que a pasta /api exista e salvar o modelo la para que a Vercel empacote junto
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "api")
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, "model.pkl")

    joblib.dump(pipeline, model_path)
    print(f"4. Modelo exportado com sucesso em: {model_path}")

    # Teste rapido de sanidade
    exemplos = [
        "O Banco Central divulgou o relatório trimestral de inflação segundo dados do IBGE.",
        "URGENTE! Compartilhe antes que apaguem! Chá de limão cura tudo em 24 horas!",
    ]
    for texto in exemplos:
        proba_true = pipeline.predict_proba([texto])[0][1]
        print(f"   Teste: '{texto[:55]}...' -> Confiabilidade: {proba_true:.1%}")


if __name__ == "__main__":
    train_and_export()

