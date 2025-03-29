# --- modulos

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing import StandardScaler
import logging 

# --- configs

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__) 

MODEL_DIR = './data/models/'
DATA_SPLIT_FILE = './data/train_test.joblib'

MODEL_FILES = {
    'MLP': os.path.join(MODEL_DIR, 'mlp.joblib'),
    'SVM': os.path.join(MODEL_DIR, 'svm.joblib'),
    'DecisionTree': os.path.join(MODEL_DIR, 'dt.joblib'),
    'RandomForest': os.path.join(MODEL_DIR, 'rf.joblib')
}

st.set_page_config(
    layout="wide",
    initial_sidebar_state="expanded", 
    page_title="Simulador de Risco de Crédito",
    page_icon="📈",
    )


# --- funcoes
@st.cache_resource
def load_models():
    loaded_models = {}
    logger.info("Iniciando carregamento dos modelos...")

    try:
        for name, filepath in MODEL_FILES.items():

            if os.path.exists(filepath): # tem modelo?
                model = joblib.load(filepath)

                if hasattr(model, "predict_proba"): # modelo tem predict_proba?
                    loaded_models[name] = model
                    logger.info(f"Modelo '{name}' carregado com sucesso de {filepath}.")
                else:
                    logger.warning(f"Modelo '{name}' carregado, mas não suporta 'predict_proba'. Será ignorado no Soft Voting.")
            else:
                logger.error(f"Arquivo do modelo '{name}' não encontrado em {filepath}.")
                st.error(f"Erro Crítico: Arquivo do modelo '{name}' não encontrado. Verifique o caminho.")
                return None 
        
        if not loaded_models:
             logger.warning("Nenhum modelo com suporte a 'predict_proba' foi carregado.")
        
        return {k: v for k, v in loaded_models.items() if hasattr(v, "predict_proba")}
    
    except Exception as e:
        logger.error(f"Exceção durante o carregamento dos modelos: {e}", exc_info=True) # Loga o traceback
        st.error(f"Erro ao carregar modelos: {e}")
        return None

@st.cache_resource
def get_scaler():
    logger.info("Iniciando carregamento de dados para ajuste do Scaler...")
    try:
        X_train, _, _, _ = joblib.load(DATA_SPLIT_FILE)
        scaler = StandardScaler()
        scaler.fit(X_train)
        logger.info("StandardScaler ajustado com sucesso.")
        return scaler
    except FileNotFoundError:
        logger.error(f"Arquivo de dados para o Scaler não encontrado: {DATA_SPLIT_FILE}", exc_info=True)
        st.error(f"Erro Crítico: Arquivo '{DATA_SPLIT_FILE}' não encontrado.")
        return None
    except Exception as e:
        logger.error(f"Exceção durante o ajuste do Scaler: {e}", exc_info=True)
        st.error(f"Erro ao carregar/ajustar o Scaler: {e}")
        return None

# --- inicializacao
models = load_models()
scaler = get_scaler()

# --- Interface streamlit
st.title("📈 Simulador de Análise de Risco de Crédito (Soft Voting)")
st.markdown("Insira os dados do cliente para avaliar a probabilidade de pagamento do empréstimo usando a média das probabilidades dos modelos.")

if models is None or scaler is None or not models: # sem modelo ou scaler?
    logger.critical("Aplicação não pôde iniciar devido à falha no carregamento de modelos ou scaler.")
    st.error("A aplicação não pôde ser iniciada. Verifique os logs do console para mais detalhes.")
else:
    logger.info(f"Aplicação iniciada com {len(models)} modelos carregados e scaler pronto.")
    st.sidebar.header("⚙️ Dados do Cliente")

    # inputs
    income_stats = {'min': 20014, 'max': 69995, 'mean': 45331}
    age_stats = {'min': 18, 'max': 64, 'mean': 41}
    loan_stats = {'min': 1, 'max': 13766, 'mean': 4444}

    input_income = st.sidebar.number_input(
        "Renda Anual (R$)", min_value=0.0, value=50000.0, step=1000.0, format="%.2f",
        help=f"Dados originais: Min={income_stats['min']:,.0f}, Max={income_stats['max']:,.0f}, Média={income_stats['mean']:,.0f}"
    )
    input_age = st.sidebar.number_input(
        "Idade", min_value=18, max_value=100, value=40, step=1,
        help=f"Dados originais: Min={age_stats['min']}, Max={age_stats['max']}, Média={age_stats['mean']}"
    )
    input_loan = st.sidebar.number_input(
        "Valor do Empréstimo Solicitado (R$)", min_value=0.0, value=5000.0, step=100.0, format="%.2f",
        help=f"Dados originais: Min={loan_stats['min']:,.0f}, Max={loan_stats['max']:,.0f}, Média={loan_stats['mean']:,.0f}"
    )

    analyze_button = st.sidebar.button("📊 Analisar Risco e Sugerir Limite")

    st.divider()
    st.header("Resultados da Análise")

    if analyze_button: # clicou? some msg inicial e inicia
        logger.info(f"Botão 'Analisar' pressionado. Dados de entrada: Renda={input_income}, Idade={input_age}, Empréstimo={input_loan}")

        # 1. preparar dados
        input_data = {'income': [input_income], 'age': [input_age], 'loan': [input_loan]}
        input_df = pd.DataFrame(input_data)
        logger.debug(f"DataFrame de entrada criado: {input_df.to_dict()}")

        # 2. escalonar
        try:
            input_scaled = scaler.transform(input_df.values)
            logger.info("Pré-processamento (scaling) aplicado aos dados de entrada.")
            logger.debug(f"Dados escalonados: {input_scaled}")
        except Exception as e:
            logger.error(f"Erro durante o pré-processamento: {e}", exc_info=True)
            st.error(f"Erro durante o pré-processamento: {e}")
            st.stop() 

        # 3. probabilidades de cada modelo
        all_probabilities = []
        model_details = {}

        logger.info("Iniciando previsão de probabilidades com os modelos...")
        with st.spinner('Calculando probabilidades médias...'):
            for name, model in models.items():
                try:
                    proba = model.predict_proba(input_scaled)[0]
                    all_probabilities.append(proba)
                    model_details[name] = {'paga': proba[0], 'nao_paga': proba[1]}
                    logger.debug(f"Modelo '{name}' - Probabilidades: Paga={proba[0]:.4f}, Não Paga={proba[1]:.4f}")
                except Exception as e:
                    logger.warning(f"Erro ao obter probabilidades do modelo '{name}': {e}", exc_info=True)
                    model_details[name] = {'paga': 'Erro', 'nao_paga': 'Erro'}

        # 4. prob avg
        if all_probabilities: # vetor não vazio?
            avg_proba = np.mean(all_probabilities, axis=0)
            avg_prob_pay = avg_proba[0]
            avg_prob_default = avg_proba[1]
            avg_pay_probability_percent = avg_prob_pay * 100
            logger.info(f"Probabilidade média calculada: Paga={avg_prob_pay:.4f}, Não Paga={avg_prob_default:.4f}")
        else:
            logger.error("Não foi possível calcular probabilidades médias - nenhum modelo retornou probabilidades.")
            st.error("Não foi possível calcular probabilidades médias.")
            avg_pay_probability_percent = 0
            avg_prob_pay = 0
            avg_prob_default = 0

        # 5. limite sugerido
        suggested_limit = input_loan * avg_prob_pay
        logger.info(f"Limite sugerido calculado: R$ {suggested_limit:.2f}")

        # 6. render em results
        st.subheader("Sumário da Avaliação (Soft Voting):")
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(
                label="Probabilidade Média de Pagamento", 
                value=f"{avg_pay_probability_percent:.1f}%", 
                help=f"Média das probabilidades da classe 'Paga' (0) entre {len(all_probabilities)} modelos."
            )
        
        with col2:
             st.metric(
                 label="Limite de Empréstimo Sugerido", 
                 value=f"R$ {suggested_limit:,.2f}", 
                 help=f"Calculado como {avg_pay_probability_percent:.1f}% do valor solicitado (R$ {input_loan:,.2f})"
            )
        
        # decisao
        if avg_pay_probability_percent >= 70:
            st.success("✔️ **Risco Avaliado:** Baixo a Moderado")
            final_decision = "Aprovar (com limite sugerido)"
        elif avg_pay_probability_percent >= 50:
            st.warning("⚠️ **Risco Avaliado:** Moderado a Alto")
            final_decision = "Análise Adicional / Aprovar com Cautela"
        else:
            st.error("❌ **Risco Avaliado:** Alto")
            final_decision = "Recusar ou Análise Muito Criteriosa"
        
        st.info(f"**Decisão Sugerida:** {final_decision}")
        logger.info(f"Resultado final: Prob Pagamento={avg_pay_probability_percent:.1f}%, Limite=R${suggested_limit:.2f}, Decisão={final_decision}")

        with st.expander("Ver probabilidades detalhadas por modelo"):
             details_list = []
             for name, probs in model_details.items():
                 details_list.append({
                     'Modelo': name,
                     'Prob. Pagar': f"{probs['paga']:.2%}" if isinstance(probs['paga'], (float, np.float_)) else probs['paga'],
                     'Prob. Não Pagar': f"{probs['nao_paga']:.2%}" if isinstance(probs['nao_paga'], (float, np.float_)) else probs['nao_paga']
                 })
             details_df = pd.DataFrame(details_list)
             st.dataframe(details_df, use_container_width=True)

    else:
        st.info("Aguardando dados do cliente e clique no botão 'Analisar Risco' na barra lateral.")


# --- disclaimer
st.sidebar.markdown("---")
st.sidebar.caption("Nota: Este é um simulador para fins de demonstração. Os resultados são baseados em modelos treinados e não representam uma análise de crédito real completa.")

# --- contatos
st.sidebar.markdown("---") 
st.sidebar.markdown("**Desenvolvido por:** Wellington M Santos")
st.sidebar.markdown("[LinkedIn](https://www.linkedin.com/in/wellington-moreira-santos/) | [GitHub](https://github.com/esscova)")