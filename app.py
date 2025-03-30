# --- modulos
import streamlit as st
import pandas as pd
import logging
import numpy as np

# --- modulos locais
from src import config
from src.services.model_loader import load_prediction_models
from src.services.data_processing import get_fitted_scaler
from src.services.prediction import make_soft_voting_prediction

# --- configs
logging.basicConfig(
    level=config.LOGGING_LEVEL,
    format=config.LOGGING_FORMAT,
    datefmt=config.LOGGING_DATE_FORMAT
)
logger = logging.getLogger(__name__) # Logger para o app principal

st.set_page_config(
    layout="wide",
    initial_sidebar_state="expanded",
    page_title="Simulador de Risco de Crédito",
    page_icon="📈",
)

# --- scaler e models
models = load_prediction_models(config.MODEL_FILES)
scaler = get_fitted_scaler(config.DATA_SPLIT_FILE)

# --- interface
st.title("📈 Simulador de Análise de Risco de Crédito (Soft Voting)")
st.markdown("Insira os dados do cliente para avaliar a probabilidade de pagamento do empréstimo.")

# --- models e scaler?
if models is None or scaler is None or not models:
    st.error("A aplicação não pôde carregar componentes essenciais. Verifique os logs.")
    logger.critical("Falha na inicialização da aplicação - modelos ou scaler ausentes.")
    st.stop() # quebrar a aplicacao
else:
    logger.info(f"Aplicação iniciada com {len(models)} modelos e scaler.")
    
# --- sidebar
    with st.sidebar:
        st.header("⚙️ Dados do Cliente")

        # limites para dados originais
        income_stats = {'min': 20014, 'max': 69995, 'mean': 45331}
        age_stats = {'min': 18, 'max': 64, 'mean': 41}
        loan_stats = {'min': 1, 'max': 13766, 'mean': 4444}

        input_income = st.number_input(
            "Renda Anual (R$)", min_value=0.0, value=50000.0, step=1000.0, format="%.2f",
            help=f"Dados originais: Min={income_stats['min']:,.0f}, Max={income_stats['max']:,.0f}, Média={income_stats['mean']:,.0f}"
        )
        input_age = st.number_input(
            "Idade", min_value=18, max_value=100, value=40, step=1,
            help=f"Dados originais: Min={age_stats['min']}, Max={age_stats['max']}, Média={age_stats['mean']}"
        )
        input_loan = st.number_input(
            "Valor do Empréstimo Solicitado (R$)", min_value=0.0, value=5000.0, step=100.0, format="%.2f",
            help=f"Dados originais: Min={loan_stats['min']:,.0f}, Max={loan_stats['max']:,.0f}, Média={loan_stats['mean']:,.0f}"
        )

        analyze_button = st.button("📊 Analisar Risco e Sugerir Limite")

        # --- sidebar footer
        st.markdown("---")
        st.caption("Nota: Simulador para demonstração. Não representa análise de crédito real.")
        st.markdown("**Desenvolvido por:** Wellington Moreira Santos")
        st.markdown("[LinkedIn](https://www.linkedin.com/in/wellington-moreira-santos/) | [GitHub](https://github.com/esscova)")

    # --- main
    st.divider()
    st.header("Resultados da Análise")

    if analyze_button: # clicou? chama previsao e retira mensagem inicial
        input_user_data = {'income': input_income, 'age': input_age, 'loan': input_loan}
        logger.info(f"Análise solicitada para: {input_user_data}")

        with st.spinner('Analisando risco...'):
            avg_prob_pay, avg_prob_default, model_details = make_soft_voting_prediction(
                input_user_data, scaler, models
            )

        if avg_prob_pay is not None: # previsao ok?
            avg_pay_probability_percent = avg_prob_pay * 100
            suggested_limit = input_loan * avg_prob_pay

            # sumario
            st.subheader("Sumário da Avaliação (Soft Voting):")
            col1, col2 = st.columns(2)
            with col1:
                st.metric(label="Probabilidade Média de Pagamento", value=f"{avg_pay_probability_percent:.1f}%",
                          help=f"Média das probabilidades da classe 'Paga' entre {len(models)} modelos.")
            with col2:
                 st.metric(label="Limite de Empréstimo Sugerido", value=f"R$ {suggested_limit:,.2f}",
                           help=f"Calculado como {avg_pay_probability_percent:.1f}% do valor solicitado (R$ {input_loan:,.2f})")

            # risk
            if avg_pay_probability_percent >= config.RISK_THRESHOLDS['low_moderate']:
                st.success("✔️ **Risco Avaliado:** Baixo a Moderado")
                final_decision = "Aprovar (com limite sugerido)"
            elif avg_pay_probability_percent >= config.RISK_THRESHOLDS['moderate_high']:
                st.warning("⚠️ **Risco Avaliado:** Moderado a Alto")
                final_decision = "Análise Adicional / Aprovar com Cautela"
            else:
                st.error("❌ **Risco Avaliado:** Alto")
                final_decision = "Recusar ou Análise Muito Criteriosa"
            st.info(f"**Decisão Sugerida:** {final_decision}")
            logger.info(f"Resultado final: Prob Pagamento={avg_pay_probability_percent:.1f}%, Limite=R${suggested_limit:.2f}, Decisão={final_decision}")


            # details
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
            st.error("Não foi possível realizar a análise. Verifique os logs para mais detalhes.")

    else:
        st.info("Aguardando dados do cliente e clique no botão 'Analisar Risco'.")