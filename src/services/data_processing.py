# --- modulos
import joblib
import logging
from sklearn.preprocessing import StandardScaler
import streamlit as st 
import os

# --- configs
logger = logging.getLogger(__name__)

@st.cache_resource
def get_fitted_scaler(data_split_filepath: str) -> StandardScaler:
    """Carrega dados de treino e ajusta um StandardScaler."""
    logger.info("Iniciando carregamento de dados para ajuste do Scaler...")

    try:
        if not os.path.exists(data_split_filepath):
             logger.error(f"Arquivo de dados para o Scaler não encontrado: {data_split_filepath}")
             st.error(f"Erro Crítico: Arquivo '{data_split_filepath}' não encontrado.")
             return None

        X_train, _, _, _ = joblib.load(data_split_filepath)
        scaler = StandardScaler()
        scaler.fit(X_train)
        logger.info("StandardScaler ajustado com sucesso.")
        return scaler
    
    except Exception as e:
        logger.error(f"Exceção durante o carregamento/ajuste do Scaler: {e}", exc_info=True)
        st.error(f"Erro fatal ao preparar o pré-processador: {e}")
        return None