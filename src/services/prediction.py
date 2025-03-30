# --- modulos
import pandas as pd
import numpy as np
import logging
from src import config

logger = logging.getLogger(__name__)

def make_soft_voting_prediction(input_data: dict, scaler, models: dict) -> tuple:
    """
    Realiza a previsão usando Soft Voting nos modelos fornecidos.

    Args:
        input_data (dict): Dicionário com os dados de entrada {'income': val, 'age': val, 'loan': val}.
        scaler: Objeto StandardScaler ajustado.
        models (dict): Dicionário com os modelos carregados.

    Returns:
        tuple: (avg_prob_pay, avg_prob_default, model_details) ou (None, None, None) em caso de erro.
    """
    logger.info("Iniciando processo de previsão (Soft Voting)...")
    
    if not models:
        logger.error("Nenhum modelo válido fornecido para previsão.")
        return None, None, {}

    try:
        # 1. df na ordem correta
        input_df = pd.DataFrame([input_data])[config.FEATURE_ORDER] 
        logger.debug(f"DataFrame de entrada ordenado: {input_df.to_dict()}")

        # 2. escalonar
        input_scaled = scaler.transform(input_df.values) 
        logger.debug(f"Dados de entrada escalonados: {input_scaled}")

        # 3. probabilidades
        all_probabilities = []
        model_details = {}

        for name, model in models.items():
            try:
                proba = model.predict_proba(input_scaled)[0] # [[p0, p1]] -> [p0, p1]
                all_probabilities.append(proba)
                model_details[name] = {'paga': proba[0], 'nao_paga': proba[1]}
                logger.debug(f"Modelo '{name}' probs: Paga={proba[0]:.4f}, Não Paga={proba[1]:.4f}")
            except Exception as e:
                logger.warning(f"Erro ao obter probabilidades do modelo '{name}': {e}", exc_info=True)
                model_details[name] = {'paga': 'Erro', 'nao_paga': 'Erro'}

        # 4. calc avg
        if not all_probabilities:
            logger.error("Nenhum modelo retornou probabilidades válidas.")
            return None, None, model_details

        avg_proba = np.mean(all_probabilities, axis=0)
        avg_prob_pay = avg_proba[0]
        avg_prob_default = avg_proba[1]
        logger.info(f"Probabilidade média final: Paga={avg_prob_pay:.4f}, Não Paga={avg_prob_default:.4f}")

        return avg_prob_pay, avg_prob_default, model_details

    except Exception as e:
        logger.error(f"Erro inesperado durante a previsão: {e}", exc_info=True)
        return None, None, {}