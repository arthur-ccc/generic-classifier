import numpy as np
import pandas as pd


def generate_mock_dataset(n_samples=1000, random_state=42):
    np.random.seed(random_state)

    # Gerar features sintéticas
    idade = np.random.randint(30, 80, n_samples)
    sexo = np.random.choice([0, 1], n_samples)  # 0 = F, 1 = M
    pressao = np.random.randint(100, 180, n_samples)
    colesterol = np.random.randint(150, 300, n_samples)
    batimento_max = np.random.randint(60, 200, n_samples)
    dor_peito = np.random.choice([0, 1], n_samples)
    dor_exercicio = np.random.choice([0, 1], n_samples)
    glicose = np.random.randint(70, 200, n_samples)
    historico_familiar = np.random.choice([0, 1], n_samples)
    fuma = np.random.choice([0, 1], n_samples)

    # Criar DataFrame
    df = pd.DataFrame({
        "idade": idade,
        "sexo": sexo,
        "pressao": pressao,
        "colesterol": colesterol,
        "batimento_max": batimento_max,
        "dor_peito": dor_peito,
        "dor_exercicio": dor_exercicio,
        "glicose": glicose,
        "historico_familiar": historico_familiar,
        "fuma": fuma
    })

    # Gerar label com alguma correlação simples
    score = (idade*0.03 + sexo*0.1 + pressao*0.02 + colesterol*0.015 +
             dor_peito*0.3 + dor_exercicio*0.2 + historico_familiar*0.25 +
             fuma*0.2 + np.random.normal(0, 0.1, n_samples))
    df["doenca_cardiaca"] = (score > np.percentile(score, 60)).astype(int)

    return df
