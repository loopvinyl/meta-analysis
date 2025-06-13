def run_statistical_analysis_and_plot(df, dependent_var_name, group_var_name):
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from statsmodels.stats.multicomp import pairwise_tukeyhsd
    from statsmodels.stats.multicomp import MultiComparison

    # Preparar os dados para o gráfico
    plot_df = df[[group_var_name, dependent_var_name]].dropna()

    # Fazer análise estatística: Tukey HSD
    mc = MultiComparison(plot_df[dependent_var_name], plot_df[group_var_name])
    tukey_result = mc.tukeyhsd()

    # Gerar letras para grupos estatísticos (significância)
    cld_letters = {}
    groups = mc.groupsunique

    # Criar dicionário com letras indicando grupos diferentes
    # Exemplo simples: se dois grupos não são diferentes, recebem mesma letra
    # Aqui simplificado; ideal usar statsmodels ou outra lib para letras
    # Mas vamos usar o resultado do tukey para gerar letras:
    from statsmodels.sandbox.stats.multicomp import get_tukey_pvalue
    # Usar um método alternativo para gerar letras (exemplo prático):
    from statsmodels.stats.multicomp import tukeyhsd

    # Função auxiliar para letras (simplificada)
    def generate_letters(tukey_result):
        from collections import defaultdict
        letters = {}
        reject = tukey_result.reject
        groups = tukey_result.groupsunique
        # Criar grafo de conexões para grupos não rejeitados
        connected = defaultdict(set)
        for i, (g1, g2) in enumerate(tukey_result._multicomp.pairindices):
            if not reject[i]:
                connected[groups[g1]].add(groups[g2])
                connected[groups[g2]].add(groups[g1])
        # Atribuir letras
        # Aqui, simplificação extrema: cada grupo recebe uma letra, ou compartilha se conectado
        # Melhor usar scikit-posthocs ou outras libs para agrupamento real
        import string
        assigned = {}
        available_letters = list(string.ascii_lowercase)
        for g in groups:
            assigned[g] = None
        letter_index = 0
        for g in groups:
            if assigned[g] is None:
                assigned[g] = available_letters[letter_index]
                queue = [g]
                while queue:
                    current = queue.pop()
                    for neighbor in connected[current]:
                        if assigned[neighbor] is None:
                            assigned[neighbor] = assigned[g]
                            queue.append(neighbor)
                letter_index += 1
        return assigned

    cld_letters = generate_letters(tukey_result)

    # Criar DataFrame das letras
    letter_df = pd.DataFrame([
        {'group': k, 'letter': v} for k, v in cld_letters.items()
    ])

    # Renomear para casar com y_max e plot_df
    letter_df = letter_df.rename(columns={'group': group_var_name})

    # Calcular máximos para posicionar letras
    y_max = plot_df.groupby(group_var_name)[dependent_var_name].max().reset_index()

    # Merge letras com valores máximos
    letter_df = letter_df.merge(y_max, on=group_var_name)

    # Plotar boxplot com jitter
    plt.figure(figsize=(10, 6))
    ax = sns.boxplot(data=plot_df, x=group_var_name, y=dependent_var_name)
    sns.stripplot(data=plot_df, x=group_var_name, y=dependent_var_name,
                  color='black', size=5, jitter=True, ax=ax)

    # Adicionar letras no topo das caixas
    for _, row in letter_df.iterrows():
        ax.text(row[group_var_name], row[dependent_var_name] + 0.05 * plot_df[dependent_var_name].max(),
                row['letter'], horizontalalignment='center', color='red', fontsize=14)

    plt.title(f'Boxplot with Jitter and Statistical Letters for {dependent_var_name}')
    plt.tight_layout()
    plt.show()
