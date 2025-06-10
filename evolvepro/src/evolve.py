import os
import random
import time
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple

from evolvepro.src.data import load_dms_data
from evolvepro.src.utils import pca_embeddings
from evolvepro.src.model import first_round, top_layer

def directed_evolution_simulation(
    labels: pd.DataFrame,
    embeddings: pd.DataFrame,
    num_simulations: int,
    num_iterations: int,
    num_mutants_per_round: int = 10,
    measured_var: str = 'activity',
    regression_type: str = 'ridge',
    learning_strategy: str = 'topn',
    top_n: int = None,
    final_round: int = 10,
    first_round_strategy: str = 'random',
    embedding_type: str = None,
    explicit_variants: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Запускает имитацию направленной эволюции.
    Возвращает DataFrame со всеми раундами / симуляциями.
    """
    output_list = []

    for i in range(1, num_simulations + 1):

        iteration_old = None
        # Списки, куда будем записывать значения метрик
        num_mutants_per_round_list = []
        first_round_strategy_list = []
        measured_var_list = []
        learning_strategy_list = []
        regression_type_list = []
        embedding_type_list = []
        simulation_list = []
        round_list = []
        test_error_list = []
        train_error_list = []
        train_r_squared_list = []
        test_r_squared_list = []
        alpha_list = []
        median_activity_scaled_list = []
        top_variant_list = []
        top_final_round_variants_list = []
        top_activity_scaled_list = []
        spearman_corr_list = []
        activity_binary_percentage_list = []

        # Для логирования, какие варианты берутся на каждом раунде
        this_round_variants_list = []
        next_round_variants_list = []

        j = 0
        while j <= num_iterations:

            if j == 0:
                # Первый раунд
                labels_new, iteration_new, this_round_variants = first_round(
                    labels,
                    embeddings,
                    explicit_variants=explicit_variants,
                    num_mutants_per_round=num_mutants_per_round,
                    first_round_strategy=first_round_strategy,
                    embedding_type=embedding_type,
                    random_seed=i
                )

                # Записываем «пустые» метрики, ведь это ещё до тренировки
                num_mutants_per_round_list.append(num_mutants_per_round)
                first_round_strategy_list.append(first_round_strategy)
                measured_var_list.append(measured_var)
                learning_strategy_list.append(learning_strategy)
                regression_type_list.append(regression_type)
                embedding_type_list.append(embedding_type)
                simulation_list.append(i)
                round_list.append(j)

                test_error_list.append("None")
                train_error_list.append("None")
                train_r_squared_list.append("None")
                test_r_squared_list.append("None")
                alpha_list.append("None")
                median_activity_scaled_list.append("None")
                top_activity_scaled_list.append("None")
                top_variant_list.append("None")
                top_final_round_variants_list.append("None")
                activity_binary_percentage_list.append("None")
                spearman_corr_list.append("None")

                this_round_variants_list.append("None")
                next_round_variants_list.append(",".join(this_round_variants))

                j += 1

            else:
                # Остальные раунды
                iteration_old = iteration_new
                print("iterations considered:", iteration_old)

                train_error, test_error, train_r_squared, test_r_squared, alpha, median_activity_scaled, top_activity_scaled, top_variant, top_final_round_variants, activity_binary_percentage, spearman_corr, df_test_new, this_round_variants = top_layer(
                    iter_train=iteration_old['iteration'].unique().tolist(),
                    iter_test=None,
                    embeddings_pd=embeddings,
                    labels_pd=labels_new,
                    measured_var=measured_var,
                    regression_type=regression_type,
                    top_n=top_n,
                    final_round=final_round
                )

                if learning_strategy == 'dist':
                    iteration_new_ids = df_test_new.sort_values(by='dist_metric', ascending=False).head(num_mutants_per_round).variant
                elif learning_strategy == 'random':
                    iteration_new_ids = random.sample(list(df_test_new.variant), num_mutants_per_round)
                elif learning_strategy == 'topn2bottomn2':
                    top_half = df_test_new.sort_values(by='y_pred', ascending=False).head(int(num_mutants_per_round/2)).variant
                    bottom_half = df_test_new.sort_values(by='y_pred', ascending=False).tail(int(num_mutants_per_round/2)).variant
                    iteration_new_ids = pd.concat([top_half, bottom_half])
                elif learning_strategy == 'topn':
                    iteration_new_ids = df_test_new.sort_values(by='y_pred', ascending=False).head(num_mutants_per_round).variant

                iteration_new = pd.DataFrame({'variant': iteration_new_ids, 'iteration': j})
                iteration_new = pd.concat([iteration_new, iteration_old], ignore_index=True)
                labels_new = pd.merge(labels, iteration_new, on='variant', how='left')

                num_mutants_per_round_list.append(num_mutants_per_round)
                first_round_strategy_list.append(first_round_strategy)
                measured_var_list.append(measured_var)
                learning_strategy_list.append(learning_strategy)
                regression_type_list.append(regression_type)
                embedding_type_list.append(embedding_type)
                simulation_list.append(i)
                round_list.append(j)

                test_error_list.append(test_error)
                train_error_list.append(train_error)
                train_r_squared_list.append(train_r_squared)
                test_r_squared_list.append(test_r_squared)
                alpha_list.append(alpha)
                median_activity_scaled_list.append(median_activity_scaled)
                top_activity_scaled_list.append(top_activity_scaled)
                top_variant_list.append(top_variant)
                top_final_round_variants_list.append(top_final_round_variants)
                activity_binary_percentage_list.append(activity_binary_percentage)
                spearman_corr_list.append(spearman_corr)

                this_round_variants_list.append(",".join(iteration_old.variant))
                next_round_variants_list.append(",".join(iteration_new_ids))

                j += 1

            # Собираем метрики в DataFrame
            df_metrics = pd.DataFrame({
                'simulation_num': simulation_list,
                'round_num': round_list,
                'num_mutants_per_round': num_mutants_per_round_list,
                'first_round_strategy': first_round_strategy_list,
                'measured_var': measured_var_list,
                'learning_strategy': learning_strategy_list,
                'regression_type': regression_type_list,
                'embedding_type': embedding_type_list,
                'test_error': test_error_list,
                'train_error': train_error_list,
                'train_r_squared': train_r_squared_list,
                'test_r_squared': test_r_squared_list,
                'alpha': alpha_list,
                'spearman_corr': spearman_corr_list,
                'median_activity_scaled': median_activity_scaled_list,
                'top_activity_scaled': top_activity_scaled_list,
                'activity_binary_percentage': activity_binary_percentage_list,
                'top_variant': top_variant_list,
                'top_final_round_variants': top_final_round_variants_list,
                'this_round_variants': this_round_variants_list,
                'next_round_variants': next_round_variants_list
            })
        output_list.append(df_metrics)

    output_table = pd.concat(output_list)
    return output_table


def grid_search(
    dataset_name: str,
    experiment_name: str,
    model_name: str,
    embeddings_path: str,
    labels_path: str,
    num_simulations: int,
    num_iterations: List[int],
    measured_var: List[str],
    learning_strategies: List[str],
    num_mutants_per_round: List[int],
    num_final_round_mutants: int,
    first_round_strategies: List[str],
    embedding_types: List[str],
    pca_components: List[int],
    regression_types: List[str],
    embeddings_file_type: str,
    output_dir: str,
    embeddings_type_pt: Optional[str] = None,
    explicit_variants: Optional[List[str]] = None
) -> None:
    """
    Проводит перебор параметров (grid search) для задачи directed evolution.
    Сохраняет итоговый CSV, добавляя dataset_name и experiment_name в колонки,
    чтобы read_dms_data мог затем сгруппировать по "dataset" и "experiment".
    """
    # 1) Загрузка исходного датасета (зависит от load_dms_data)
    embeddings, labels = load_dms_data(
        dataset_name,
        model_name,
        embeddings_path,
        labels_path,
        embeddings_file_type,
        embeddings_type_pt
    )
    if embeddings is None or labels is None:
        print("Failed to load data. Exiting.")
        return

    # 2) PCA-эмбеддинги (при необходимости)
    if pca_components is not None:
        embeddings_pca = {
            f'embeddings_pca_{n}': pca_embeddings(embeddings, n_components=n)
            for n in pca_components
        }
        embeddings_list = {
            'embeddings': embeddings,
            **embeddings_pca
        }
    else:
        embeddings_list = {'embeddings': embeddings}

    total_combinations = 0
    for strategy in learning_strategies:
        for var in measured_var:
            for iterations in num_iterations:
                for mutants_per_round in num_mutants_per_round:
                    for embtype in embedding_types:
                        for regrtype in regression_types:
                            for fr_strategy in first_round_strategies:
                                total_combinations += 1
    print(f"Total combinations: {total_combinations}")

    output_list = []
    combination_count = 0
    start_time = time.time()

    for strategy in learning_strategies:
        for var in measured_var:
            for iterations in num_iterations:
                for mutants_per_round in num_mutants_per_round:
                    for embtype in embedding_types:
                        for regrtype in regression_types:
                            for fr_strategy in first_round_strategies:
                                combination_count += 1

                                # Запуск directed_evolution_simulation для текущей комбинации
                                output_table = directed_evolution_simulation(
                                    labels=labels,
                                    embeddings=embeddings_list[embtype],
                                    num_simulations=num_simulations,
                                    num_iterations=iterations,
                                    num_mutants_per_round=mutants_per_round,
                                    measured_var=var,
                                    regression_type=regrtype,
                                    learning_strategy=strategy,
                                    final_round=num_final_round_mutants,
                                    first_round_strategy=fr_strategy,
                                    embedding_type=embtype,
                                    explicit_variants=explicit_variants
                                )
                                print(
                                    f"Progress: {combination_count}/{total_combinations} "
                                    f"({(combination_count/total_combinations)*100:.2f}%)"
                                )
                                output_list.append(output_table)

    execution_time = time.time() - start_time
    print(f"Total execution time: {execution_time:.2f} seconds")


    df_results = pd.concat(output_list, ignore_index=True)

    df_results["dataset"] = dataset_name
    df_results["experiment"] = experiment_name

    os.makedirs(output_dir, exist_ok=True)
    if embeddings_type_pt is None:
        out_csv = f"{output_dir}/{dataset_name}_{model_name}_{experiment_name}.csv"
    else:
        out_csv = f"{output_dir}/{dataset_name}_{model_name}_{experiment_name}_{embeddings_type_pt}.csv"

    df_results.to_csv(out_csv, index=False)
    print(f"Saved final CSV to: {out_csv}")
