# agentlens eval: bfcl-v4-irrelevance

Agent **qwen0.5b**, model `qwen2.5-0.5b-instruct-q4_k_m`, 50 cases, 2026-10-09T17:49:33+00:00.

| Metric | Value |
|---|---|
| Cases | 50 |
| Task success | 34.0% |
| Tool precision | 0.340 |
| Tool recall | 0.340 |
| Tool F1 (macro) | 0.340 |
| Tool F1 (micro) | 0.000 |
| Args exact match | 100.0% |
| Args partial match | 100.0% |
| Trajectory match | 34.0% |
| Constraints pass | 100.0% |
| Final-answer checks pass | 100.0% |
| Error rate | 0.0% |
| Steps / task | 1.00 |
| Tokens / task | 343 |
| $ / task | $0.000000 |
| Total $ | $0.000000 |
| Latency p50 (s) | 16.33 |
| Latency p95 (s) | 79.00 |

> Some models are missing from the pricing table: their cost is counted as unknown ($0 here).

## Failed cases (33/50)

| Case | Why | Expected | Actual |
|---|---|---|---|
| irrelevance_0 | trajectory (strict) | (none) | determine_body_mass_index({"weight": 0, "height": 5}) |
| irrelevance_1 | trajectory (strict) | (none) | math_sum({"numbers": [1, 2, 3], "decimal_places": 2}) |
| irrelevance_2 | trajectory (strict) | (none) | distance_calculator_calculate({"coordinate_1": [0, 0], "coordinate_2": [1, 0]}) |
| irrelevance_3 | trajectory (strict) | (none) | find_critical_points({"function": "y = 3x + 2", "variable": "x"}) |
| irrelevance_4 | trajectory (strict) | (none) | find_roots({"a": 1, "b": 1, "c": 0}) |
| irrelevance_5 | trajectory (strict) | (none) | solve_quadratic_equation({"a": 5, "b": 4, "c": 0}) |
| irrelevance_6 | trajectory (strict) | (none) | draw_circle({"radius": 5, "unit": "m"}) |
| irrelevance_8 | trajectory (strict) | (none) | str_to_int({"value": "x^3"}) |
| irrelevance_9 | trajectory (strict) | (none) | CalculateTax({"annual_income": 1000000, "tax_rate": 0.2, "other_deductions": 50000}) |
| irrelevance_10 | trajectory (strict) | (none) | calculus_compute_definite_integral({"function": "2x", "interval": [1], "num_of_partitions": 1000}) |
| irrelevance_11 | trajectory (strict) | (none) | get_closest_prime({"number": 30, "skip": 0}) |
| irrelevance_13 | trajectory (strict) | (none) | calculate_compound_interest({"principal_amount": 100, "annual_interest_rate": 0.05, "compounding_periods_per_year": 12, "years": 10}) |
| irrelevance_14 | trajectory (strict) | (none) | calculate_maximum_height({"initial_velocity": 5, "gravity": 9.8}) |
| irrelevance_19 | trajectory (strict) | (none) | investment_calculator_calculate_return({"initial_investment": 1000, "annual_rate": 0.05, "years": 10}) |
| irrelevance_20 | trajectory (strict) | (none) | currency_converter({"base_currency": "USD", "target_currency": "EUR", "amount": 1000000}) |
| irrelevance_21 | trajectory (strict) | (none) | calculate_wave_amplitude({"max_electric_field_strength": 2, "c": 3000000000000000, "wave_frequency": 1}) |
| irrelevance_22 | trajectory (strict) | (none) | magnetic_field_intensity({"current": 1, "distance": 1, "permeability": 4e-07}) |
| irrelevance_25 | trajectory (strict) | (none) | thermodynamics_calculate_boiling_point({"substance": "water", "pressure": 10, "unit": "kPa"}) |
| irrelevance_26 | trajectory (strict) | (none) | thermodynamics_calc_gas_pressure({"volume": 2, "initial_temperature": 25, "final_temperature": 100}) |
| irrelevance_27 | trajectory (strict) | (none) | calculate_heat({"mass": 3, "specific_heat": 4.184, "change_in_temp": 4}) |
| irrelevance_29 | trajectory (strict) | (none) | get_cell_function({"cell_part": "mitochondria", "detail_level": "basic"}) |
| irrelevance_30 | trajectory (strict) | (none) | bloodcell_classification({"cell_shape": "multiple nuclei", "cell_size": "medium"}) |
| irrelevance_32 | trajectory (strict) | (none) | cellBiology_getCellType({"nucleus_count": 0, "organism_type": "blood cells"}) |
| irrelevance_33 | trajectory (strict) | (none) | identify_species({"sequence": "ATCG"}) |
| irrelevance_34 | trajectory (strict) | (none) | genetics_get_variant_frequency({"variant_id": "dominant_genetic_trait_of_Lion", "population": "Lion"}) |

...and 8 more (see the JSON results).
