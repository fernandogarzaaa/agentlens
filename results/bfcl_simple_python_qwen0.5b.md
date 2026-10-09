# agentlens eval: bfcl-v4-simple_python

Agent **qwen0.5b**, model `qwen2.5-0.5b-instruct-q4_k_m`, 100 cases, 2026-10-09T17:48:04+00:00.

| Metric | Value |
|---|---|
| Cases | 100 |
| Task success | 82.0% |
| Tool precision | 1.000 |
| Tool recall | 1.000 |
| Tool F1 (macro) | 1.000 |
| Tool F1 (micro) | 1.000 |
| Args exact match | 82.0% |
| Args partial match | 92.5% |
| Trajectory match | 100.0% |
| Constraints pass | 100.0% |
| Final-answer checks pass | 100.0% |
| Error rate | 0.0% |
| Steps / task | 1.00 |
| Tokens / task | 306 |
| $ / task | $0.000000 |
| Total $ | $0.000000 |
| Latency p50 (s) | 17.44 |
| Latency p95 (s) | 25.12 |

> Some models are missing from the pricing table: their cost is counted as unknown ($0 here).

## Failed cases (18/100)

| Case | Why | Expected | Actual |
|---|---|---|---|
| simple_python_5 | args 0.00 | solve_quadratic({"a": 3, "b": -11, "c": -4, "root_type": "all"}) | solve_quadratic({"a": 3, "b": -11, "c": -4}) |
| simple_python_7 | args 0.00 | calculate_circumference({"radius": 4, "unit": {"$any": ["inches", "in"]}}) | calculate_circumference({"radius": 4, "unit": "cm"}) |
| simple_python_13 | args 0.00 | calculate_area_under_curve({"function": {"$any": ["x**2", "lambda x: x**2", "y=x**2"]}, "interval": [1.0, 3.0], "method": {"$any": ["$absent", "trapezoidal"]}}) | calculate_area_under_curve({"function": "x^2", "interval": [1, 3], "method": "trapezoidal"}) |
| simple_python_14 | args 0.00 | calculate_derivative({"function": {"$any": ["3x**2 + 2x - 1", "lambda x: 3x**2 + 2x - 1"]}, "x_value": {"$any": ["$absent", 0.0]}}) | calculate_derivative({"function": "3x^2 + 2x - 1", "x_value": 0.0}) |
| simple_python_15 | args 0.00 | integrate({"function": {"$any": ["x**3", "lambda x: x**3"]}, "start_x": -2, "end_x": 3, "method": "simpson"}) | integrate({"function": "x^3", "start_x": -2, "end_x": 3, "method": "simpson"}) |
| simple_python_16 | args 0.00 | calculus_derivative({"function": {"$any": ["2x**2", "lambda x: 2x**2"]}, "value": 1, "function_variable": {"$any": ["x", "$absent"]}}) | calculus_derivative({"function": "2x^2", "value": 1}) |
| simple_python_17 | args 0.00 | get_prime_factors({"number": 450, "formatted": {"$any": [true, "$absent"]}}) | get_prime_factors({"number": 450, "formatted": false}) |
| simple_python_35 | args 0.00 | vegan_restaurant_find_nearby({"location": "New York, NY", "operating_hours": 23}) | vegan_restaurant_find_nearby({"location": "New York, NY", "operating_hours": 11}) |
| simple_python_42 | args 0.00 | calculate_resonant_frequency({"inductance": 0.05, "capacitance": 0.0001, "round_off": {"$any": ["$absent", 2]}}) | calculate_resonant_frequency({"inductance": 50.0, "capacitance": 100.0}) |
| simple_python_56 | args 0.00 | cellbio_get_proteins({"cell_compartment": "plasma membrane", "include_description": {"$any": ["$absent", true, false]}}) | cellbio_get_proteins({"cell_compartment": "plasma", "include_description": true}) |
| simple_python_61 | args 0.00 | diabetes_prediction({"weight": 150, "height": 70, "activity_level": "lightly active"}) | diabetes_prediction({"weight": 150, "height": 5, "activity_level": "lightly active"}) |
| simple_python_65 | args 0.00 | calculate_density({"country": "Brazil", "year": "2022", "population": 213000000, "land_area": 8500000}) | calculate_density({"country": "Brazil", "year": "2022", "population": 213, "land_area": 850000}) |
| simple_python_69 | args 0.00 | ecology_get_turtle_population({"location": "Mississippi river", "year": 2020, "species": true}) | ecology_get_turtle_population({"location": "Mississippi river", "year": 2020, "species": false}) |
| simple_python_83 | args 0.00 | calculate_distance({"coord1": [33.4484, -112.074], "coord2": [34.0522, -118.2437], "unit": "miles"}) | calculate_distance({"coord1": [33.4484, 112.074], "coord2": [34.0522, 118.2437], "unit": "miles"}) |
| simple_python_89 | args 0.00 | db_fetch_records({"database_name": "StudentDB", "table_name": "students", "conditions": {"department": "Science", "school": {"$any": ["Bluebird High School", "Bluebird HS"]}}, "fetch_limit": {"$any": ["$absent", 0]}}) | db_fetch_records({"database_name": "StudentDB", "table_name": "students", "conditions": {"school": "Bluebird High School"}, "fetch_limit": 0}) |
| simple_python_91 | args 0.00 | get_restaurant({"cuisine": "sushi", "location": "Boston", "condition": {"$any": ["open on Sundays", "opens on Sundays"]}}) | get_restaurant({"cuisine": "sushi", "location": "Boston", "condition": "operating days, Sunday"}) |
| simple_python_96 | args 0.00 | database_query({"table": "user", "conditions": [{"field": "age", "operation": ">", "value": "25"}, {"field": "job", "operation": "=", "value": "engineer"}]}) | database_query({"table": "user", "conditions": [{"field": "age", "operation": ">", "value": "25"}, {"field": "job", "operation": ">", "value": "engineer"}]}) |
| simple_python_99 | args 0.00 | plot_sine_wave({"start_range": 0.0, "end_range": 6.2832, "frequency": 5, "amplitude": {"$any": [1, "$absent"]}, "phase_shift": {"$any": [0, "$absent"]}}) | plot_sine_wave({"start_range": 0, "end_range": 2.0, "frequency": 5, "phase_shift": 0}) |
