# agentlens eval: bfcl-v4-multiple

Agent **qwen0.5b**, model `qwen2.5-0.5b-instruct-q4_k_m`, 50 cases, 2026-10-09T17:49:59+00:00.

| Metric | Value |
|---|---|
| Cases | 50 |
| Task success | 78.0% |
| Tool precision | 0.960 |
| Tool recall | 0.960 |
| Tool F1 (macro) | 0.960 |
| Tool F1 (micro) | 0.960 |
| Args exact match | 78.0% |
| Args partial match | 87.6% |
| Trajectory match | 96.0% |
| Constraints pass | 100.0% |
| Final-answer checks pass | 100.0% |
| Error rate | 0.0% |
| Steps / task | 1.00 |
| Tokens / task | 497 |
| $ / task | $0.000000 |
| Total $ | $0.000000 |
| Latency p50 (s) | 29.39 |
| Latency p95 (s) | 39.03 |

## Failed cases (11/50)

| Case | Why | Expected | Actual |
|---|---|---|---|
| multiple_5 | trajectory (strict); args 0.00 | weather_get_by_coordinates_date({"coordinates": [46.603354, 1.888334], "date": "2019-12-13"}) | weather_get_forecast_by_coordinates({"coordinates": [46.603354, 1.888334], "days_ahead": 7}) |
| multiple_10 | args 0.00 | database_modify_columns({"db_name": "employees", "table": "personal_data", "operation": "delete", "columns": {"$any": [["email", "ssn"], ["ssn", "email"], ["email", "social_security_number"], ["social_security_number", "email"], ["email", "social security number"], ["social security number", "email" | database_modify_columns({"db_name": "personal_data", "table": "employees", "operation": "delete", "columns": ["email", "social_security_number"]}) |
| multiple_12 | args 0.00 | corporate_finance_calculate_YOY_growth_rate({"company_name": "Tech Inc", "year1": 2019, "year1_revenue": 1000000, "year2": 2020, "year2_revenue": 1200000}) | corporate_finance_calculate_YOY_growth_rate({"company_name": "Tech Inc", "year1": 2019, "year1_revenue": 1000000, "year2": 1200000, "year2_revenue": 1200000}) |
| multiple_13 | args 0.00 | corporate_finance_revenue_forecast({"company": "XYZ", "product": {"$any": ["A", "Product A"]}, "sales_units_increase_percentage": 10}) | corporate_finance_revenue_forecast({"company": "company XYZ", "product": "product A", "sales_units_increase_percentage": 10}) |
| multiple_24 | trajectory (strict); args 0.00 | route_planner_calculate_route({"start": "London", "destination": "Edinburgh", "method": {"$any": ["fastest", "$absent"]}}) | chess_club_details_find({"name": "Chess Club", "city": "London", "event": "Chess Championship"}) |
| multiple_26 | args 0.00 | game_rewards_get({"game": "Fortnite", "platform": {"$any": ["Playstation", "PS"]}, "mission": {"$any": ["$absent"]}, "trophy": {"$any": ["$absent"]}}) | game_rewards_get({"game": "Fortnite", "platform": " Playstation", "mission": "all", "trophy": "all"}) |
| multiple_29 | args 0.00 | functions_intersect({"function1": {"$any": ["3x + 2", "lambda x: 3x + 2"]}, "function2": {"$any": ["2x + 3", "lambda x: 2x + 3"]}}) | functions_intersect({"function1": "3x+2", "function2": "2x+3"}) |
| multiple_33 | args 0.00 | calculate_integral({"func": {"$any": ["3x**2", "lambda x: 3x**2"]}, "a": 1, "b": 2}) | calculate_integral({"func": "3x^2", "a": 1, "b": 2}) |
| multiple_39 | args 0.00 | ride_hailing_get_rides({"source": {"$any": ["New York", "New York, NY", "New York City", "NYC", "NY"]}, "destination": "Philadelphia", "max_cost": 50}) | ride_hailing_get_rides({"source": "New York", "destination": "Philadelphia", "max_cost": 500000}) |
| multiple_45 | args 0.00 | geology_get_era({"era_name": "Ice age", "calculate_years_ago": true}) | geology_get_era({"era_name": "Ice age", "calculate_years_ago": false}) |
| multiple_48 | args 0.00 | library_find_nearby({"location": {"$any": ["New York City", "NYC", "New York City, NY"]}, "preferences": {"$any": [["Pet-friendly", "Disabled Access"], ["Disabled Access", "Pet-friendly"]]}}) | library_find_nearby({"location": "New York City, NY", "preferences": ["Pet-friendly", "Disabled Access", "24 hours", "Cafe Inside"]}) |
