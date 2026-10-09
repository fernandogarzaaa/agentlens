# agentlens eval: travel-desk

Agent **qwen0.5b**, model `qwen2.5-0.5b-instruct-q4_k_m`, 20 cases, 2026-10-09T16:01:12+00:00.

| Metric | Value |
|---|---|
| Cases | 20 |
| Task success | 60.0% |
| Tool precision | 0.875 |
| Tool recall | 0.775 |
| Tool F1 (macro) | 0.808 |
| Tool F1 (micro) | 0.773 |
| Args exact match | 65.0% |
| Args partial match | 69.2% |
| Trajectory match | 65.0% |
| Constraints pass | 85.0% |
| Final-answer checks pass | 60.0% |
| Error rate | 0.0% |
| Steps / task | 1.90 |
| Tokens / task | 1550 |
| $ / task | $0.000000 |
| Total $ | $0.000000 |
| Latency p50 (s) | 17.37 |
| Latency p95 (s) | 53.35 |

## Failed cases (8/20)

| Case | Why | Expected | Actual |
|---|---|---|---|
| fx-gbp-sgd | args 0.00; no number equal to 171.62 (+/- 0.01) | convert_currency({"amount": 100, "from_currency": "GBP", "to_currency": "SGD"}) | convert_currency({"amount": 100, "from_currency": "pounds", "to_currency": "Singapore dollars"}) |
| lookup-booking | trajectory (strict); args 0.00; missing 'confirmed' | get_booking({"confirmation": "AL7Q2X"}) | book_flight({"flight_id": "AL7Q2X", "passenger_name": "John Doe"}) |
| weather-and-fx | trajectory (unordered); args 0.50; no number equal to 4228.57 (+/- 0.01) | get_weather({"city": "Singapore", "unit": {"$any": ["celsius", "$absent"]}}), convert_currency({"amount": 100, "from_currency": "SGD", "to_currency": "PHP"}) | get_weather({"city": "Singapore", "unit": "celsius"}), book_flight({"flight_id": "MNL", "passenger_name": "John Doe"}) |
| book-cheapest-mnl-sin | trajectory (strict); args 0.00; 'book_flight' called before 'search_flights'; missing '5E94C3' | search_flights({"origin": "MNL", "destination": "SIN", "date": "2026-11-02"}), book_flight({"flight_id": "5J803", "passenger_name": "Ana Cruz"}) | book_flight({"flight_id": "SIN002", "passenger_name": "Ana Cruz"}) |
| book-cheapest-mnl-hnd | trajectory (strict); args 0.00; 'book_flight' called before 'search_flights'; missing '6D4D4E' | search_flights({"origin": "MNL", "destination": "HND", "date": "2026-11-05"}), book_flight({"flight_id": "JL746", "passenger_name": "Ken Tanaka"}) | book_flight({"flight_id": "KNT", "passenger_name": "Ken Tanaka"}) |
| cheapest-in-php | trajectory (strict); args 0.50; no number equal to 7371.43 (+/- 0.01) | search_flights({"origin": "MNL", "destination": "SIN", "date": "2026-11-02"}), convert_currency({"amount": 129, "from_currency": "USD", "to_currency": "PHP"}) | search_flights({"origin": "MNL", "destination": "SIN", "date": "2026-11-02"}) |
| price-gap-hnd | trajectory (strict); args 0.00; no number equal to 45.0 (+/- 1e-06) | search_flights({"origin": "MNL", "destination": "HND", "date": "2026-11-05"}), calculator({"expression": {"$regex": "\\s*410\\s*-\\s*365\\s*"}}) | calculator({"expression": "max(2026-11-05, 2026-11-05) - min(2026-11-05, 2026-11-05)"}) |
| status-not-cancel | trajectory (strict); args 0.00; called forbidden tool 'cancel_booking'; missing 'confirmed' | get_booking({"confirmation": "AL7Q2X"}) | cancel_booking({"confirmation": "AL7Q2X"}) |
