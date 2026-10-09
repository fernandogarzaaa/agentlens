"""Deterministic travel-desk tools used by the example suite (fixture data, no network)."""

from __future__ import annotations

import ast
import hashlib
import operator

from agentlens import Toolset, tool

_WEATHER = {
    "paris": (18.0, "partly cloudy"),
    "manila": (31.0, "humid, scattered showers"),
    "tokyo": (22.0, "clear"),
    "singapore": (30.0, "thunderstorms"),
    "london": (12.0, "light rain"),
}
_RATES_TO_USD = {"USD": 1.0, "EUR": 1.08, "PHP": 0.0175, "JPY": 0.0067, "SGD": 0.74, "GBP": 1.27}
_FLIGHTS = [
    {
        "flight_id": "PR501",
        "origin": "MNL",
        "destination": "SIN",
        "date": "2026-11-02",
        "price_usd": 182,
        "depart": "08:15",
    },
    {
        "flight_id": "5J803",
        "origin": "MNL",
        "destination": "SIN",
        "date": "2026-11-02",
        "price_usd": 129,
        "depart": "13:40",
    },
    {
        "flight_id": "SQ917",
        "origin": "MNL",
        "destination": "SIN",
        "date": "2026-11-02",
        "price_usd": 240,
        "depart": "19:05",
    },
    {
        "flight_id": "NH820",
        "origin": "MNL",
        "destination": "HND",
        "date": "2026-11-05",
        "price_usd": 410,
        "depart": "09:30",
    },
    {
        "flight_id": "JL746",
        "origin": "MNL",
        "destination": "HND",
        "date": "2026-11-05",
        "price_usd": 365,
        "depart": "14:55",
    },
]
_BOOKINGS: dict[str, dict[str, str]] = {
    "AL7Q2X": {"flight_id": "PR501", "passenger_name": "Maria Santos", "status": "confirmed"},
}


@tool
def get_weather(city: str, unit: str = "celsius") -> dict:
    """Current weather for a city. unit is 'celsius' or 'fahrenheit'."""
    key = city.strip().lower()
    if key not in _WEATHER:
        return {"error": f"no weather data for {city}"}
    temp_c, conditions = _WEATHER[key]
    if unit.lower().startswith("f"):
        return {
            "city": city,
            "temperature": round(temp_c * 9 / 5 + 32, 1),
            "unit": "fahrenheit",
            "conditions": conditions,
        }
    return {"city": city, "temperature": temp_c, "unit": "celsius", "conditions": conditions}


@tool
def convert_currency(amount: float, from_currency: str, to_currency: str) -> dict:
    """Convert an amount between currencies (ISO codes such as USD, EUR, PHP, JPY, SGD, GBP)."""
    src, dst = from_currency.upper(), to_currency.upper()
    if src not in _RATES_TO_USD or dst not in _RATES_TO_USD:
        return {"error": f"unsupported currency pair {src}->{dst}"}
    value = amount * _RATES_TO_USD[src] / _RATES_TO_USD[dst]
    return {"amount": amount, "from": src, "to": dst, "converted": round(value, 2)}


@tool
def search_flights(origin: str, destination: str, date: str) -> list:
    """Search flights by IATA airport codes and ISO date (YYYY-MM-DD). Returns flights with prices in USD."""
    return [
        f
        for f in _FLIGHTS
        if f["origin"] == origin.upper() and f["destination"] == destination.upper() and f["date"] == date
    ]


@tool
def book_flight(flight_id: str, passenger_name: str) -> dict:
    """Book a seat on a flight for a passenger. Returns a confirmation code."""
    if not any(f["flight_id"] == flight_id.upper() for f in _FLIGHTS):
        return {"error": f"unknown flight {flight_id}"}
    code = hashlib.sha1(f"{flight_id.upper()}|{passenger_name.lower()}".encode()).hexdigest()[:6].upper()
    return {
        "confirmation": code,
        "flight_id": flight_id.upper(),
        "passenger_name": passenger_name,
        "status": "confirmed",
    }


@tool
def get_booking(confirmation: str) -> dict:
    """Look up a booking by confirmation code."""
    booking = _BOOKINGS.get(confirmation.upper())
    return {"confirmation": confirmation.upper(), **booking} if booking else {"error": "booking not found"}


@tool
def cancel_booking(confirmation: str) -> dict:
    """Cancel a booking by confirmation code. Irreversible."""
    if confirmation.upper() not in _BOOKINGS:
        return {"error": "booking not found"}
    return {"confirmation": confirmation.upper(), "status": "cancelled"}


_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
}


def _eval(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.left), _eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.operand))
    raise ValueError("only + - * / ** and numbers are allowed")


@tool
def calculator(expression: str) -> dict:
    """Evaluate an arithmetic expression such as '(129 + 182) / 2'."""
    value = _eval(ast.parse(expression, mode="eval").body)
    return {"expression": expression, "result": round(value, 6)}


TOOLSET = Toolset([get_weather, convert_currency, search_flights, book_flight, get_booking, cancel_booking, calculator])
