from __future__ import annotations

from fastapi import APIRouter, HTTPException

router = APIRouter()

_PRECAUTIONS: dict[int, dict] = {
    0: {
        "name": "No Damage",
        "color": "#008000",
        "precautions": [
            "Area appears structurally intact",
            "Continue normal monitoring",
            "Document current state for future comparison",
        ],
        "actions": [
            "Verify with ground-truth survey if possible",
            "Keep access routes clear for emergency vehicles",
        ],
        "evacuation": "No evacuation required",
    },
    1: {
        "name": "Minor Damage",
        "color": "#FFFF00",
        "precautions": [
            "Minor structural damage — approach with caution",
            "Check for broken glass, loose debris, and damaged utilities",
            "Do not enter buildings with visible foundation cracks",
        ],
        "actions": [
            "Conduct visual inspection of exterior",
            "Document damage with photographs",
            "Contact local authorities to report damage",
            "Arrange structural assessment before re-entry",
        ],
        "evacuation": "Precautionary evacuation recommended for at-risk individuals",
    },
    2: {
        "name": "Major Damage",
        "color": "#FF8C00",
        "precautions": [
            "Significant structural damage — do not enter buildings",
            "High risk of partial collapse",
            "Hazardous utilities (gas, electrical) may be exposed",
            "Risk of secondary hazards: fires, flooding",
        ],
        "actions": [
            "Immediate evacuation of all occupants",
            "Cordon off area to prevent unauthorized access",
            "Shut off utilities if safe to do so",
            "Contact emergency services and structural engineers",
            "Establish triage and rescue operations",
        ],
        "evacuation": "Mandatory evacuation — do not re-enter until cleared by engineers",
    },
    3: {
        "name": "Destroyed",
        "color": "#DC143C",
        "precautions": [
            "Total structural failure — extreme danger",
            "High risk of collapse, entrapment, and secondary disasters",
            "Hazardous materials may be present in debris",
            "Air quality may be compromised",
        ],
        "actions": [
            "Treat as active search-and-rescue zone",
            "Deploy heavy rescue teams immediately",
            "Establish command post outside affected radius",
            "Coordinate with NDRF/military for large-scale operations",
            "Set up emergency shelters for displaced residents",
            "Conduct survivor accounting and missing-person registration",
        ],
        "evacuation": "Full evacuation — long-term displacement expected",
    },
}


@router.get("/precautions")
def list_precautions():
    return [{"damage_class": k, **v} for k, v in _PRECAUTIONS.items()]


@router.get("/precautions/{damage_class}")
def get_precautions(damage_class: int):
    if damage_class not in _PRECAUTIONS:
        raise HTTPException(404, f"damage_class must be 0-3, got {damage_class}")
    return {"damage_class": damage_class, **_PRECAUTIONS[damage_class]}
