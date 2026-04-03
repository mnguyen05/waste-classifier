"""
Maps each dataset category folder name to binary labels.

recycling: materials commonly accepted in municipal recycling streams
  (paper, cardboard, clean rigid containers, metal cans, glass).

trash: landfill / non-curbside recycling (film plastics, organics, textiles,
  small/problem plastics, styrofoam, etc.).

Adjust CATEGORY_TO_LABEL if your local rules differ.
"""

from typing import Literal

LabelName = Literal["recycling", "trash"]

# 1 = recycling, 0 = trash (index used in training)
CATEGORY_TO_LABEL: dict[str, LabelName] = {
    # Recycling (typical curbside)
    "aerosol_cans": "recycling",
    "aluminum_food_cans": "recycling",
    "aluminum_soda_cans": "recycling",
    "cardboard_boxes": "recycling",
    "cardboard_packaging": "recycling",
    "glass_beverage_bottles": "recycling",
    "glass_cosmetic_containers": "recycling",
    "glass_food_jars": "recycling",
    "magazines": "recycling",
    "newspaper": "recycling",
    "office_paper": "recycling",
    "plastic_detergent_bottles": "recycling",
    "plastic_food_containers": "recycling",
    "plastic_soda_bottles": "recycling",
    "plastic_water_bottles": "recycling",
    "steel_food_cans": "recycling",
    # Trash / not curbside recycling
    "clothing": "trash",
    "coffee_grounds": "trash",
    "disposable_plastic_cutlery": "trash",
    "eggshells": "trash",
    "food_waste": "trash",
    "paper_cups": "trash",
    "plastic_cup_lids": "trash",
    "plastic_shopping_bags": "trash",
    "plastic_straws": "trash",
    "plastic_trash_bags": "trash",
    "shoes": "trash",
    "styrofoam_cups": "trash",
    "styrofoam_food_containers": "trash",
    "tea_bags": "trash",
}

CLASS_NAMES = ["trash", "recycling"]  # index 0, 1


def category_to_index(category: str) -> int:
    name = CATEGORY_TO_LABEL[category]
    return 1 if name == "recycling" else 0
