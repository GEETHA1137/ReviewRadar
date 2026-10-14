"""Versioned aspect ontology: stable ids, display labels, aliases and tag templates.

Aliases are nouns/verbs that name a feature. A few feature-naming adjectives
(sturdy, comfortable, delicious ...) are included deliberately; generic opinion
adjectives such as "fast" or "good" never trigger an aspect by themselves.
"""
from __future__ import annotations

ONTOLOGY_VERSION = "2026.10-1"

# id: (label, aliases, positive tag, negative tag, neutral tag)
_RAW: dict[str, tuple[str, list[str], str, str, str]] = {
    # --- generic ---
    "price": ("Price", ["price", "prices", "pricing", "cost", "bucks", "money"],
              "Affordable Price", "High Price", "Neutral Price Feedback"),
    "value": ("Value", ["value", "bargain", "deal", "worth"],
              "Good Value", "Poor Value", "Neutral Value Feedback"),
    "quality": ("Build Quality", ["quality", "build", "construction", "craftsmanship", "workmanship",
                                  "material", "materials", "stitching", "finish", "sturdy", "flimsy"],
                "Great Build Quality", "Poor Build Quality", "Neutral Build Quality Feedback"),
    "durability": ("Durability", ["durability", "lasted", "lasts", "lasting", "broke", "broken", "cracked",
                                  "snapped", "chipped", "fell apart", "came apart", "wore out"],
                   "Durable", "Not Durable", "Neutral Durability Feedback"),
    "delivery": ("Delivery", ["delivery", "delivered", "shipping", "shipped", "shipment", "arrived", "arrival"],
                 "Fast Delivery", "Poor Delivery", "Neutral Delivery Feedback"),
    "packaging": ("Packaging", ["packaging", "package", "packaged", "wrapped", "wrapping", "sealed"],
                  "Great Packaging", "Poor Packaging", "Neutral Packaging Feedback"),
    "service": ("Customer Service", ["customer service", "service", "seller", "vendor", "support",
                                     "refund", "warranty", "replacement"],
                "Helpful Customer Service", "Poor Customer Service", "Neutral Customer Service Feedback"),
    "accuracy": ("As Described", ["description", "described", "advertised", "pictured", "listing"],
                 "Matches Description", "Misleading Description", "Neutral Description Feedback"),
    "design": ("Design", ["design", "look", "looks", "color", "colors", "colour", "appearance", "shine", "sparkle"],
              "Attractive Design", "Unattractive Design", "Neutral Design Feedback"),
    "comfort": ("Comfort", ["comfort", "comfortable", "uncomfortable"],
                "Comfortable", "Uncomfortable", "Neutral Comfort Feedback"),
    "fit": ("Fit & Size", ["fit", "fits", "fitted", "size", "sizes", "sizing", "sized"],
            "Great Fit", "Poor Fit", "Neutral Fit Feedback"),
    "ease": ("Ease of Use", ["ease of use", "setup", "set-up", "set up", "install", "installation", "instructions"],
             "Easy To Use", "Hard To Use", "Neutral Ease of Use Feedback"),
    # --- food ---
    "taste": ("Taste", ["taste", "tastes", "flavor", "flavors", "flavour", "delicious", "tasty", "yummy", "bland"],
              "Delicious Taste", "Poor Taste", "Neutral Taste Feedback"),
    "freshness": ("Freshness", ["fresh", "freshness", "stale", "expired", "rancid"],
                  "Fresh", "Stale", "Neutral Freshness Feedback"),
    "ingredients": ("Ingredients", ["ingredient", "ingredients", "allergen", "allergens"],
                    "Good Ingredients", "Poor Ingredients", "Neutral Ingredients Feedback"),
    # --- books ---
    "story": ("Story", ["story", "stories", "plot", "storyline", "ending", "narrative", "twist", "premise"],
              "Engaging Story", "Weak Story", "Neutral Story Feedback"),
    "characters": ("Characters", ["character", "characters", "protagonist"],
                   "Strong Characters", "Weak Characters", "Neutral Characters Feedback"),
    "writing": ("Writing", ["writing", "author", "writer", "prose"],
                "Great Writing", "Poor Writing", "Neutral Writing Feedback"),
    "content": ("Content", ["content", "information", "advice", "research"],
                "Informative Content", "Weak Content", "Neutral Content Feedback"),
    "illustrations": ("Illustrations", ["illustration", "illustrations", "artwork", "pictures", "photos", "images"],
                      "Great Illustrations", "Poor Illustrations", "Neutral Illustrations Feedback"),
    "formatting": ("Formatting", ["formatting", "format", "typo", "typos", "layout"],
                   "Clean Formatting", "Poor Formatting", "Neutral Formatting Feedback"),
    "length": ("Length", ["length", "pages", "page count"],
               "Good Length", "Poor Length", "Neutral Length Feedback"),
    # --- laptop / electronics ---
    "battery": ("Battery", ["battery life", "battery", "batteries"],
                "Great Battery", "Poor Battery", "Neutral Battery Feedback"),
    "charging": ("Charging", ["charger", "charging", "adapter"],
                 "Fast Charging", "Poor Charging", "Neutral Charging Feedback"),
    "display": ("Display", ["display", "screen", "monitor", "resolution", "brightness"],
                "Great Display", "Poor Display", "Neutral Display Feedback"),
    "performance": ("Performance", ["performance", "speed", "lag"],
                    "Great Performance", "Poor Performance", "Neutral Performance Feedback"),
    "processor": ("Processor", ["processor", "cpu"],
                  "Powerful Processor", "Weak Processor", "Neutral Processor Feedback"),
    "ram": ("RAM", ["ram"], "Ample RAM", "Insufficient RAM", "Neutral RAM Feedback"),
    "storage": ("Storage", ["storage", "ssd", "hard drive", "capacity"],
                "Ample Storage", "Limited Storage", "Neutral Storage Feedback"),
    "keyboard": ("Keyboard", ["keyboard", "keys"],
                 "Great Keyboard", "Poor Keyboard", "Neutral Keyboard Feedback"),
    "trackpad": ("Trackpad", ["trackpad", "touchpad"],
                 "Great Trackpad", "Poor Trackpad", "Neutral Trackpad Feedback"),
    "camera": ("Camera", ["camera", "webcam", "lens"],
               "Great Camera", "Poor Camera", "Neutral Camera Feedback"),
    "speakers": ("Speakers", ["speaker", "speakers", "sound", "audio", "volume"],
                 "Great Speakers", "Poor Speakers", "Neutral Speakers Feedback"),
    "weight": ("Weight", ["weight", "lightweight", "heavy"],
               "Low Weight", "Heavy Build", "Neutral Weight Feedback"),
    "heating": ("Heating", ["heat", "heating", "overheats", "overheating", "fan"],
                "Runs Cool", "Overheats", "Neutral Heating Feedback"),
    "ports": ("Ports", ["port", "ports", "usb"],
              "Plenty Of Ports", "Limited Ports", "Neutral Ports Feedback"),
    "connectivity": ("Connectivity", ["wifi", "wi-fi", "bluetooth", "signal", "connection"],
                     "Reliable Connectivity", "Poor Connectivity", "Neutral Connectivity Feedback"),
    "compatibility": ("Compatibility", ["compatible", "compatibility", "drivers"],
                      "Good Compatibility", "Poor Compatibility", "Neutral Compatibility Feedback"),
}

SENTIMENTS = ("negative", "neutral", "positive")  # label ids 0, 1, 2 (matches the training protocol)


class Aspect:
    __slots__ = ("id", "label", "aliases", "tags")

    def __init__(self, id_, label, aliases, pos, neg, neu):
        self.id, self.label, self.aliases = id_, label, tuple(aliases)
        self.tags = {"positive": pos, "negative": neg, "neutral": neu}


ASPECTS: dict[str, Aspect] = {k: Aspect(k, *v) for k, v in _RAW.items()}


def alias_index() -> dict[str, str]:
    """alias -> aspect id; raises if two aspects claim the same alias."""
    idx: dict[str, str] = {}
    for a in ASPECTS.values():
        for al in a.aliases:
            key = al.lower()
            if key in idx and idx[key] != a.id:
                raise ValueError(f"alias {key!r} owned by both {idx[key]} and {a.id}")
            idx[key] = a.id
    return idx


ALIAS_INDEX = alias_index()  # import-time collision check
