"""Unit tests for Intent Classifiers."""
import pytest
import numpy as np
from src.intent_classification.baselines import MajorityClassifier, TFIDFLogisticClassifier


def test_majority_classifier():
    X_train = ["My battery died", "Battery is draining", "iOS update broke my phone"]
    y_train = ["battery_performance", "battery_performance", "ios_software_update"]

    clf = MajorityClassifier()
    clf.fit(X_train, y_train)

    assert clf.majority_intent == "battery_performance"
    preds = clf.predict(["Wi-Fi not working", "Screen cracked"])
    assert list(preds) == ["battery_performance", "battery_performance"]


def test_tfidf_logistic_classifier():
    X_train = [
        "iPhone battery is draining fast",
        "Battery percentage drops to zero",
        "iOS 11 update stuck on apple logo",
        "Cannot download latest software update",
        "AirPods bluetooth connection dropped",
        "Wi-Fi keeps disconnecting"
    ]
    y_train = [
        "battery_performance",
        "battery_performance",
        "ios_software_update",
        "ios_software_update",
        "connectivity_network_bluetooth",
        "connectivity_network_bluetooth"
    ]

    clf = TFIDFLogisticClassifier()
    clf.fit(X_train, y_train)

    pred = clf.predict(["My battery health is bad"])[0]
    assert pred == "battery_performance"
