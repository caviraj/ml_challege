"""
Unit tests for preprocessing normalization functions:
- normalize_name
- normalize_address
- normalize_country
"""

import pytest
from src.preprocessing.normalize import (
    normalize_name,
    normalize_address,
    normalize_country,
)


class TestNormalizeCountry:
    def test_basic_country(self):
        assert normalize_country("US") == "us"
        assert normalize_country("  India  ") == "india"

    def test_unseen_country(self):
        # Open-set country string handling (e.g., France in test set)
        assert normalize_country("France") == "france"
        assert normalize_country("FRANCE") == "france"
        assert normalize_country("Germany") == "germany"

    def test_null_and_empty_country(self):
        assert normalize_country(None) == ""
        assert normalize_country("") == ""
        assert normalize_country(float("nan")) == ""


class TestNormalizeName:
    def test_legal_suffix_variants(self):
        # Suffixes should be standardized to canonical expanded form
        assert normalize_name("Acme Corp") == "acme corporation"
        assert normalize_name("Acme Corporation") == "acme corporation"
        assert normalize_name("Omega Inc") == "omega incorporated"
        assert normalize_name("Omega Incorporated") == "omega incorporated"
        assert normalize_name("Apex Pvt Ltd") == "apex private limited"
        assert normalize_name("Apex Private Limited") == "apex private limited"
        assert normalize_name("Delta Co") == "delta company"
        assert normalize_name("Delta Company") == "delta company"
        assert normalize_name("Global LLC") == "global llc"
        assert normalize_name("Tech LLP") == "tech llp"
        assert normalize_name("Euro SAS") == "euro sas"
        assert normalize_name("French SARL") == "french sarl"

    def test_ampersand_vs_and(self):
        # '&' and 'and' must produce the exact same normalized tokens
        assert normalize_name("Barnes & Noble") == normalize_name("Barnes and Noble")
        assert normalize_name("Johnson & Johnson Inc") == "johnson and johnson incorporated"
        assert normalize_name("A & B Ltd") == "a and b limited"

    def test_domain_and_url_stripping(self):
        # URLs and domain suffixes embedded in name
        assert normalize_name("SHIVSHAKTI CORP | www.shivshakti.com") == "shivshakti corporation"
        assert normalize_name("metrohealth.com LLC") == "metrohealth llc"
        assert normalize_name("Cardiology Care http://care.org") == "cardiology care"

    def test_punctuation_and_noise_stripping(self):
        assert normalize_name("<< Team Ecole >>") == "team ecole"
        assert normalize_name("-- Holloway Peak Inc Seafood") == "holloway peak incorporated seafood"
        assert normalize_name("B+ Retail Inc") == "b and retail incorporated"

    def test_unicode_preservation(self):
        # Hindi script
        assert "राम" in normalize_name("राम मार्केटिंग प्राइवेट लिमिटेड")
        # Accented characters
        assert "président" in normalize_name("Hôtel du Président SAS")

    def test_null_and_empty_name(self):
        assert normalize_name(None) == ""
        assert normalize_name("") == ""
        assert normalize_name(float("nan")) == ""


class TestNormalizeAddress:
    def test_missing_address_components(self):
        # Single token or incomplete address should not raise errors
        res = normalize_address("Nowhere")
        assert res["raw_address"] == "Nowhere"
        assert res["clean_address"] == "nowhere"
        assert res["street_tokens"] == ["nowhere"]
        assert res["city"] is None
        assert res["state"] is None
        assert res["postal_code"] is None
        assert res["raw_landmark"] is None

        # None input
        res_none = normalize_address(None)
        assert res_none["clean_address"] == ""
        assert res_none["street_tokens"] == []
        assert res_none["city"] is None
        assert res_none["state"] is None

    def test_street_abbreviation_expansion(self):
        res = normalize_address("1795 Westchester Dr, High Point, NC")
        assert "drive" in res["clean_address"]
        assert "dr" not in res["clean_address"].split()
        assert res["state"] == "nc"
        assert res["city"] == "high point"

        res2 = normalize_address("105 ELM ST, MORGANTON, NC")
        assert "street" in res2["clean_address"]
        assert res2["state"] == "nc"
        assert res2["city"] == "morganton"

    def test_landmark_extraction(self):
        addr = "Runwal Greens, Mulund Link Road, Near Fortis Hospital, Mumbai, Maharashtra"
        res = normalize_address(addr)
        assert res["raw_landmark"] is not None
        assert "near fortis hospital" in res["raw_landmark"]
        assert res["state"] == "maharashtra"
        assert res["city"] == "mumbai"

        # Hindi landmark "ke pas"
        addr_hindi = "Karauli, Rajasthan, Pani Ki Tanki Ke Pas Choubepada"
        res_hindi = normalize_address(addr_hindi)
        assert res_hindi["raw_landmark"] is not None
        assert "ke pas" in res_hindi["raw_landmark"]

    def test_postal_code_extraction(self):
        # US ZIP
        res_us = normalize_address("123 Main St, Springfield, IL 62701")
        assert res_us["postal_code"] == "62701"

        # India PIN
        res_in = normalize_address("Plot 45, Salt Lake, Kolkata, West Bengal 700091")
        assert res_in["postal_code"] == "700091"

    def test_france_address(self):
        addr = "175 Boulevard du Président Franklin Roosevelt, Bordeaux, Nouvelle-Aquitaine"
        res = normalize_address(addr)
        assert res["clean_address"] != ""
        assert res["state"] == "nouvelle aquitaine"
        assert res["city"] == "bordeaux"
