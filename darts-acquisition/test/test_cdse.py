from darts_acquisition import search_cdse_s2_mosaic


def test_manytiles():
    tile_ids = [
        "01VCK",
        "01VCL",
        "01WCM",
        "01WCN",
        "01WCP",
        "01WCQ",
        "01WCR",
        "01WCS",
        "01WCT",
        "01WCU",
        "01WCV",
        "01WDN",
        "01WDP",
        "01WDQ",
        "01WDR",
        "01WDS",
        "01WDU",
        "01WDV",
        "01WEM",
        "01WEN",
        "01WEP",
        "01WEQ",
        "01WER",
        "01WEV",
        "01WFM",
        "01WFN",
        "01WFP",
        "01WFQ",
    ]
    scene_ids = search_cdse_s2_mosaic(None, tile_ids, quarters=[3], years=range(2017, 2027))
    assert len(scene_ids) > len(tile_ids)
