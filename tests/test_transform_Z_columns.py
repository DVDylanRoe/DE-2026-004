from main import transform_Z_columns
import polars as pl


def test_transform_Z_columns():
    # input
    PER_90_COLS = [
        "Hdrs A per 90",
        "Clear per 90",
        "Cr A per 90",
    ]

    input_df = pl.DataFrame(
        [
            {
                "UID": "719601",
                "Hdrs A per 90": 10.031988188976378,
                "Clear per 90": 1.528051181102362,
                "Cr A per 90": 0.5314960629921259,
            },
            {
                "UID": "29179241",
                "Hdrs A per 90": 13.639751552795031,
                "Clear per 90": 1.9006211180124224,
                "Cr A per 90": 1.498136645962733,
            },
            {
                "UID": "62182055",
                "Hdrs A per 90": 10.754430379746834,
                "Clear per 90": 1.1620253164556962,
                "Cr A per 90": 0.6835443037974683,
            },
            {
                "UID": "18007344",
                "Hdrs A per 90": 10.33393269971744,
                "Clear per 90": 2.1268944258926274,
                "Cr A per 90": 0.6241972771641408,
            },
            {
                "UID": "67220143",
                "Hdrs A per 90": 11.232980332829046,
                "Clear per 90": 1.1800302571860817,
                "Cr A per 90": 1.6338880484114977,
            },
        ]
    )

    output_df = transform_Z_columns(input_df, PER_90_COLS)

    expected_df = pl.DataFrame(
        [
            {
                "UID": "719601",
                "Hdrs A per 90": 10.031988188976378,
                "Clear per 90": 1.528051181102362,
                "Cr A per 90": 0.5314960629921259,
                "Hdrs A per 90 - Z": -0.8115333853508422,
                "Clear per 90 - Z": -0.11973220833447697,
                "Cr A per 90 - Z": -0.8781980863253934,
            },
            {
                "UID": "29179241",
                "Hdrs A per 90": 13.639751552795031,
                "Clear per 90": 1.9006211180124224,
                "Cr A per 90": 1.498136645962733,
                "Hdrs A per 90 - Z": 1.6981091976598752,
                "Clear per 90 - Z": 0.7469042775229967,
                "Cr A per 90 - Z": 0.9562485070788767,
            },
            {
                "UID": "62182055",
                "Hdrs A per 90": 10.754430379746834,
                "Clear per 90": 1.1620253164556962,
                "Cr A per 90": 0.6835443037974683,
                "Hdrs A per 90 - Z": -0.3089860997101245,
                "Clear per 90 - Z": -0.9711465003762574,
                "Cr A per 90 - Z": -0.5896478409784589,
            },
            {
                "UID": "18007344",
                "Hdrs A per 90": 10.33393269971744,
                "Clear per 90": 2.1268944258926274,
                "Cr A per 90": 0.6241972771641408,
                "Hdrs A per 90 - Z": -0.6014938884532298,
                "Clear per 90 - Z": 1.2732395637540634,
                "Cr A per 90 - Z": -0.7022739325725159,
            },
            {
                "UID": "67220143",
                "Hdrs A per 90": 11.232980332829046,
                "Clear per 90": 1.1800302571860817,
                "Cr A per 90": 1.6338880484114977,
                "Hdrs A per 90 - Z": 0.023904175854320035,
                "Clear per 90 - Z": -0.9292651325663254,
                "Cr A per 90 - Z": 1.2138713527974914,
            },
        ]
    )

    assert output_df.equals(expected_df)
