from unittest.mock import patch

import numpy as np
from allensdk.brain_observatory.behavior.behavior_project_cache.project_apis.data_io.natural_movie_one_cache import (  # noqa: E501
    NaturalMovieOneCache,
)


def test_natural_movie_cache():
    """
    Test that the natural movie is loaded and processed correctly
    """
    rng = np.random.default_rng(1234)
    with patch(
        target="allensdk.brain_observatory.behavior."
        "behavior_project_cache.project_apis.data_io."
        "natural_movie_one_cache.NaturalMovieOneCache."
        "get_raw_movie",
        return_value=rng.integers(
            low=0, high=256, size=(1, 304, 608), dtype=np.uint8
        ),
    ):
        cache = NaturalMovieOneCache(
            cache_dir="fake_dir", bucket_name="fake_bucket"
        )
        movie = cache.get_processed_template_movie(n_workers=1)
        assert movie.index.name == "movie_frame_index"
        assert movie.columns.to_list() == ["unwarped", "warped"]

        unwarped = movie.loc[0, "unwarped"]
        warped = movie.loc[0, "warped"]
        assert unwarped.shape == (1200, 1920)
        assert warped.shape == (1200, 1920)
        assert unwarped.dtype == "float64"
        assert warped.dtype == "uint8"
